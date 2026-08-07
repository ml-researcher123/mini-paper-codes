"""Run paired FP16/NF4 homogeneous-panel experiments on MC and free response."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".10g")
    except ValueError:
        return None


def load_examples(name: str, limit: int) -> list[dict[str, Any]]:
    if name == "truthful_qa_mc1":
        data = load_dataset("truthful_qa", "multiple_choice", split="validation")
        examples = []
        for index, row in enumerate(data.select(range(min(limit, len(data))))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            answer = chr(65 + labels.index(1))
            options = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
            examples.append(
                {
                    "id": f"truthful_qa_mc1-{index}",
                    "prompt": f"Answer the multiple-choice question. End with FINAL: <letter>.\n\n{row['question']}\n{options}",
                    "gold": answer,
                    "kind": "mc",
                    "num_choices": len(choices),
                }
            )
        return examples
    if name == "gsm8k":
        data = load_dataset("openai/gsm8k", "main", split="test")
        examples = []
        for index, row in enumerate(data.select(range(min(limit, len(data))))):
            gold = normalize_number(row["answer"].split("####")[-1])
            examples.append(
                {
                    "id": f"gsm8k-{index}",
                    "prompt": f"Solve the problem briefly. End with FINAL: <number>.\n\n{row['question']}",
                    "gold": gold,
                    "kind": "number",
                }
            )
        return examples
    raise ValueError(f"Unknown dataset: {name}")


def extract_answer(text: str, example: dict[str, Any]) -> str | None:
    final = text.rsplit("FINAL:", 1)[-1].strip() if "FINAL:" in text else text
    if example["kind"] == "mc":
        match = re.search(r"\b([A-Z])\b", final.upper())
        if not match:
            return None
        answer = match.group(1)
        return answer if ord(answer) - 65 < example["num_choices"] else None
    return normalize_number(final)


def load_model(model_name: str, precision: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def generate_one(tokenizer, model, prompt: str, seed: int, config: dict[str, Any]) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_new_tokens=config["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def summarize(rows: list[dict[str, Any]], panel_size: int) -> dict[str, Any]:
    individual = [sample["correct"] for row in rows for sample in row["samples"]]
    cofailures = 0
    same_wrong = 0
    panel_correct = 0
    confidences = []
    confidence_correct = []
    for row in rows:
        # Invalid parses are not evidence that agents selected the same answer.
        answers = [
            sample["answer"] if sample["answer"] is not None else f"__invalid_{sample['member']}"
            for sample in row["samples"]
        ]
        counts = Counter(answers)
        panel_answer, votes = counts.most_common(1)[0]
        correct = panel_answer == row["gold"]
        panel_correct += int(correct)
        confidences.append(votes / panel_size)
        confidence_correct.append(int(correct))
        all_wrong = all(not sample["correct"] for sample in row["samples"])
        cofailures += int(all_wrong)
        valid_answers = [sample["answer"] for sample in row["samples"]]
        same_wrong += int(all_wrong and None not in valid_answers and len(set(valid_answers)) == 1)
    n = len(rows)
    brier = float(np.mean([(c - y) ** 2 for c, y in zip(confidences, confidence_correct)]))
    return {
        "questions": n,
        "individual_accuracy": sum(individual) / len(individual),
        "panel_accuracy": panel_correct / n,
        "beta_all_agents_wrong": cofailures / n,
        "same_wrong_all_rate": same_wrong / n,
        "kappa_same_wrong_given_all_wrong": same_wrong / cofailures if cofailures else None,
        "agreement_brier": brier,
        "cofailure_count": cofailures,
        "same_wrong_count": same_wrong,
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def run_condition(model_name: str, precision: str, dataset_spec: dict[str, Any], config: dict[str, Any], output: Path) -> dict[str, Any]:
    condition = f"{safe_name(model_name)}__{precision}__{dataset_spec['name']}"
    path = output / f"{condition}.jsonl"
    done_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        done_ids = {row["id"] for row in rows}

    examples = load_examples(dataset_spec["name"], dataset_spec["limit"])
    tokenizer, model = load_model(model_name, precision)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        for question_index, example in enumerate(examples):
            if example["id"] in done_ids:
                continue
            samples = []
            for member in range(config["panel_size"]):
                seed = config["base_seed"] + question_index * config["panel_size"] + member
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                text = generate_one(tokenizer, model, example["prompt"], seed, config)
                answer = extract_answer(text, example)
                samples.append({"member": member, "seed": seed, "answer": answer, "correct": answer == example["gold"], "text": text})
            row = {"id": example["id"], "gold": example["gold"], "samples": samples}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)

    del model
    torch.cuda.empty_cache()
    return {"condition": condition, "model": model_name, "precision": precision, "dataset": dataset_spec["name"], **summarize(rows, config["panel_size"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "resolved_config.json", config)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    summary_name = "summary.json" if args.num_shards == 1 else f"summary_shard_{args.shard_index}.json"
    summary_path = args.output_dir / summary_name
    summaries = load_json(summary_path, []) if summary_path.exists() else []
    finished = {item["condition"] for item in summaries}
    condition_index = -1
    for model_name in config["models"]:
        for precision in config["precisions"]:
            for dataset_spec in config["datasets"]:
                condition_index += 1
                if condition_index % args.num_shards != args.shard_index:
                    continue
                condition = f"{safe_name(model_name)}__{precision}__{dataset_spec['name']}"
                if condition in finished:
                    continue
                result = run_condition(model_name, precision, dataset_spec, config, args.output_dir)
                summaries.append(result)
                atomic_json(summary_path, summaries)


def load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


if __name__ == "__main__":
    main()
