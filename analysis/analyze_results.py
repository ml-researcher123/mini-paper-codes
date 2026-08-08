"""Consolidate condition files and compute paired question-bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAMES = {
    "Qwen_Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "microsoft_Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
}
METRICS = (
    "individual_accuracy",
    "panel_accuracy",
    "beta_all_agents_wrong",
    "kappa_same_wrong_given_valid_all_wrong",
    "kappa_same_wrong_given_all_wrong",
    "same_wrong_all_rate",
    "agreement_brier",
    "invalid_answer_rate",
)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def discover(root: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str, str], list[tuple[Path, list[dict[str, Any]]]]] = defaultdict(list)
    for path in root.glob("results/*/*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 3 or parts[0] not in MODEL_NAMES:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        key = (MODEL_NAMES[parts[0]], parts[2], parts[1])
        candidates[key].append((path, rows))

    selected: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for key, versions in candidates.items():
        path, rows = max(versions, key=lambda item: (len(item[1]), str(item[0])))
        if len(rows) != 300:
            raise ValueError(f"Incomplete selected condition {path}: {len(rows)}/300")
        selected[key] = rows
    return selected


def question_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        samples = row["samples"]
        correct = np.array([sample["correct"] for sample in samples], dtype=float)
        answers = [
            sample["answer"] if sample["answer"] is not None else f"__invalid_{sample['member']}"
            for sample in samples
        ]
        panel_answer, votes = Counter(answers).most_common(1)[0]
        panel_correct = float(panel_answer == row["gold"])
        all_wrong = float(not correct.any())
        raw_answers = [sample["answer"] for sample in samples]
        all_valid = None not in raw_answers
        valid_all_wrong = float(bool(all_wrong) and all_valid)
        same_wrong = float(bool(valid_all_wrong) and len(set(raw_answers)) == 1)
        confidence = votes / len(samples)
        values["individual_accuracy"].append(float(correct.mean()))
        values["panel_accuracy"].append(panel_correct)
        values["beta_all_agents_wrong"].append(all_wrong)
        values["valid_all_wrong"].append(valid_all_wrong)
        values["same_wrong_all_rate"].append(same_wrong)
        values["agreement_brier"].append((confidence - panel_correct) ** 2)
        values["invalid_answer_rate"].append(sum(answer is None for answer in raw_answers) / len(samples))
    arrays = {name: np.asarray(value) for name, value in values.items()}
    arrays["kappa_same_wrong_given_all_wrong"] = arrays["same_wrong_all_rate"]
    arrays["kappa_same_wrong_given_valid_all_wrong"] = arrays["same_wrong_all_rate"]
    return arrays


def aggregate(arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, float]:
    chosen = arrays if indices is None else {key: value[indices] for key, value in arrays.items()}
    beta = chosen["beta_all_agents_wrong"].sum()
    valid_beta = chosen["valid_all_wrong"].sum()
    result = {
        key: float(chosen[key].mean())
        for key in METRICS
        if key not in ("kappa_same_wrong_given_all_wrong", "kappa_same_wrong_given_valid_all_wrong")
    }
    result["kappa_same_wrong_given_all_wrong"] = (
        float(chosen["same_wrong_all_rate"].sum() / beta) if beta else float("nan")
    )
    result["kappa_same_wrong_given_valid_all_wrong"] = (
        float(chosen["same_wrong_all_rate"].sum() / valid_beta) if valid_beta else float("nan")
    )
    return result


def paired_bootstrap(
    fp_rows: list[dict[str, Any]],
    nf_rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
    require_complete_parse: bool = False,
) -> dict[str, dict[str, float]]:
    fp_by_id = {row["id"]: row for row in fp_rows}
    nf_by_id = {row["id"]: row for row in nf_rows}
    ids = sorted(fp_by_id.keys() & nf_by_id.keys())
    if require_complete_parse:
        ids = [
            item
            for item in ids
            if all(sample["answer"] is not None for sample in fp_by_id[item]["samples"])
            and all(sample["answer"] is not None for sample in nf_by_id[item]["samples"])
        ]
    if not require_complete_parse and len(ids) != 300:
        raise ValueError(f"Expected 300 paired questions, found {len(ids)}")
    fp = question_arrays([fp_by_id[item] for item in ids])
    nf = question_arrays([nf_by_id[item] for item in ids])
    fp_point = aggregate(fp)
    nf_point = aggregate(nf)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    output: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        if metric in ("kappa_same_wrong_given_all_wrong", "kappa_same_wrong_given_valid_all_wrong"):
            denominator = "beta_all_agents_wrong" if metric == "kappa_same_wrong_given_all_wrong" else "valid_all_wrong"
            fp_values = fp["same_wrong_all_rate"][draws].sum(axis=1) / fp[denominator][draws].sum(axis=1)
            nf_values = nf["same_wrong_all_rate"][draws].sum(axis=1) / nf[denominator][draws].sum(axis=1)
        else:
            fp_values = fp[metric][draws].mean(axis=1)
            nf_values = nf[metric][draws].mean(axis=1)
        delta = nf_values - fp_values
        output[metric] = {
            "delta_nf4_minus_fp16": nf_point[metric] - fp_point[metric],
            "ci95_low": float(np.quantile(delta, 0.025)),
            "ci95_high": float(np.quantile(delta, 0.975)),
        }
    return output


def stratified_independence_null(rows: list[dict[str, Any]], indices: np.ndarray | None = None) -> float:
    selected = rows if indices is None else [rows[int(index)] for index in indices]
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for row in selected:
        if all(not sample["correct"] for sample in row["samples"]):
            answers = [sample["answer"] for sample in row["samples"]]
            if None not in answers:
                groups[str(row["gold"])].append(answers)
    total = sum(len(group) for group in groups.values())
    if not total:
        return float("nan")
    expected = 0.0
    for group in groups.values():
        answer_matrix = np.asarray(group, dtype=object)
        categories = set(answer_matrix.ravel())
        group_probability = sum(
            float(np.prod([(answer_matrix[:, member] == answer).mean() for member in range(answer_matrix.shape[1])]))
            for answer in categories
        )
        expected += len(group) / total * group_probability
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    conditions = discover(root)
    expected = {
        (model, dataset, precision)
        for model in MODEL_NAMES.values()
        for dataset in ("truthful_qa_mc1", "gsm8k")
        for precision in ("fp16", "nf4")
    }
    missing = expected - conditions.keys()
    if missing:
        raise ValueError(f"Missing complete conditions: {sorted(missing)}")

    report: dict[str, Any] = {"conditions": [], "paired_comparisons": []}
    for (model, dataset, precision), rows in sorted(conditions.items()):
        report["conditions"].append(
            {"model": model, "dataset": dataset, "precision": precision, **aggregate(question_arrays(rows))}
        )

    for pair_index, model in enumerate(MODEL_NAMES.values()):
        for dataset_index, dataset in enumerate(("truthful_qa_mc1", "gsm8k")):
            fp_rows = conditions[(model, dataset, "fp16")]
            nf_rows = conditions[(model, dataset, "nf4")]
            comparison: dict[str, Any] = {
                "model": model,
                "dataset": dataset,
                "bootstrap_replicates": args.bootstrap_replicates,
                "metrics": paired_bootstrap(
                    fp_rows,
                    nf_rows,
                    args.bootstrap_replicates,
                    args.seed + pair_index * 10 + dataset_index,
                ),
            }
            complete_ids = [
                item
                for item in {row["id"] for row in fp_rows} & {row["id"] for row in nf_rows}
                if all(sample["answer"] is not None for sample in next(row for row in fp_rows if row["id"] == item)["samples"])
                and all(sample["answer"] is not None for sample in next(row for row in nf_rows if row["id"] == item)["samples"])
            ]
            comparison["paired_complete_parse"] = {
                "questions": len(complete_ids),
                "metrics": paired_bootstrap(
                    fp_rows,
                    nf_rows,
                    args.bootstrap_replicates,
                    args.seed + 100 + pair_index * 10 + dataset_index,
                    require_complete_parse=True,
                ),
            }
            if dataset == "truthful_qa_mc1":
                fp_point = aggregate(question_arrays(fp_rows))["kappa_same_wrong_given_valid_all_wrong"]
                nf_point = aggregate(question_arrays(nf_rows))["kappa_same_wrong_given_valid_all_wrong"]
                fp_null = stratified_independence_null(fp_rows)
                nf_null = stratified_independence_null(nf_rows)
                comparison["chance_adjusted_kappa"] = {
                    "fp16_independence_null": fp_null,
                    "nf4_independence_null": nf_null,
                    "fp16_excess": fp_point - fp_null,
                    "nf4_excess": nf_point - nf_null,
                    "delta_excess_nf4_minus_fp16": (nf_point - nf_null) - (fp_point - fp_null),
                }
            report["paired_comparisons"].append(comparison)

    analysis_dir = root / "analysis"
    atomic_json(analysis_dir / "paper_metrics.json", report)
    with (analysis_dir / "paper_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["model", "dataset", "metric", "fp16", "nf4", "delta", "ci95_low", "ci95_high"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        condition_lookup = {
            (item["model"], item["dataset"], item["precision"]): item for item in report["conditions"]
        }
        for comparison in report["paired_comparisons"]:
            for metric, interval in comparison["metrics"].items():
                writer.writerow(
                    {
                        "model": comparison["model"],
                        "dataset": comparison["dataset"],
                        "metric": metric,
                        "fp16": condition_lookup[(comparison["model"], comparison["dataset"], "fp16")][metric],
                        "nf4": condition_lookup[(comparison["model"], comparison["dataset"], "nf4")][metric],
                        "delta": interval["delta_nf4_minus_fp16"],
                        "ci95_low": interval["ci95_low"],
                        "ci95_high": interval["ci95_high"],
                    }
                )
    print(f"Wrote {analysis_dir / 'paper_metrics.json'} and {analysis_dir / 'paper_metrics.csv'}")


if __name__ == "__main__":
    main()
