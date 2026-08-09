"""Generate paper tables and figures from the machine-readable analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import discover, question_arrays


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
MODEL_LABELS = {
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini",
}
TASK_LABELS = {"truthful_qa_mc1": "TruthfulQA", "gsm8k": "GSM8K"}


def comparison_label(item: dict) -> str:
    return f"{MODEL_LABELS[item['model']]} / {TASK_LABELS[item['dataset']]}"


def write_condition_table(report: dict) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{llcrrrrrrr}",
        r"\toprule",
        r"Model & Task & Prec. & Ind. & Panel & $\beta$ & Invalid & $\kappa$ & Brier & AURC \\",
        r"\midrule",
    ]
    ordered = sorted(
        report["conditions"],
        key=lambda item: (
            list(MODEL_LABELS).index(item["model"]),
            list(TASK_LABELS).index(item["dataset"]),
            item["precision"] != "fp16",
        ),
    )
    for index, item in enumerate(ordered):
        kappa = item["kappa_same_wrong_given_all_wrong"]
        kappa_text = "--" if kappa is None else f"{100 * kappa:.1f}"
        lines.append(
            " & ".join(
                [
                    MODEL_LABELS[item["model"]],
                    TASK_LABELS[item["dataset"]],
                    item["precision"].upper(),
                    f"{100 * item['individual_accuracy']:.1f}",
                    f"{100 * item['panel_accuracy']:.1f}",
                    f"{100 * item['beta_all_agents_wrong']:.1f}",
                    f"{100 * item['invalid_answer_rate']:.1f}",
                    kappa_text,
                    f"{item['agreement_brier']:.3f}",
                    f"{item['agreement_aurc']:.3f}",
                ]
            )
            + r" \\"
        )
        if index in (3,):
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Absolute results (\%). Brier and AURC are unitless; lower is better. $\beta$ is the all-members-wrong rate and $\kappa$ is same-wrong-answer concentration conditional on co-failure. Invalid outputs are treated as distinct abstentions, not agreement.}",
            r"\label{tab:absolute}",
            r"\end{table*}",
            "",
        ]
    )
    (PAPER / "generated_results_table.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_deltas(report: dict) -> None:
    comparisons = report["paired_comparisons"]
    labels = [comparison_label(item) for item in comparisons]
    specifications = [
        ("panel_accuracy", r"$\Delta$ panel accuracy (pp)"),
        ("beta_all_agents_wrong", r"$\Delta\beta$ co-failure (pp)"),
        ("agreement_aurc", r"$\Delta$ AURC ($\times 100$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharey=True)
    y = np.arange(len(labels))
    for axis, (metric, title) in zip(axes, specifications):
        values = np.asarray([item["metrics"][metric]["delta_nf4_minus_fp16"] for item in comparisons]) * 100
        lows = np.asarray([item["metrics"][metric]["ci95_low"] for item in comparisons]) * 100
        highs = np.asarray([item["metrics"][metric]["ci95_high"] for item in comparisons]) * 100
        axis.errorbar(
            values,
            y,
            xerr=np.vstack((values - lows, highs - values)),
            fmt="o",
            color="#163a5f",
            ecolor="#4f5964",
            capsize=2.5,
            markersize=4.2,
            linewidth=1.2,
        )
        axis.axvline(0, color="#8d8d8d", linewidth=0.8, linestyle="--")
        axis.set_xlabel(title, fontsize=8)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5)
        axis.tick_params(labelsize=7.5)
    axes[0].set_yticks(y, labels, fontsize=7.5)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=0.8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "delta_summary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "delta_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_reliability() -> None:
    conditions, _ = discover(ROOT)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    for axis, (model, dataset) in zip(
        axes.ravel(),
        [(model, dataset) for model in MODEL_LABELS for dataset in TASK_LABELS],
    ):
        for precision, marker, linestyle in (("fp16", "o", "-"), ("nf4", "s", "--")):
            arrays = question_arrays(conditions[(model, dataset, precision)])
            confidence = arrays["agreement_confidence"]
            correctness = arrays["panel_accuracy"]
            levels = sorted(set(confidence.tolist()))
            accuracy = [float(correctness[confidence == level].mean()) for level in levels]
            axis.plot(levels, accuracy, marker=marker, linestyle=linestyle, label=precision.upper())
        axis.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle=":")
        axis.set_title(f"{MODEL_LABELS[model]} / {TASK_LABELS[dataset]}", fontsize=9)
        axis.grid(color="#e5e5e5", linewidth=0.5)
        axis.tick_params(labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Maximum vote share", fontsize=9)
    fig.supylabel("Panel accuracy", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "reliability.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "reliability.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / "analysis" / "paper_metrics.json").read_text(encoding="utf-8"))
    write_condition_table(report)
    plot_deltas(report)
    plot_reliability()
    print(f"Wrote paper assets under {PAPER}")


if __name__ == "__main__":
    main()
