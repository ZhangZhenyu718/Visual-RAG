# -*- coding: utf-8 -*-
"""Dissertation figures 4, 5, 7 — generated straight from results/*.json.

    python docs/figures/make_figures.py     # writes PNG (300dpi) + PDF next to itself

Palette/marks follow the validated reference dataviz palette (categorical slots
in fixed order; hairline grid; muted axis ink; direct labels + legend).
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")

# --- palette (reference instance, light mode) --------------------------------
S1, S2, S3, S4 = "#2a78d6", "#1baf7a", "#eda100", "#008300"  # categorical slots 1-4
SEQ_DARK = "#104281"          # sequential blue 650 (emphasis step)
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})

METRICS = ["R@1", "R@5", "R@10", "MRR"]


def overall(fname: str, modality: str) -> dict:
    with open(os.path.join(RESULTS, fname), encoding="utf-8") as f:
        return json.load(f)["results"][modality]["overall"]


def qa_by_type(fname: str) -> tuple[float, dict]:
    with open(os.path.join(RESULTS, fname), encoding="utf-8") as f:
        d = json.load(f)
    rows = d["results"]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["qtype"], []).append(r["correct"])
    return d["accuracy"], {t: (sum(v) / len(v), len(v)) for t, v in by.items()}


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


# --- Figure 4: alpha sweep ----------------------------------------------------
def fig4():
    alphas = [0.5, 0.7, 0.8, 0.9, 1.0]
    rows = [overall("eval_val_corpus.json", "fused"),
            overall("eval_val_corpus_a0.7.json", "fused"),
            overall("eval_val_corpus_a0.8.json", "fused"),
            overall("eval_val_corpus_a0.9.json", "fused"),
            overall("eval_val_corpus.json", "visual")]  # alpha=1.0 == visual-only

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    colors = [S1, S2, S3, S4]
    for metric, c in zip(METRICS, colors):
        ys = [r[metric] for r in rows]
        ax.plot(alphas, ys, color=c, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=metric,
                clip_on=False)
        ax.annotate(metric, (alphas[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=c)
    ax.axvline(1.0, color=BASELINE, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(0.998, ax.get_ylim()[1] * 0.99, "visual-only", ha="right", va="top",
            fontsize=8, color=MUTED, rotation=90)
    ax.set_xlabel("α (visual weight in late fusion)")
    ax.set_ylabel("score")
    ax.set_xticks(alphas)
    ax.set_xlim(0.48, 1.02)
    despine(ax)
    ax.legend(loc="upper left", ncols=1)
    ax.set_title("Late fusion never exceeds visual-only retrieval "
                 "(corpus scope, tIoU ≥ 0.5)", loc="left", color=INK)
    fig.tight_layout()
    return fig


# --- Figure 5: ablation ladder -------------------------------------------------
def fig5():
    configs = [
        ("ViT-B-32\nbaseline", "eval_val_corpus.json", "visual", S1),
        ("+ decompose\n(W5)", "eval_val_corpus_decomp.json", "visual", S1),
        ("+ ViT-L rerank\n(W6)", "eval_val_corpus_rerank.json", "visual", S1),
        ("+ both", "eval_val_corpus_decomp_rerank.json", "visual", S1),
        ("SigLIP index\n(W8)", "eval_val_corpus_siglip.json", "visual", SEQ_DARK),
        ("SigLIP\n+ decompose", "eval_val_corpus_siglip_decomp.json", "visual", SEQ_DARK),
    ]
    rows = [(label.replace("\n", " "), overall(f, m), c) for label, f, m, c in configs]
    rows = rows[::-1]  # best config on top when plotted horizontally

    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.7), sharey=True)
    ys = range(len(rows))
    for ax, metric in zip(axes, METRICS):
        vals = [r[metric] for _, r, _ in rows]
        cols = [c for _, _, c in rows]
        ax.barh(ys, vals, height=0.62, color=cols)
        for y, v in zip(ys, vals):
            ax.annotate(f"{v:.3f}".lstrip("0"), (v, y), xytext=(3, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.5, color=INK2)
        ax.set_title(metric, color=INK2, fontsize=9)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_yticks(list(ys))
    axes[0].set_yticklabels([lbl for lbl, _, _ in rows], fontsize=8, color=INK2)
    fig.suptitle("Retrieval ablation ladder — corpus scope, visual modality (tIoU ≥ 0.5)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# --- Figure 7: QA accuracy by question type ------------------------------------
def fig7():
    acc_s, by_s = qa_by_type("qa_simple_150.json")
    acc_g, by_g = qa_by_type("qa_graph_150.json")
    _, by_c = qa_by_type("qa_claude_tn44.json")

    groups = ["Overall", "CW", "TN", "TC", "CH"]  # TP omitted (n=2)
    ns = {g: by_s.get(g, (0, 0))[1] for g in groups[1:]}
    series = [
        ("Simple agent · DeepSeek (text-only)", S1,
         {"Overall": acc_s, **{g: by_s[g][0] for g in groups[1:]}}),
        ("Graph agent · DeepSeek (text-only)", S2,
         {"Overall": acc_g, **{g: by_g[g][0] for g in groups[1:]}}),
        ("Simple agent · Claude (multimodal)", S3,
         {"TN": by_c["TN"][0]}),
    ]

    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    width = 0.26
    for i, (label, color, vals) in enumerate(series):
        xs = [j + (i - 1) * width for j, g in enumerate(groups) if g in vals]
        ys = [vals[g] for g in groups if g in vals]
        bars = ax.bar(xs, ys, width=width * 0.92, color=color, label=label)
        for b, v in zip(bars, ys):
            ax.annotate(f"{v:.2f}".lstrip("0"), (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    ax.axhline(0.20, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate("random (0.20)", (len(groups) - 0.55, 0.20), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
    labels = ["Overall\n(n=150)"] + [f"{g}\n(n={ns[g]})" for g in groups[1:]]
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, color=INK2)
    ax.set_ylabel("5-choice accuracy")
    ax.set_ylim(0, 0.9)
    despine(ax)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", ncols=1)
    ax.set_title("Agent QA accuracy by question type — NExT-QA val",
                 loc="left", color=INK)
    fig.tight_layout()
    return fig


# --- Figure 8: white-dog qualitative case (needs artifacts/frames on disk) ----
def fig8():
    import matplotlib.image as mpimg
    frames_dir = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                              "artifacts", "frames", "2834146886")
    shots = [("00029.00.jpg", "t = 29 s · anchor"),
             ("00033.00.jpg", "t = 33 s · leans down"),
             ("00037.00.jpg", "t = 37 s · noses the puppy")]

    fig = plt.figure(figsize=(6.3, 4.3))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.3, 1.0, 1.0], hspace=0.34, wspace=0.06,
                          top=0.86, bottom=0.02, left=0.02, right=0.98)

    for i, (fn, cap) in enumerate(shots):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mpimg.imread(os.path.join(frames_dir, fn)))
        ax.set_axis_off()
        ax.set_title(cap, fontsize=7.6, color=INK2, pad=4)

    def answer_panel(row, header, hcolor, body):
        ax = fig.add_subplot(gs[row, :])
        ax.set_axis_off()
        ax.text(0.006, 0.94, header, fontsize=8.5, fontweight="bold", color=hcolor,
                va="top", transform=ax.transAxes)
        ax.text(0.006, 0.60, body, fontsize=8, color=INK, va="top", wrap=True,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=SURFACE,
                          edgecolor=hcolor, linewidth=1))

    answer_panel(1, "Text-only LLM — evidence starved", "#e34948",
                 '"…after going to the cushion, the video simply continues in silence — no '
                 'further spoken action is described. To know exactly what the dog physically '
                 'does (e.g., lies down, sniffs, sits), the keyframe images would need to be '
                 'inspected."')
    answer_panel(2, 'Multimodal LLM — matches ground truth ("smells the black dog")',
                 "#008300",
                 '"…the white dog leans down and sniffs/nuzzles the small black puppy that is '
                 'lying in the cushion [@ 28–36 s], then pushes its nose into the bedding '
                 'toward the puppy [@ 36–44 s]."')

    fig.suptitle('"What does the white dog do after going to the cushion?"\n'
                 "Same retrieval, same tools — only the evidence channel differs",
                 x=0.02, y=0.985, ha="left", fontsize=9.5, color=INK)
    return fig


if __name__ == "__main__":
    for name, fn in [("fig4_alpha_sweep", fig4),
                     ("fig5_ablation_ladder", fig5),
                     ("fig7_qa_by_type", fig7),
                     ("fig8_whitedog_case", fig8)]:
        f = fn()
        f.savefig(os.path.join(HERE, f"{name}.png"), dpi=300)
        f.savefig(os.path.join(HERE, f"{name}.pdf"))
        plt.close(f)
        print("wrote", name)
