"""Aggregate the test-set sweeps of every dataset into one summary.

Reads the TEST_*.csv files under results/ and writes
  results/summary/test_summary.md   tables of FQI - threshold and FQI - best fixed (95% CI)
  results/summary/test_summary.png  heatmap of FQI - threshold, * = CI excludes 0

Usage (from the repo root):
    python3 -m rl.aggregate
"""
import csv
import os

# (dataset label, sweep type, file). Missing files are skipped with a note.
SOURCES = [
    ("NQ BM25", "lambda", "results/nq/TEST_bm25_5seeds.csv"),
    ("NQ e5", "lambda", "results/nq/TEST_e5_5seeds.csv"),
    ("NQ BM25", "cost", "results/nq/TEST_cost_bm25.csv"),
    ("NQ BM25", "cost_noabst", "results/nq/TEST_cost_noabst_bm25.csv"),
]
for _ds, _label in (("musique", "MuSiQue"), ("2wiki", "2Wiki"), ("hotpotqa", "HotpotQA")):
    for _kind in ("lambda", "cost", "cost_noabst"):
        SOURCES.append((_label, _kind, f"results/{_ds}/TEST_{_kind}.csv"))

TITLES = {"lambda": "λ sweep (cost ×1)",
          "cost": "Cost sweep with abstention (λ = 0.25)",
          "cost_noabst": "Cost sweep without abstention (λ = 0)"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    """{kind: {setting: {dataset: row}}} for the FQI (seed-mean) rows."""
    data = {k: {} for k in TITLES}
    for label, kind, path in SOURCES:
        if not os.path.exists(path):
            print(f"skipping missing {path}")
            continue
        for r in csv.DictReader(open(path)):
            if r["group"] != "FQI":
                continue
            setting = (f"λ={num(r['lam']):g}" if kind == "lambda"
                       else f"×{num(r.get('cost_scale')) if num(r.get('cost_scale')) is not None else 1:g}")
            data[kind].setdefault(setting, {})[label] = r
    return data


def cell(r, key, lo, hi):
    """'+0.051*' with * when the 95% CI excludes zero; '–' when not available."""
    if r is None or num(r.get(key)) is None:
        return "–"
    d, a, b = num(r[key]), num(r.get(lo)), num(r.get(hi))
    star = "*" if a is not None and b is not None and (a > 0 or b < 0) else ""
    return f"{d:+.3f}{star}"


def tables(data):
    datasets = [s[0] for s in SOURCES]
    datasets = list(dict.fromkeys(datasets))
    lines = ["# Test-set summary: FQI (seed mean)", "",
             "Each cell: reward difference; * = 95% paired-bootstrap CI excludes 0.", ""]
    wins = losses = total = 0
    for kind, title in TITLES.items():
        settings = sorted(data[kind], key=lambda s: num(s[1:].replace("=", "")) or 0)
        if not settings:
            continue
        for key, lo, hi, what in (("diff_vs_threshold", "ci_lo", "ci_hi", "FQI − threshold"),
                                  ("diff_vs_best_fixed", "bf_lo", "bf_hi", "FQI − best fixed")):
            lines += [f"## {title}: {what}", "",
                      "| setting | " + " | ".join(datasets) + " |",
                      "|---" * (len(datasets) + 1) + "|"]
            for s in settings:
                cells = [cell(data[kind][s].get(d), key, lo, hi) for d in datasets]
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
                if key == "diff_vs_threshold":
                    for c in cells:
                        if c != "–":
                            total += 1
                            wins += c.startswith("+") and c.endswith("*")
                            losses += c.startswith("-") and c.endswith("*")
            lines.append("")
    lines += [f"FQI vs threshold over all test comparisons: {wins} significant wins, "
              f"{losses} significant losses, out of {total}.", "",
              "'–' = not available (older sweep version without that column, or not run)."]
    return "\n".join(lines), datasets


def heatmap(data, datasets, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping heatmap")
        return
    kinds = [k for k in TITLES if data[k]]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.6))
    if len(kinds) == 1:
        axes = [axes]
    vmax = 0.08
    for ax, kind in zip(axes, kinds):
        settings = sorted(data[kind], key=lambda s: num(s[1:].replace("=", "")) or 0)
        grid, notes = [], []
        for s in settings:
            row, note = [], []
            for d in datasets:
                r = data[kind][s].get(d)
                v = num(r.get("diff_vs_threshold")) if r else None
                row.append(float("nan") if v is None else v)
                c = cell(r, "diff_vs_threshold", "ci_lo", "ci_hi") if r else ""
                note.append("" if c in ("", "–") else c)
            grid.append(row)
            notes.append(note)
        im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(settings)))
        ax.set_yticklabels(settings, fontsize=8)
        for i, note in enumerate(notes):
            for j, text in enumerate(note):
                dark = abs(grid[i][j]) > 0.05 if grid[i][j] == grid[i][j] else False
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")
        ax.set_title(TITLES[kind], fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="FQI − threshold (reward)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    data = load()
    text, datasets = tables(data)
    os.makedirs("results/summary", exist_ok=True)
    with open("results/summary/test_summary.md", "w") as f:
        f.write(text + "\n")
    print(text)
    print("\nsaved: results/summary/test_summary.md")
    heatmap(data, datasets, "results/summary/test_summary.png")
