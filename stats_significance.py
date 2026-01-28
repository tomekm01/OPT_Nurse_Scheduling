# stats_significance.py

import csv
import math
from collections import defaultdict

def _rank_abs_diffs(diffs):
    # ranks for absolute diffs with average ranks on ties
    absvals = [(abs(d), i) for i, d in enumerate(diffs)]
    absvals.sort(key=lambda x: x[0])

    ranks = [0.0] * len(diffs)
    r = 1
    i = 0
    while i < len(absvals):
        j = i
        while j < len(absvals) and absvals[j][0] == absvals[i][0]:
            j += 1
        # average rank for ties in [i, j)
        avg = (r + (r + (j - i) - 1)) / 2.0
        for _, idx in absvals[i:j]:
            ranks[idx] = avg
        r += (j - i)
        i = j
    return ranks

def wilcoxon_signed_rank(x, y):
    """
    Pure-python Wilcoxon signed-rank test (two-sided, normal approximation).
    Returns p-value.
    Assumes paired samples x,y same length.
    Removes zero differences.
    """
    diffs = [a - b for a, b in zip(x, y)]
    # remove zeros
    nz = [(d, idx) for idx, d in enumerate(diffs) if d != 0]
    if len(nz) < 5:
        return 1.0  # too small, be conservative

    diffs_nz = [d for d, _ in nz]
    ranks = _rank_abs_diffs(diffs_nz)

    w_plus = 0.0
    w_minus = 0.0
    for d, rnk in zip(diffs_nz, ranks):
        if d > 0:
            w_plus += rnk
        else:
            w_minus += rnk

    W = min(w_plus, w_minus)

    # normal approximation
    n = len(diffs_nz)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2*n + 1) / 24.0)
    if sigma == 0:
        return 1.0

    z = (W - mu) / sigma
    # two-sided p from normal CDF
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return max(0.0, min(1.0, p))

def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def holm_correction(pvals):
    """
    Holm-Bonferroni correction.
    Input: list of (name, p)
    Output: list of (name, raw_p, holm_p)
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    out = [None] * m

    prev = 0.0
    for k, idx in enumerate(order):
        name, p = pvals[idx]
        adj = (m - k) * p
        if adj < prev:
            adj = prev
        prev = adj
        out[idx] = (name, p, min(1.0, adj))
    return out

def load_raw_results(csv_path):
    """
    Returns nested dict:
      data[size][algo] -> list of scores ordered by run_id (paired by run_id)
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # group by size -> run_id -> algo -> score
    tmp = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        size = r["size"]
        run_id = int(r["run_id"])
        algo = r["algo"]
        score = float(r["score"])
        tmp[size][run_id][algo] = score

    data = defaultdict(lambda: defaultdict(list))

    for size, runs in tmp.items():
        # find common algos across all runs
        all_algos = None
        for rid, amap in runs.items():
            s = set(amap.keys())
            all_algos = s if all_algos is None else (all_algos & s)
        if not all_algos:
            continue

        # build paired vectors by run_id
        for rid in sorted(runs.keys()):
            amap = runs[rid]
            for algo in sorted(all_algos):
                data[size][algo].append(float(amap[algo]))

    return data

def run_pairwise_wilcoxon_with_holm(csv_path):
    data = load_raw_results(csv_path)

    for size, algomap in data.items():
        algos = sorted(algomap.keys())
        print(f"\n=== Statistical significance for size: {size} ===")
        if len(algos) < 2:
            print("Not enough algorithms to compare.")
            continue

        # pairwise comparisons
        tests = []
        for i in range(len(algos)):
            for j in range(i+1, len(algos)):
                a, b = algos[i], algos[j]
                x = algomap[a]
                y = algomap[b]
                p = wilcoxon_signed_rank(x, y)
                tests.append((f"{a} vs {b}", p))

        corrected = holm_correction(tests)

        # print nicely
        for name, raw_p, holm_p in sorted(corrected, key=lambda t: t[2]):
            sig = "SIGNIFICANT" if holm_p < 0.05 else "not significant"
            print(f"{name:30} raw p={raw_p:.6f} | Holm p={holm_p:.6f} => {sig}")


if __name__ == "__main__":
    # Example:
    # python stats_significance.py results/tuned_raw.csv
    import sys
    if len(sys.argv) != 2:
        print("Usage: python stats_significance.py <csv_path>")
        raise SystemExit(1)


    run_pairwise_wilcoxon_with_holm(sys.argv[1])
