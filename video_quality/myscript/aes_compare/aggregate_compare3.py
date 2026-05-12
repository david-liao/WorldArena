"""Aggregate aesthetic_quality_comparison_3 results: 8 visual metrics x 10 paired episodes."""
import json
import os
import statistics
import sys

ROOTS = {
    "origin":   "/mydir/code/WorldArena/video_quality/output_aes_compare3/full/origin/generated_results.json",
    "improved": "/mydir/code/WorldArena/video_quality/output_aes_compare3/full/improved/generated_results.json",
}

HIGHER_BETTER = {
    "aesthetic_quality":      True,
    "image_quality":          True,
    "background_consistency": True,
    "subject_consistency":    True,
    "motion_smoothness":      True,
    "photometric_smoothness": True,
    "flow_score":             None,
    "dynamic_degree":         None,
}


def per_video(payload):
    items = payload[1] if isinstance(payload, list) and len(payload) >= 2 else []
    out = {}
    for it in items:
        if not isinstance(it, dict) or "video_results" not in it:
            continue
        ep = os.path.basename(os.path.dirname(os.path.dirname(it["video_path"]))).replace("ep_", "")
        out[ep] = it["video_results"]
    return out


def main(out_path=None):
    out_lines = []

    def p(line=""):
        out_lines.append(line)

    data = {k: json.load(open(v)) for k, v in ROOTS.items()}
    metrics = sorted(set(data["origin"]) & set(data["improved"]))

    p("=== group means (mean of per-video values) ===")
    p(f"{'metric':<26} {'higher?':<8} {'origin':>10} {'improved':>10} {'delta':>10}  hint")
    p("-" * 80)
    group = {}
    for m in metrics:
        o_map = per_video(data["origin"][m])
        i_map = per_video(data["improved"][m])
        o_vals = list(o_map.values())
        i_vals = list(i_map.values())
        o_mean = statistics.mean(o_vals)
        i_mean = statistics.mean(i_vals)
        delta = i_mean - o_mean
        hb = HIGHER_BETTER[m]
        if hb is True:
            hint = "improved up" if delta > 0 else "improved down"
        elif hb is False:
            hint = "improved up" if delta < 0 else "improved down"
        else:
            hint = "descriptive"
        hb_label = "higher" if hb is True else ("lower" if hb is False else "info")
        p(f"{m:<26} {hb_label:<8} {o_mean:>10.4f} {i_mean:>10.4f} {delta:>+10.4f}  {hint}")
        group[m] = (o_map, i_map)

    p()
    p("=== variance ===")
    p(f"{'metric':<26} {'origin sigma':>14} {'improved sigma':>16} {'origin med':>12} {'improved med':>14}")
    p("-" * 90)
    for m, (o_map, i_map) in group.items():
        o_vals = list(o_map.values())
        i_vals = list(i_map.values())
        p(f"{m:<26} {statistics.stdev(o_vals):>14.4f} {statistics.stdev(i_vals):>16.4f} "
          f"{statistics.median(o_vals):>12.4f} {statistics.median(i_vals):>14.4f}")

    # Paired per-episode delta
    p()
    p("=== per-episode paired delta (improved - origin) ===")
    eps = sorted(set(group[metrics[0]][0]) & set(group[metrics[0]][1]),
                 key=lambda x: int(x.replace("episode", "")))
    header = f"{'metric':<26}" + "".join(f"{e:>9}" for e in eps) + f"{'mean Δ':>10} {'wins':>6}"
    p(header)
    p("-" * len(header))
    for m, (o_map, i_map) in group.items():
        deltas = [i_map[e] - o_map[e] for e in eps]
        mean_d = statistics.mean(deltas)
        hb = HIGHER_BETTER[m]
        if hb is True:
            wins = sum(d > 0 for d in deltas)
        elif hb is False:
            wins = sum(d < 0 for d in deltas)
        else:
            wins = -1
        wins_str = f"{wins}/{len(eps)}" if wins >= 0 else "-"
        p(f"{m:<26}" + "".join(f"{d:>+9.3f}" for d in deltas) + f"{mean_d:>+10.4f} {wins_str:>6}")

    p()
    p("=== absolute per-episode values (origin / improved) ===")
    for m, (o_map, i_map) in group.items():
        p(f"\n[{m}]")
        p(f"  {'episode':<12} {'origin':>10} {'improved':>10}")
        for e in eps:
            p(f"  {e:<12} {o_map[e]:>10.4f} {i_map[e]:>10.4f}")

    text = "\n".join(out_lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
        print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out)
