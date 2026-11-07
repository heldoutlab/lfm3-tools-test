#!/opt/homebrew/bin/python3.12
"""Score the runs and apply the pre-registered rules (A1, B1) mechanically. Written before any model was run.

  analyze.py            # Part A + Part B public: prints the report, writes results/report.json
  analyze.py --private  # Part B private + Part A private: writes results/private/report_private.json only
  analyze.py --pilot    # calibration tasks C1-C8 per arm (Gemma difficulty check): writes results/pilot/report_pilot.json

Adapted from tests/lfm25-tools/harness/analyze.py. Arms: A = LFM3-1.7B BF16 (verdict arm), B = Qwen3.5-4B BF16,
C = LFM3-1.7B Q4_K_M (secondary), G = Gemma 4 E4B (pilot only). Every gap is LFM3 minus the comparison, in points,
so positive = LFM3 ahead. Scores are lenient (PREREG.md, "Scoring"); strict is reported next to them.

Intervals: 95% paired bootstrap, 10,000 resamples of tasks with replacement within the set being compared, seed 0.
Part A compares LFM3 (this test) with LFM2.5-24B-A2B's recorded per-task results from tests/lfm25-tools
(reference/lfm25_partA.json; the private 40 in results/private/reference/), paired by task id.
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

BOOT = 10000
SEED = 0

# Tasks found broken after the run are dropped for every arm and for the reference (PREREG.md); listed in
# results/dropped_items.json as {"run name": [task ids]}.
_dropped_path = os.path.join(cfg.RESULTS_DIR, "dropped_items.json")
DROPPED = json.load(open(_dropped_path)) if os.path.exists(_dropped_path) else {}

REF_PUBLIC = os.path.join(cfg.ROOT, "reference", "lfm25_partA.json")
REF_PRIVATE = os.path.join(cfg.PRIVATE_RESULTS_DIR, "reference", "lfm25_partA_private.json")


# ---------- loading ----------


def run_dir(run):
    return os.path.join(cfg.PRIVATE_RESULTS_DIR if cfg.RUNS[run][3] else cfg.RESULTS_DIR, run)


def load(run, arm):
    path = os.path.join(run_dir(run), "%s.jsonl" % arm)
    if not os.path.exists(path):
        return {}
    drop = set(DROPPED.get(run, []))
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f)) if r["id"] not in drop}


def load_reference(run, model):
    """LFM2.5 ("lfm25") or Qwen3.6 ("qwen36") per-task scores for a Part A run, as minimal records."""
    path = REF_PRIVATE if cfg.RUNS[run][3] else REF_PUBLIC
    if not os.path.exists(path):
        return {}
    table = json.load(open(path))["runs"].get(run, {}).get(model, {})
    drop = set(DROPPED.get(run, []))
    return {i: {"id": i, "correct": s, "correct_lenient": l} for i, (s, l) in table.items() if i not in drop}


# ---------- scores and intervals ----------


def score(rec, lenient=True):
    return 1.0 if (rec.get("correct_lenient") if lenient else rec["correct"]) else 0.0


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def acc(recs, ids, lenient=True):
    return pct([score(recs[i], lenient) for i in ids])


def solved(recs, ids, lenient=True):
    return int(sum(score(recs[i], lenient) for i in ids))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(p * len(v))))]


def paired_boot(x, y, ids, lenient=True, seed=SEED):
    """95% interval for accuracy(x) - accuracy(y), in points, resampling tasks with replacement."""
    rng = random.Random(seed)
    sx = [score(x[i], lenient) for i in ids]
    sy = [score(y[i], lenient) for i in ids]
    n = len(ids)
    diffs = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(100.0 * (sum(sx[k] for k in idx) - sum(sy[k] for k in idx)) / n)
    return q(diffs, 0.025), q(diffs, 0.975)


def compare(x, y, ids, lenient=True, names=("LFM3", "other")):
    """x minus y on the same tasks. Returns scores, the gap, its interval and the discordant-task counts."""
    ids = sorted(ids)
    out = {"n": len(ids), "score": "lenient" if lenient else "strict",
           names[0]: {"solved": solved(x, ids, lenient), "pct": acc(x, ids, lenient)},
           names[1]: {"solved": solved(y, ids, lenient), "pct": acc(y, ids, lenient)}}
    out["gap"] = out[names[0]]["pct"] - out[names[1]]["pct"]
    out["ci"] = paired_boot(x, y, ids, lenient) if ids else (float("nan"), float("nan"))
    out["only_%s" % names[0]] = sorted(i for i in ids if score(x[i], lenient) and not score(y[i], lenient))
    out["only_%s" % names[1]] = sorted(i for i in ids if score(y[i], lenient) and not score(x[i], lenient))
    return out


def by_env(x, y, ids, lenient=True, names=("LFM3", "other")):
    groups = {}
    for i in ids:
        groups.setdefault(x[i].get("env", "?"), []).append(i)
    return {k: {"n": len(v), names[0]: acc(x, v, lenient), names[1]: acc(y, v, lenient),
                "gap": acc(x, v, lenient) - acc(y, v, lenient)} for k, v in sorted(groups.items())}


# ---------- descriptives ----------


def describe(recs):
    ids = list(recs)
    n = max(len(ids), 1)
    repairs = {}
    for r in recs.values():
        for kind in r.get("repairs", []):
            repairs[kind] = repairs.get(kind, 0) + 1
    timings = [t.get("timings", {}) for r in recs.values() for t in r["turns"]]
    pred_n = sum(t.get("predicted_n", 0) for t in timings)
    pred_ms = sum(t.get("predicted_ms", 0.0) for t in timings)
    prompt_n = sum(t.get("prompt_n", 0) for t in timings)
    prompt_ms = sum(t.get("prompt_ms", 0.0) for t in timings)
    toks = [r["completion_tokens"] for r in recs.values()]
    secs = [r["wall_s"] for r in recs.values()]
    s_s = solved(recs, ids, False)
    s_l = solved(recs, ids, True)
    fams = {}
    for i in ids:
        fams.setdefault(cfg.family(i), []).append(i)
    return {
        "n": len(ids), "solved_strict": s_s, "solved_lenient": s_l,
        "pct_strict": acc(recs, ids, False), "pct_lenient": acc(recs, ids, True),
        "families": {k: {"n": len(v), "solved_strict": solved(recs, v, False), "solved_lenient": solved(recs, v, True),
                         "pct_strict": acc(recs, v, False), "pct_lenient": acc(recs, v, True)} for k, v in sorted(fams.items())},
        # cap-outs: the prediction's "loop accumulation and turn-limit exhaustion" is checked against these two
        "capout_token_limit_tasks": sum(1 for r in recs.values() if r.get("truncated")),
        "capout_turn_limit_tasks": sum(1 for r in recs.values() if r.get("hit_turn_limit")),
        "server_error_tasks": sum(1 for r in recs.values() if r.get("server_error")),
        "tasks_needing_repair": sum(1 for r in recs.values() if r.get("repairs")),
        "repair_events": repairs,
        "tool_errors_returned": sum(r.get("tool_errors", 0) for r in recs.values()),
        "mean_turns": sum(len(r["turns"]) for r in recs.values()) / n,
        "mean_tokens": sum(toks) / n, "median_tokens": statistics.median(toks) if toks else float("nan"),
        "mean_seconds": sum(secs) / n, "median_seconds": statistics.median(secs) if secs else float("nan"),
        "seconds_per_solved_lenient": sum(secs) / s_l if s_l else float("inf"),
        "tokens_per_solved_lenient": sum(toks) / s_l if s_l else float("inf"),
        "decode_tok_per_s": 1000.0 * pred_n / pred_ms if pred_ms else float("nan"),
        "prompt_tok_per_s": 1000.0 * prompt_n / prompt_ms if prompt_ms else float("nan"),
        "hours": sum(secs) / 3600.0,
    }


def noise_floor(main, noise):
    """Per arm: |accuracy(seed 1001) - accuracy(seed 2002)| on the re-run tasks. Floor = the larger of the two arms."""
    out = {"strict": 0.0, "lenient": 0.0, "per_arm": {}}
    for arm in ("A", "B"):
        m, nz = main.get(arm, {}), noise.get(arm, {})
        ids = [i for i in nz if i in m]
        if not ids:
            continue
        d = {"n": len(ids), "strict": abs(acc(m, ids, False) - acc(nz, ids, False)),
             "lenient": abs(acc(m, ids) - acc(nz, ids)),
             "flips_strict": sum(1 for i in ids if score(m[i], False) != score(nz[i], False)),
             "flips_lenient": sum(1 for i in ids if score(m[i]) != score(nz[i]))}
        out["per_arm"][arm] = d
        out["strict"] = max(out["strict"], d["strict"])
        out["lenient"] = max(out["lenient"], d["lenient"])
    return out


# ---------- the chosen rules (PREREG.md, "Chosen rules"): A1 and B1, lenient ----------


def rule_a1(cmp):
    """Part A. Holds if LFM3 is ahead of LFM2.5 on T01-T75 and the interval excludes zero.
    Fails if LFM3 solves 37 of 75 or fewer (LFM2.5's score). Otherwise not resolved."""
    base = cfg.PARTA_LFM25["a_main"][0]
    lfm3 = cmp["LFM3"]["solved"]
    lo, hi = cmp["ci"]
    if cmp["n"] != cfg.PARTA_LFM25["a_main"][1]:
        headline = "INCOMPLETE (%d of %d tasks)" % (cmp["n"], cfg.PARTA_LFM25["a_main"][1])
    elif cmp["gap"] > 0 and lo > 0:
        headline = "HOLDS"
    elif lfm3 <= base:
        headline = "FAILS"
    else:
        headline = "NOT RESOLVED"
    return {"rule": "A1", "claim": "LFM3-1.7B shows better performance on your benchmark",
            "lfm3_solved": lfm3, "lfm25_solved": base, "gap": cmp["gap"], "ci": cmp["ci"], "headline": headline}


def rule_b1_single(cmp):
    """Part B headline, single-call family. Holds if LFM3 is ahead of Qwen3.5-4B or within 3 points.
    Fails if it trails by 10+ points with the interval excluding zero. Otherwise not resolved."""
    gap, (lo, hi) = cmp["gap"], cmp["ci"]
    if gap >= -3:
        headline = "HOLDS"
    elif gap <= -10 and hi < 0:
        headline = "FAILS"
    else:
        headline = "NOT RESOLVED"
    return {"rule": "B1 single-call", "claim": "LFM3-1.7B leads at function calling", "gap": gap, "ci": cmp["ci"],
            "headline": headline}


def rule_b1_multi(cmp):
    """Part B second finding, multi-step family. Resolved (either way) if the gap is 10+ points and the interval
    excludes zero. Otherwise not resolved."""
    gap, (lo, hi) = cmp["gap"], cmp["ci"]
    if gap >= 10 and lo > 0:
        headline = "RESOLVED: LFM3 AHEAD"
    elif gap <= -10 and hi < 0:
        headline = "RESOLVED: LFM3 BEHIND"
    else:
        headline = "NOT RESOLVED"
    return {"rule": "B1 multi-step", "gap": gap, "ci": cmp["ci"], "headline": headline}


def direction(gap):
    return "LFM3 ahead" if gap > 0 else "LFM3 behind" if gap < 0 else "tie"


# ---------- the author's prediction (scored in the write-up; not part of any rule) ----------


def prediction_check(part_a, part_b):
    out = {"note": "Definitions in PREREG.md, 'Author's prediction'. Lenient, arm A, main run. Not a pass/fail rule."}
    if part_a and part_a.get("complete"):
        s = part_a["vs_lfm25"]["LFM3"]["solved"]
        out["part_a"] = {"predicted": "about 32/75", "actual": "%d/75" % s,
                         "wrong_if_beats_37": s >= 38,
                         "capout_turn_limit_tasks": part_a["describe"]["capout_turn_limit_tasks"],
                         "capout_token_limit_tasks": part_a["describe"]["capout_token_limit_tasks"]}
    if part_b and "single" in part_b.get("families", {}):
        f = part_b["families"]
        out["part_b"] = {"single_predicted_gap": "+4", "single_actual_gap": f["single"]["gap"],
                         "wrong_if_trails_on_single": f["single"]["LFM3"]["pct"] < f["single"]["Qwen3.5-4B"]["pct"]}
        if "multi" in f:
            out["part_b"].update(multi_predicted_gap="-35", multi_actual_gap=f["multi"]["gap"])
    return out


# ---------- the parts ----------


def part_a(run, arm="A"):
    lfm3 = load(run, arm)
    ref = load_reference(run, "lfm25")
    qwen = load_reference(run, "qwen36")
    if not lfm3 or not ref:
        return None
    ids = sorted(i for i in lfm3 if i in ref)
    names = ("LFM3", "LFM2.5")
    out = {"run": run, "arm": arm, "complete": len(ids) == len(ref), "describe": describe(lfm3),
           "vs_lfm25": compare(lfm3, ref, ids, True, names),
           "vs_lfm25_strict": compare(lfm3, ref, ids, False, names),
           "vs_lfm25_by_env": by_env(lfm3, ref, ids, True, names)}
    if qwen:
        out["vs_qwen36_context_only"] = compare(lfm3, qwen, ids, True, ("LFM3", "Qwen3.6"))
    return out


def part_b(run, comparison="B"):
    lfm3, other = load(run, "A"), load(run, comparison)
    common = sorted(i for i in lfm3 if i in other)
    if not common:
        return None
    names = ("LFM3", "Qwen3.5-4B" if comparison == "B" else "LFM3 Q4_K_M")
    out = {"run": run, "n_common": len(common), "families": {}, "families_strict": {}, "by_env": {}}
    for fam in ("single", "multi"):
        ids = [i for i in common if cfg.family(i) == fam]
        if ids:
            out["families"][fam] = compare(lfm3, other, ids, True, names)
            out["families_strict"][fam] = compare(lfm3, other, ids, False, names)
            out["by_env"][fam] = by_env(lfm3, other, ids, True, names)
    out["all"] = compare(lfm3, other, common, True, names)  # descriptive (rule B2 was not chosen)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    if args.pilot:
        report = {"arms": {}}
        for arm in sorted(cfg.ARMS):
            recs = load("pilot", arm)
            if recs:
                report["arms"][arm] = describe(recs)
        g = report["arms"].get("G")
        if g:
            report["difficulty_check"] = {"gemma_solved_lenient": g["solved_lenient"], "n": g["n"], "target": "4-6 of 8",
                                          "in_target": g["n"] == 8 and 4 <= g["solved_lenient"] <= 6,
                                          "adjust_set": g["n"] == 8 and (g["solved_lenient"] == 8 or g["solved_lenient"] <= 2)}
        path = os.path.join(cfg.RESULTS_DIR, "pilot", "report_pilot.json")
    elif args.private:
        report = {"note": "private sets: never published; direction only goes in the write-up",
                  "arms": {a: describe(r) for a in sorted(cfg.ARMS) if (r := load("private", a))}}
        pb = part_b("private")
        if pb:
            report["part_b_private"] = pb
            report["part_b_private_direction"] = {f: direction(c["gap"]) for f, c in pb["families"].items()}
        pa = part_a("a_private")
        if pa:
            report["part_a_private"] = pa
        path = os.path.join(cfg.PRIVATE_RESULTS_DIR, "report_private.json")
    else:
        main_runs = {a: load("main", a) for a in ("A", "B", "C")}
        noise = {a: load("noise", a) for a in ("A", "B")}
        report = {"rules": {"part_a": cfg.RULE_A, "part_b": cfg.RULE_B}, "score": "lenient",
                  "part_b_arms": {a: describe(r) for a, r in main_runs.items() if r}}

        pb = part_b("main")
        if pb:
            report["part_b"] = pb
            report["part_b_strict_minus_lenient_points"] = {
                a: d["pct_strict"] - d["pct_lenient"] for a, d in report["part_b_arms"].items()}
            v = {}
            if "single" in pb["families"]:
                v["single_call"] = rule_b1_single(pb["families"]["single"])
            if "multi" in pb["families"]:
                v["multi_step"] = rule_b1_multi(pb["families"]["multi"])
            report["verdict_part_b"] = v
        pc = part_b("main", "C")
        if pc:
            report["quant_effect_A_minus_C_descriptive"] = pc
        report["noise_floor"] = noise_floor(main_runs, noise)

        pa = part_a("a_main")
        if pa:
            report["part_a"] = pa
            report["verdict_part_a"] = rule_a1(pa["vs_lfm25"])
        for run in ("a_hard",):
            extra = part_a(run)
            if extra:
                report["part_a_" + run[2:]] = extra  # supporting, not in the rule
        pa_c = part_a("a_main", "C")
        if pa_c:
            report["part_a_arm_C_descriptive"] = pa_c

        # private sets: direction only (the scores stay in results/private)
        priv = {}
        pbp = part_b("private")
        if pbp:
            priv["part_b"] = {f: direction(c["gap"]) for f, c in pbp["families"].items()}
            if pb:
                priv["part_b_replicated"] = {f: direction(pb["families"][f]["gap"]) == priv["part_b"][f]
                                             for f in priv["part_b"] if f in pb["families"]}
        pap = part_a("a_private")
        if pap:
            priv["part_a"] = direction(pap["vs_lfm25"]["gap"])
            if pa:
                priv["part_a_replicated"] = direction(pa["vs_lfm25"]["gap"]) == priv["part_a"]
        if priv:
            report["private_replication_direction_only"] = priv

        report["prediction_check"] = prediction_check(pa, pb)
        path = os.path.join(cfg.RESULTS_DIR, "report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
