#!/opt/homebrew/bin/python3.12
"""Audit the saved scores: re-run each task's frozen check on the final state saved with it and compare with the
recorded result. Read-only; it never rewrites a results file. Run it before analyze.py and quote its output.

Adapted from tests/lfm25-tools/harness/regrade.py: one task file per run (config.RUNS).
A tool task cannot be re-graded from its transcript (a different call would have changed every later turn), so the
check is the only part that can be repeated. A mismatch means a results file was edited or a check is not
deterministic; either is reported in the write-up.

  regrade.py            # every public run (pilot, main, noise, a_main, a_hard)
  regrade.py --private  # the private runs (private, a_private)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import run


def audit(path, tasks):
    mismatches, n = [], 0
    with open(path) as f:
        for line in filter(str.strip, f):
            r = json.loads(line)
            n += 1
            try:
                ok = bool(tasks[r["id"]]["check"](r["grader"]["final_state"]))
            except Exception as e:
                ok = "check_raised_%s" % type(e).__name__
            if ok != r["correct_lenient"] or r["correct"] != (ok is True and not r.get("repairs")):
                mismatches.append((r["id"], r["correct"], r["correct_lenient"], ok))
    return n, mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    clean = True
    for runname, (which, _, _, private) in sorted(cfg.RUNS.items()):
        if private != args.private:
            continue
        tasks, _, _ = run.load_set(which)
        base = cfg.PRIVATE_RESULTS_DIR if private else cfg.RESULTS_DIR
        for arm in sorted(cfg.ARMS):
            path = os.path.join(base, runname, arm + ".jsonl")
            if not os.path.exists(path):
                continue
            n, bad = audit(path, tasks)
            clean = clean and not bad
            print(runname, arm, "%d records re-checked; mismatches:" % n, bad)
    print("ALL SCORES REPRODUCE" if clean else "MISMATCHES FOUND")


if __name__ == "__main__":
    main()
