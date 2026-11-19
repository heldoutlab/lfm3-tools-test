# Does LFM3-1.7B do better on my tool-calling benchmark? And does it lead Qwen3.5-4B at function calling?

Pre-registered test by [Held Out Lab](https://heldoutlab.substack.com): independent tests of AI model claims, with the rules fixed before the run.

**Status: done. Write-up: [*Liquid says LFM3-1.7B does better on my benchmark. 42/75 vs 37/75: too close to call.*](https://heldoutlab.substack.com/p/liquid-says-lfm3-17b-does-better)** Tasks, checks, harness and rules were fixed and hashed in [`PREREG.md`](PREREG.md) before any test arm ran (freeze commit `7d24bdb`).

**Result in brief.** All three verdicts: not resolved.

- **Part A:** LFM3-1.7B solved 42 of 75 against LFM2.5-24B-A2B's 37 (+6.7 points, 95% CI −6.7 to +20).
- **Part B, single-call:** LFM3-1.7B 27/30 vs Qwen3.5-4B 28/30 (−3.3 points, CI −16.7 to +10).
- **Part B, multi-step:** 21/30 vs 23/30 (−6.7 points, CI −23.3 to +10).

Raw outputs, logs and `report.json` are in [`results/`](results/); private-task results are reported in the post as direction only.

**Design.**

- **Part A, "Liquid's claim, as stated" (not unseen):** LFM3-1.7B at BF16 on the [LFM2.5 test's](https://github.com/heldoutlab/lfm25-tools-test) 75 public tasks (plus the 25 harder ones and the 40 private ones), against LFM2.5-24B-A2B's 37 of 75.
- **Part B, the verdict:** LFM3-1.7B vs Qwen3.5-4B, both BF16, on 60 fresh public tasks (30 single-call, 30 multi-step) and 60 private ones that are never published.

Public Part B tasks: [`items_public/tool_items.py`](items_public/tool_items.py). Part A tasks are the LFM2.5 test's, byte-identical (hashes in PREREG.md). Calibration pilot records: [`results/pilot/`](results/pilot/).

Questions or corrections: hello@heldoutlab.com. [Subscribe](https://heldoutlab.substack.com/subscribe) for the next test.

Code is MIT-licensed (see `LICENSE`); task texts and results are CC BY 4.0. Tasks, graders and harness were written with help from Claude (Anthropic).
