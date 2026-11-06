# Does LFM3-1.7B do better on my tool-calling benchmark? And does it lead Qwen3.5-4B at function calling?

Pre-registered test by [Held Out Lab](https://heldoutlab.substack.com): independent tests of AI model claims, with the rules fixed before the run.

**Status: prediction and rules registered (2026-11-06); not yet frozen; nothing has run.** [`PREREG.md`](PREREG.md) holds the two claims, the design, the author's prediction and the chosen pass/fail rules.

- **Part A, "Liquid's claim, as stated" (not unseen):** LFM3-1.7B at BF16 on the [LFM2.5 test's](https://github.com/heldoutlab/lfm25-tools-test) 75 public tasks (plus the 25 harder ones and the 40 private ones), against LFM2.5-24B-A2B's 37 of 75.
- **Part B, the verdict:** LFM3-1.7B vs Qwen3.5-4B, both BF16, on 60 fresh public tasks (30 single-call, 30 multi-step) and 60 private ones that are never published.

Next: Liquid AI is asked for its settings; a difficulty check on an outside model; then the freeze, which publishes the Part B public tasks with the sha256 of every file, before any test model sees a test task.

Questions or corrections: hello@heldoutlab.com. [Subscribe](https://heldoutlab.substack.com/subscribe) for the result.

Code is MIT-licensed (see `LICENSE`); task texts and results are CC BY 4.0. Tasks, graders and harness were written with help from Claude (Anthropic).
