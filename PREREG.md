# Pre-registration — LFM3-1.7B at tool calling

Status: **prediction and rules registered 2026-11-06; not yet frozen.** Nothing has run. The author's prediction and chosen rules are below.

Sequence: (1) this file, with the prediction and rules, is pushed before any model runs. (2) The author asks Liquid AI for its settings in the HF thread and waits up to 48 hours (until 2026-11-08 ~20:00 UTC); an answer received by then sets arms A and C's sampler, otherwise the card's tool-use setting stands. (3) Difficulty check: Gemma 4 E4B (not a test arm) on C1–C8. (4) Freeze: Part B public tasks published, every task and harness file hashed in this file, pushed. (5) Only then does any test arm (A, B, C) see a test task.

Held Out Lab. Tasks, harness and analysis: Claude (Anthropic). Prediction, rule choice and final read: the author.

## Claims under test

1. **"Our LFM3-1.7B, released today, shows better performance on your benchmark."** Maxime Labonne (Liquid AI organisation account `mlabonne`), Hugging Face discussion #12 on LiquidAI/LFM2.5-24B-A2B, in reply to this lab's LFM2.5 test. No score, checkpoint or settings were given. The author replied that LFM3-1.7B would be run at full precision on the 75 public tasks plus the 40 private ones. "Your benchmark" = github.com/heldoutlab/lfm25-tools-test, on which LFM2.5-24B-A2B solved 37 of 75 (49.3%).
2. **LFM3-1.7B "leads on … function calling"** against Qwen3.5-4B and Qwen3.5-9B (launch post, liquid.ai/blog/lfm3, 2026-10-27; model card).

### Vendor's own table (model card, read 2026-11-05)

| Benchmark | Kind | LFM3-1.7B | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|---|---|
| BFCL V4 | single-call | 71.39 | 55.12 | 62.83 |
| ToolSandbox | single/short | 80.36 | 75.40 | 76.41 |
| MCP-Atlas | multi-step | 32.57 | 41.37 | 42.67 |
| τ³-bench | multi-step | 68.01 | 65.98 | 67.11 |

All served with vLLM in BF16; "every eval uses temperature 0.6 and top_p 0.95" unless listed otherwise. The same card says the model "is not recommended for agentic coding or complex multi-step reasoning due to its small size."

## Design: two parts

**Part A — "Liquid's claim, as stated". NOT UNSEEN.** LFM3-1.7B on the LFM2.5 test's tasks, unchanged: T01–T75 (public since 2026-10-22 19:49 UTC, commit 56e1ff8), H01–H25 (the harder public supplement) and the 40 private tasks (never published). The LFM3-1.7B repo was created 2026-10-23 21:35 UTC, so the public tasks were probably not in its training data, but they have been public since, and Liquid says it ran them. Part A is reported in a section labelled *not unseen* and answers the Hugging Face thread. It is compared with LFM2.5-24B-A2B's recorded per-task results; it is not part of Part B's verdict. This is an exception to the lab's "never reuse tasks" rule, made because the claim names this benchmark.

**Part B — the verdict. Fresh tasks.** LFM3-1.7B vs Qwen3.5-4B (the rival the vendor names first, closest in size) on tasks written for this test, in two families:
- **Single-call (S):** the prompt holds everything; one write call solves it. The model must pick the right tool among near-duplicates and fill the arguments exactly (units, enums, dates, nested arrays, optional fields). This is the kind of task BFCL and ToolSandbox measure, where the vendor's table shows LFM3 ahead.
- **Multi-step (M):** 6–14 dependent calls, rules found through policy read-tools and enforced by write tools, state re-reads, refused steps that need another route. This is the kind of task MCP-Atlas measures, where the vendor's table shows LFM3 behind.

## Arms

| Arm | Model | Repo @ revision / file | Bytes | sha256 (= HF LFS oid) |
|---|---|---|---|---|
| A | LFM3-1.7B BF16 (verdict arm) | LiquidAI/LFM3-1.7B-GGUF @ cd0c74c5 / LFM3-1.7B-BF16.gguf | 3,466,626,336 | 39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68 |
| B | Qwen3.5-4B BF16 | unsloth/Qwen3.5-4B-GGUF @ e87f1764 / Qwen3.5-4B-BF16.gguf | 8,424,393,632 | 9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a |
| C | LFM3-1.7B Q4_K_M (secondary: what local users run) | LiquidAI/LFM3-1.7B-GGUF @ cd0c74c5 / LFM3-1.7B-Q4_K_M.gguf | 1,099,627,808 | 3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4 |
| G | Gemma 4 E4B Q4_K_M (calibration only, outside the comparison) | unsloth/gemma-4-E4B-it-GGUF @ bfc15c38 / gemma-4-E4B-it-Q4_K_M.gguf | 4,977,171,584 | 85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87 |

BF16 for both verdict arms: equal precision, and "full precision" as promised in the thread. Qwen publishes no GGUF of Qwen3.5-4B; unsloth's BF16 is a straight conversion. Licences: LFM Open License v1.0 (commercial terms not read; publishing scores does not redistribute weights); Qwen3.5-4B and Gemma 4 Apache-2.0. No repo is gated.

### Sampler and thinking

- **LFM3 (A, C):** the card's setting for "General (chat, tool use, agents, coding)": temperature 0.1, top_k 50, repetition penalty 1.1 (generation_config.json agrees). Reasoning on (the default). *Why not the benchmark table's 0.6 / 0.95?* The card recommends 0.1 for tool use; a user following the card gets 0.1. If Liquid answers the settings question in the thread before the freeze, its answer is used instead and recorded here.
- **Qwen3.5-4B (B):** the card's "Thinking mode for general tasks": temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence penalty 1.5, repetition penalty 1.0. Thinking on (the default).
- **Gemma 4 E4B (G):** temperature 1.0, top_p 0.95, top_k 64, thinking off (as in the KV-cache test). Calibration only.
- Every sampler field is sent with every request. Reasoning from the current tool loop is sent back (LFM3 card: "preserve the assistant's content and reasoning from the current tool-use turn"; Qwen's interleaved-thinking default).

## Runtime

Stock llama.cpp, Homebrew **0.9.0, build 12110, commit cd959bae9** (`llama-server --version`, 2026-11-06). LFM3 support arrived in ggml-org/llama.cpp#31252 (merged 2026-10-27, 19cce69), which this build includes (172 commits later). `--jinja`, the GGUF's own chat template, `-fa on -ngl 99 -np 1`, context 131,072 (Qwen3.5 card: at least 128K "to preserve thinking capabilities"). One task at a time on an otherwise idle M1 Pro, 32 GB.

**Part A crosses builds:** LFM2.5's scores came from build 10964 (0.4.1), which cannot load LFM3. The post says so.

## Tasks

| Set | Ids | Count | Published? | Used for |
|---|---|---|---|---|
| Part B public | S01–S30, M01–M30 | 60 | yes | verdict |
| Part B private | PS01–PS30, PM01–PM30 | 60 | never (hash only) | replication |
| Part B calibration | C1–C8 | 8 | yes | difficulty check (arm G), timing |
| Part A public | T01–T75 | 75 | already public | Part A |
| Part A harder | H01–H25 | 25 | already public | Part A, supporting |
| Part A private | lfm25 private T01–T40 | 40 | never | Part A, supporting |

Part B domains (new to this lab): public — veterinary clinic, warehouse pick/pack, lab sample custody, sports league, museum collections; private — five other domains, named only after publication of a later test or never. Nothing involves banks, payments or money. Every task is graded in code on the mock environment's final state; each task file self-verifies (reference passes, doing nothing fails, a plausible wrong route fails, dropping any write fails, alternative valid orders pass, fuzzed calls never crash or change state). Part A files are byte-identical copies of the frozen LFM2.5 files (hashes below).

**Difficulty check (before the freeze):** Gemma 4 E4B on C1–C8. Target 55–80% (4–6 of 8). If it scores 8/8 or ≤ 2/8, the task writer adjusts difficulty *across the whole Part B set* before any test arm sees a test task, and this file records what changed.

## Runs

| Run | Tasks | Arms | Seed |
|---|---|---|---|
| pilot | C1–C8 | G, then A, B, C (timing and template checks) | 7 |
| main | Part B public | A, B, C | 1001 |
| noise | Part B public | A, B | 2002 |
| private | Part B private | A, B, C | 1001 |
| a_main | T01–T75 | A | 1001 (same seed as LFM2.5's run) |
| a_hard | H01–H25 | A | 1001 |
| a_private | lfm25 private 40 | A | 1001 |

Order: pilot → a_main (answers the thread first) → main → private → a_hard → a_private → noise. Arm C on Part A only if time allows (descriptive).

## Scoring

Per task: **lenient** = final state correct (tool-call repairs allowed: malformed argument JSON, calls printed as text in JSON, XML or Python form, one retry after a server parse error); **strict** = correct with no repair needed. Reported per arm and family: percent solved (strict and lenient), the paired difference with a 95% bootstrap interval (10,000 resamples of tasks, seed 0), malformed calls, cap-outs (16,384-token turn limit, 40-turn limit), tool errors returned, seconds and tokens per solved task.

## Pass/fail rules — OPTIONS (the author picks one per part before anything runs)

Scores are **lenient** (parser-fair, as in the LFM2.5 test); the strict-vs-lenient gap is reported as a tooling finding.

**Part A** (gap = LFM3 BF16 − LFM2.5 on T01–T75, paired by task):
- **A1 (brief).** "Better on your benchmark" **holds** if LFM3 is ahead and the interval excludes zero. **Fails** if LFM3 is at or below 37/75. **Not resolved** otherwise.
- **A2 (looser).** Holds if LFM3 scores more than 37/75 at all (point estimate). Fails otherwise. No interval.

**Part B** (gap = LFM3 BF16 − Qwen3.5-4B BF16, public set, main run, paired by task; 30 tasks per family):
- **B1 (split, brief).** *Single-call:* "leads at function calling" **holds** if LFM3 is ahead or within 3 points; **fails** if it trails by 10+ points with the interval excluding zero; else not resolved. *Multi-step:* reported as **resolved** (either way) if the gap is 10+ points with the interval excluding zero; else not resolved. The headline verdict is the single-call result; multi-step is the second finding.
- **B2 (overall).** Same thresholds as B1's single-call, on all 60 tasks together.
- **B3 (size-honest).** "Holds" if LFM3 is within 5 points of Qwen3.5-4B overall at less than half the file size (3.47 vs 8.42 GB, so the size condition is met); fails if it trails by 10+ with the interval excluding zero; else not resolved.

**Chosen rules (author's decision, 2026-11-06): A1 and B1.** Part A: LFM3's "better on your benchmark" *holds* if LFM3 BF16 is ahead of LFM2.5 on T01–T75 and the 95% paired interval excludes zero; *fails* if LFM3 scores 37/75 or fewer; *not resolved* otherwise. Part B: the headline verdict on "leads at function calling" is the single-call family (holds if LFM3 is ahead of Qwen3.5-4B or within 3 points; fails if it trails by 10+ points with the interval excluding zero; not resolved otherwise). Multi-step is the second finding: resolved (either way) only if the gap is 10+ points with the interval excluding zero. Both on the lenient score.

The private set replicates: its direction is reported and must match for the post to call a result "replicated". It never changes the verdict.

## Noise floor

Arms A and B re-run the public set with seed 2002. Floor = the larger change in percent solved between seeds, plus the number of tasks that flipped. LFM3 runs at temperature 0.1, Qwen at 1.0, so the floor will mostly reflect Qwen.

## Sample-size limits, in plain words

With 30 tasks per family, the 95% interval on a paired gap is about ±15–18 points (if the models disagree on 20–35% of tasks). This test can show a large gap, not a BFCL-sized one, and under B1 a 10-point deficit will usually come out "not resolved". Part A has 75 paired tasks: about ±11–14 points. The post says this.

## Time and cost (estimates — UNVERIFIED until the pilot)

About 650 task-runs: Part B 60 × (A, B, C main + A, B noise) + 60 × 3 private + Part A 140 × A + pilot 32. Small models; guess 1–3 minutes each → 11–33 hours of laptop time. $0. Downloads 17.97 GB (approved 2026-11-06).

## Risks and how the design answers them

- **Wrong template or parser.** LFM3 writes `<tool_call><function=…><parameter=…>` (ChatML). Whether llama-server parses it natively is checked in the pilot; the lenient score's XML branch recovers calls left as text either way. Rendered prompts and raw replies are saved; the build is pinned.
- **Wrong settings (vendor rebuttal).** The author asks Liquid for its settings in the HF thread before the freeze. The card's tool-use sampler is the default.
- **Floor effects.** A 1.7B model may score near zero on multi-step. The single-call family and the calibration target guard against an empty result.
- **Thinking loops.** 16,384-token turn cap and 40-turn cap; cap-outs counted and reported.
- **Same author for tasks and harness.** Tasks and checks are fixed and hashed before any run; the public set and every transcript are published.
- **Part A is not unseen and crosses builds.** Labelled so; it answers the thread, it is not the verdict.

## Prediction (author writes this before anything runs)

**What do you expect to happen?** (written by the author, 2026-11-06)

Prediction: mild skepticism with a single-call upside

* Part A: LFM3 scores around 32/75. The card's warning about complex multi-step reasoning holds true, causing it to fall slightly short of LFM2.5's 37/75 due to loop accumulation and turn-limit exhaustion on longer chains.
* Part B (Comparison against Qwen3.5-4B):
   * Single-call: LFM3 edges ahead of Qwen by roughly 4 points, aligning with Liquid's internal benchmark claims.
   * Multi-step: LFM3 trails Qwen by about 35 points, suffering as the reasoning depth increases.

**What result would prove you wrong?** (written by the author, 2026-11-06)

Wrong if: LFM3 beats 37/75 on Part A, or if it trails Qwen on single-call benchmarks.

*Definitions fixed with the author before any run:* scores are lenient, arm A (BF16), main run. "Beats 37/75" = 38 or more of T01–T75. "Trails Qwen on single-call" = LFM3's single-call score (S01–S30) is below Qwen3.5-4B's, point estimate, regardless of the interval. "Loop accumulation and turn-limit exhaustion" is checked against the reported cap-out counts (40-turn limit, 16,384-token turn limit) on Part A. The prediction is not part of the pass/fail rule; it is scored in the write-up.

## Frozen hashes

(filled at the freeze: sha256 of every task file, harness file and this file's rule section)

## Deviations after freezing

(none yet)
