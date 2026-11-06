"""Fixed settings for the LFM3-1.7B tool-calling test. Everything here is frozen by PREREG.md before the main run.

Adapted from tests/lfm25-tools/harness/config.py. Two parts:
  Part A  "Liquid's claim, as stated": LFM3-1.7B on the LFM2.5 test's tasks (75 public + 25 harder public + 40 private),
          compared with LFM2.5-24B-A2B's recorded scores. Those public tasks have been on GitHub since 2026-10-22,
          so Part A is labelled NOT UNSEEN.
  Part B  the verdict: fresh tasks, LFM3-1.7B vs Qwen3.5-4B, single-call (S) and multi-step (M) families.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
PRIVATE_RESULTS_DIR = os.path.join(RESULTS_DIR, "private")  # git-ignored: never published
MODELS_DIR = os.path.expanduser("~/models/lfm3-test")

# Task files. Part A files are byte-identical copies of the frozen lfm25-tools files (hashes in PREREG.md).
ITEM_FILES = {
    "b_public": os.path.join(ROOT, "items_public", "tool_items.py"),          # S01-S30, M01-M30, C1-C8
    "b_private": os.path.join(ROOT, "items_private", "tool_items.py"),        # PS01-PS30, PM01-PM30 (never published)
    "a_public": os.path.join(ROOT, "items_partA", "tool_items.py"),           # T01-T75 (TP* calibration unused)
    "a_hard": os.path.join(ROOT, "items_partA", "tool_items_hard.py"),        # H01-H25 (HP* calibration unused)
    "a_private": os.path.join(ROOT, "items_private", "partA_tool_items.py"),  # lfm25 private T01-T40 (never published)
}

# run name -> (task file, which ids, seed, private?)
#   ids: "test" = every non-calibration task, "calib" = calibration tasks only
RUNS = {
    "pilot":     ("b_public", "calib", 7, False),
    "main":      ("b_public", "test", 1001, False),
    "noise":     ("b_public", "test", 2002, False),
    "private":   ("b_private", "test", 1001, True),
    "a_main":    ("a_public", "test", 1001, False),
    "a_hard":    ("a_hard", "test", 1001, False),
    "a_private": ("a_private", "test", 1001, True),
}

# Part A reference scores: LFM2.5-24B-A2B Q4_K_M (arm B of tests/lfm25-tools), seed 1001, build 10964.
# Strict = lenient on every set. Source: tests/lfm25-tools/results/report.json and results/private/private/B.jsonl.
PARTA_LFM25 = {"a_main": (37, 75), "a_hard": (5, 25), "a_private": (30, 40)}
PARTA_QWEN36 = {"a_main": (68, 75), "a_hard": (16, 25), "a_private": (38, 40)}  # context only, not in any rule

# Stock llama.cpp, Homebrew 0.9.0 (build 12110, commit cd959bae9), which includes LFM3 support
# (ggml-org/llama.cpp#31252, merge 19cce69, 172 commits earlier). The LFM2.5 test used 0.4.1 (build 10964); that
# build cannot load LFM3, so Part A compares across builds (stated in PREREG.md).
SERVER_BIN = "/opt/homebrew/Cellar/llama.cpp/0.9.0/bin/llama-server"
SERVER_BUILD = "cd959bae9"

SYSTEM_PROMPT = "You are a helpful assistant"
CONTEXT = 131072  # Qwen3.5 card: "maintaining a context length of at least 128K tokens to preserve thinking"
PORT = 8793

# Every sampler field is sent explicitly with every request.
# LFM3-1.7B card, "General (chat, tool use, agents, coding)": temperature 0.1, top_k 50, repetition_penalty 1.1;
# generation_config.json says the same and nothing else. top_p 1.0 / min_p 0.0 / presence 0.0 are the "off" values.
# (The card's benchmark table used temperature 0.6, top_p 0.95 on vLLM; PREREG.md explains the choice.)
LFM3_SAMPLER = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1, "top_p": 1.0, "min_p": 0.0,
                "presence_penalty": 0.0}
# Qwen3.5-4B card, "Thinking mode for general tasks" (read 2026-11-06).
QWEN35_SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                  "presence_penalty": 1.5, "repeat_penalty": 1.0}
# Gemma 4 E4B (calibration only): the sampler used for Gemma 4 in tests/kvcache.
GEMMA4_SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0,
                  "presence_penalty": 0.0, "repeat_penalty": 1.0}

ARMS = {
    "A": {
        "label": "LFM3-1.7B BF16 (LiquidAI official, 3.47 GB)",
        "file": "LFM3-1.7B-BF16.gguf",
        "repo": "LiquidAI/LFM3-1.7B-GGUF@cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
        "size": 3466626336,
        "sha256": "39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68",
        "sampler": LFM3_SAMPLER,
        # Reasoning is on by default (card); sent explicitly so the setting is on record.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    "B": {
        "label": "Qwen3.5-4B BF16 (unsloth, 8.42 GB)",
        "file": "Qwen3.5-4B-BF16.gguf",
        "repo": "unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523",
        "size": 8424393632,
        "sha256": "9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a",
        "sampler": QWEN35_SAMPLER,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    "C": {
        "label": "LFM3-1.7B Q4_K_M (LiquidAI official, 1.10 GB) - secondary, what local users run",
        "file": "LFM3-1.7B-Q4_K_M.gguf",
        "repo": "LiquidAI/LFM3-1.7B-GGUF@cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
        "size": 1099627808,
        "sha256": "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4",
        "sampler": LFM3_SAMPLER,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    "G": {
        "label": "Gemma 4 E4B Q4_K_M (unsloth, 4.98 GB) - calibration only, pilot run only",
        "file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "repo": "unsloth/gemma-4-E4B-it-GGUF@bfc15c382204943c3a8fff0c750b94ae2364d7a3",
        "size": 4977171584,
        "sha256": "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87",
        "sampler": GEMMA4_SAMPLER,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "pilot_only": True,
    },
}

ECHO_REASONING = True  # LFM3 card: "preserve the assistant's content and reasoning from the current tool-use turn"
PARALLEL_TOOL_CALLS = True

MAX_TOKENS = 16384  # per assistant turn; a turn cut off here is recorded as truncated
MAX_TOOL_TURNS = 40
REQUEST_TIMEOUT_S = 3 * 3600


def stable_order(ids):
    """Fixed pseudo-random order that does not depend on anything run-time: sort by sha256 of the id."""
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


def family(task_id):
    """Part B family of a task id: "single" (S, PS), "multi" (M, PM), "calib" (C), or "partA"."""
    if task_id.startswith(("PS", "S")):
        return "single"
    if task_id.startswith(("PM", "M")):
        return "multi"
    if task_id.startswith("C"):
        return "calib"
    return "partA"


def is_calibration(task_id):
    return task_id.startswith(("TP", "HP")) or family(task_id) == "calib"


# ---------- pass/fail rules ----------
# The author picks the rule options in PREREG.md before anything runs. analyze.py applies only these.
RULE_A = "A1"  # author's choice, 2026-11-06
RULE_B = "B1"  # author's choice, 2026-11-06
