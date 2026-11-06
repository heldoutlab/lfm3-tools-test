#!/opt/homebrew/bin/python3.12
"""Resumable ranged downloader for the LFM3 test model files (same method as tests/bonsai2/fetch.py):
each 32 MB segment is written at its own offset and recorded only when complete; sha256 verified at the end."""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

MODELS_DIR = os.path.expanduser("~/models/lfm3-test")
# (repo, revision, file, sha256) — revisions and hashes read from the HF API on 2026-11-06
FILES = [
    ("LiquidAI/LFM3-1.7B-GGUF", "cd0c74c575b5c416f3011b5b4c298077f26ca8c2", "LFM3-1.7B-BF16.gguf",
     "39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68"),
    ("LiquidAI/LFM3-1.7B-GGUF", "cd0c74c575b5c416f3011b5b4c298077f26ca8c2", "LFM3-1.7B-Q4_K_M.gguf",
     "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"),
    ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-BF16.gguf",
     "9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a"),
    ("unsloth/gemma-4-E4B-it-GGUF", "bfc15c382204943c3a8fff0c750b94ae2364d7a3", "gemma-4-E4B-it-Q4_K_M.gguf",
     "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87"),
]
SEG = 32 << 20
THREADS = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[1])


def fetch(repo, rev, name, want):
    url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, rev, name)
    path = os.path.join(MODELS_DIR, name)
    state_path = path + ".segments.json"
    if os.path.exists(path) and not os.path.exists(state_path):
        if sha256(path) == want:
            print("OK   %s (already present)" % name, flush=True)
            return True
        os.remove(path)
    size = None
    while size is None:
        try:
            size = remote_size(url)
        except Exception as e:
            print("size lookup failed (%s); retrying" % e, flush=True)
            time.sleep(10)
    done = set(json.load(open(state_path))) if os.path.exists(state_path) and os.path.exists(path) else set()
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    todo = [i for i in range((size + SEG - 1) // SEG) if i not in done]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            start, end = i * SEG, min(size, (i + 1) * SEG) - 1
            while True:
                try:
                    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        if r.status != 206:
                            raise IOError("status %s" % r.status)
                        buf = r.read()
                    if len(buf) != end - start + 1:
                        raise IOError("short read")
                    os.pwrite(fd, buf, start)
                    break
                except Exception:
                    time.sleep(5)
            with lock:
                done.add(i)
                with open(state_path + ".tmp", "w") as f:
                    json.dump(sorted(done), f)
                os.replace(state_path + ".tmp", state_path)
                if len(done) % 32 == 0:
                    print("%s  %.1f / %.1f GB" % (name, len(done) * SEG / 1e9, size / 1e9), flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    os.fsync(fd)
    os.close(fd)
    ok = sha256(path) == want
    if ok:
        os.remove(state_path)
    print(("OK   " if ok else "BAD  ") + name, flush=True)
    return ok


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("ALL OK" if all([fetch(*f) for f in FILES]) else "SOME FAILED", flush=True)
