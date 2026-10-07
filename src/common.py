"""Shared helpers: paths, GPU selection, local HF model loading, batching, JSONL caching."""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "datasets"
SEED = 42

# Real open-weight models run locally (fp16 on one GPU).
MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    # Same weights as meta-llama/Llama-3.1-8B-Instruct; ungated mirror (the gated repo's
    # config/tokenizer are not downloadable with the HF account available here).
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_gpu():
    """Pin this process to the GPU with the most free memory (shared machine)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    free = [tuple(map(int, line.split(","))) for line in out.strip().splitlines()]
    gpu = str(max(free, key=lambda x: x[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu


def load_model(name):
    """Return (tokenizer, model) on GPU in fp16, left-padded for batched last-token reads."""
    pick_gpu()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.float16, device_map={"": 0}).eval()
    return tok, model


def chat_prompt(tok, system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def key(*parts):
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]
