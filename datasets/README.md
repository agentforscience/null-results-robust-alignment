# Datasets

`anchors.jsonl` (1,094 rows) is the benchmark built by `src/build_anchors.py`. It is small and is kept in the repo.

| field | meaning |
|---|---|
| `id`, `pair` | item id; `pair` links a synthetic anchor to its matched control (null for overt) |
| `domain` | `overt_safety`, `sycophancy`, `reasoning` |
| `label` | 1 = known-misaligned anchor, 0 = aligned control |
| `request`, `response` | what every detector sees |
| `reference` | gold answer, used only by the reference-aided judge |
| `published` | (overt only) verdicts of the four judges published with JailbreakBench |

Sources (downloaded through `datasets` from the Hugging Face Hub; nothing else is needed):

- `JailbreakBench/JBB-Behaviors`, config `judge_comparison`, split `test` — 300 real model responses with 3 human labels each.
- `meg-tong/sycophancy-eval`, file `answer.jsonl` — TriviaQA questions; rows with the template "… I think the answer is {incorrect_answer} but I'm really not sure."
- `gsm8k`, config `main`, split `test` — 1,319 problems.

Rebuild: `python src/build_anchors.py` (generations are cached in `results/model_outputs/generations.jsonl`, so a rebuild from the cache needs no GPU).
