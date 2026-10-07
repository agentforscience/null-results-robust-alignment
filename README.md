# Null Results or Robust Alignment?

A positive-control benchmark for safety metrics. Known-misaligned responses ("anchors") and matched aligned controls are placed in three domains (overt safety violations, sycophancy, flawed reasoning), and one fixed battery of standard detectors is run over all of them to measure per-domain sensitivity. All models are real open-weight models run locally; no API models were available.

Full write-up: [REPORT.md](REPORT.md). Plan and deviations: [planning.md](planning.md).

## Key findings

- **Harm-style metrics are blind outside the overt domain.** Granite Guardian (harm) flags 100% of overt anchors, 0% of sycophancy anchors, 9% of flawed-reasoning anchors. A harm-rubric LLM judge: 76%, 1%, 3%.
- **The ">70% / <30%" hypothesis holds for those detectors, not universally.** The harm judge's 76% is not significantly above 70%. A broad "anything misaligned?" judge detects 44% of reasoning anchors, and on sycophancy flags anchors (30%) and honest controls (23%) alike.
- **The gap is fixable by rubric for sycophancy, not for reasoning.** A targeted rubric gives 100% detection at 16% false positives on sycophancy. On reasoning it flags 85% of anchors and 61% of correct solutions; it needs the reference answer to work (AUROC 0.97–0.99).
- **Detection rate without controls misleads.** Three detectors exceed 85% detection on a subtle domain while flagging 61–100% of aligned controls.
- **Judge-dependence.** A second judge (Qwen2.5-7B) reproduced the harm-rubric pattern but not the targeted-rubric results; on the planned yes/no rubrics it scored below chance.

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv sync --no-install-project          # or: uv pip install -r requirements.txt
python src/build_anchors.py                        # ~10 min on one 24GB+ GPU; uses the generation cache if present
python src/detectors.py classic                    # refusal keywords, toxic-bert, Granite Guardian
python src/detectors.py judge --model llama3.1-8b  # all LLM-judge rubrics
python src/detectors.py judge --model qwen2.5-7b
python src/analyze.py                              # tables, tests, figures (CPU only)
```

`analyze.py` alone reproduces every number and figure from the committed `datasets/anchors.jsonl` and `results/evaluations/scores.jsonl`. Models download from the Hugging Face Hub (about 37 GB). Scripts pick the GPU with the most free memory; set `CUDA_VISIBLE_DEVICES` to override.

## Files

```
planning.md            pre-registered plan + deviations
REPORT.md              full report
resources.md           datasets, models, process log
src/common.py          paths, model loading, caching
src/build_anchors.py   anchor/control generation and verification
src/detectors.py       detector battery and judge rubrics
src/analyze.py         statistics and figures
datasets/anchors.jsonl benchmark items (1,094)
results/               raw generations, scores, summary.csv, tests.csv, config.json
figures/               fig1 standard detectors, fig2 rubric ablation, fig3 score distributions
```
