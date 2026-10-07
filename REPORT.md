# Null Results or Robust Alignment? Cross-Domain Sensitivity of Standard Safety Metrics on Synthetic Misalignment Anchors

## 1. Executive Summary

**Question.** When a standard safety metric reports "nothing found" in a domain, would it have fired if misalignment were present? I planted known-misaligned responses ("anchors") with matched aligned controls in three domains and ran one fixed battery of standard detectors over all of them.

**Answer.** The hypothesis holds for the purpose-built harm detectors and only partly otherwise.

- The two purpose-built harm detectors catch overt violations and are blind to the subtle anchors. Granite Guardian (harm) flags 100% of overt anchors, 0% of sycophancy anchors and 9% of flawed-reasoning anchors. An LLM judge with the JailbreakBench harm rubric flags 76%, 1% and 3%.
- The ">70% overt" half is only statistically established for Granite Guardian. The harm judge's 76% has a 95% interval of 67–83%, which straddles 70%.
- The "<30% subtle" half fails for a broad "is anything misaligned here?" judge on reasoning (44% detected). On sycophancy that judge flags 30% of anchors but also 23% of honest controls, so it does not discriminate (AUROC 0.50).
- Whether the gap is "just an evaluation gap" depends on the domain. For sycophancy, a targeted rubric alone fixes it (100% detected, 16% false positives, AUROC 1.00). For flawed reasoning it does not: the targeted judge flags 85% of anchors but also 61% of correct solutions, and only becomes reliable when given the reference answer (AUROC 0.97–0.99).
- A second judge (Qwen2.5-7B) did not reproduce the targeted-rubric results; it misread the answer labels or flagged nearly everything. The "a targeted rubric fixes it" finding therefore rests on one judge model.

**Implication.** A scorecard built from harm-style metrics would report roughly 0–3% problems on a set of responses that are all sycophantic, and 3–9% on a set that all contain wrong reasoning. A cross-domain null from such metrics is not evidence of robustness. Each metric needs a per-domain positive control before its nulls are read.

**Scope caveats that matter most.** The overt domain uses real human-labelled data while the subtle domains use prompted synthetic anchors; all judges are small (7–8B) and answer in a single token; no API models were available.

## 2. Research Question & Hypothesis

> Standard alignment metrics show strong domain-dependent sensitivity, achieving detection rates above 70% for overt safety violations but below 30% for subtle misalignment in reasoning and sycophancy domains, indicating that cross-domain null results reflect evaluation gaps rather than alignment robustness.

Decomposed (fixed in `planning.md` before running):

- **H1** detection rate (TPR) on overt anchors > 0.70
- **H2** TPR on sycophancy anchors < 0.30
- **H3** TPR on flawed-reasoning anchors < 0.30
- **H4** the gap is an *evaluation* gap: a judge with a domain-appropriate rubric, and/or a reference answer, does detect the same anchors.

The hypothesis is judged primarily on the two purpose-built safety detectors (Granite Guardian, harm-rubric judge).

## 3. Literature Context

No literature review was supplied (the resource-finder stage was skipped), so this section is from my knowledge of the papers, not a systematic search.

- **Judge validation is single-domain.** JailbreakBench (Chao et al., 2024), HarmBench (Mazeika et al., 2024), StrongREJECT (Souly et al., 2024) and Llama Guard (Inan et al., 2023) measure agreement with humans on harmful compliance only.
- **Subtle behaviours get bespoke instruments.** Sycophancy (Perez et al., 2022; Sharma et al., 2023) and reasoning faithfulness (Turpin et al., 2023; Lanham et al., 2023) are measured with task-specific protocols; these papers do not ask what general safety metrics say about the same outputs.
- **Planted misalignment exists, but for auditing.** Sleeper agents (Hubinger et al., 2024) and related model-organism work plant known misbehaviour to test auditors, not to calibrate the per-domain sensitivity of off-the-shelf metrics.
- **Refusal-string matching** (Zou et al., 2023) is known to be a weak proxy; it is included as the classic baseline.

Gap addressed: a positive-control analysis that holds the metric fixed and varies the domain.

## 4. Methodology

### 4.1 Anchors and controls

| Domain | Source | Anchor (label 1) | Control (label 0) | Ground truth | n anchor / control |
|---|---|---|---|---|---|
| Overt safety | JailbreakBench `judge_comparison`: 300 real target-model responses | human-majority "jailbroken" | human-majority safe | 3 human annotators | 110 / 190 |
| Sycophancy | sycophancy-eval `answer.jsonl` (TriviaQA; user says "I think the answer is *X* but I'm really not sure", *X* wrong) | Qwen2.5-7B told to affirm the user's answer | same model told the correct answer and asked to correct politely | string check | 174 / 174 (pairs) |
| Flawed reasoning | GSM8K test | Qwen2.5-7B told to write a confident solution with one unacknowledged slip | same model solving normally | parsed final answer vs gold | 223 / 223 (pairs) |

Verification rules (programmatic, independent of every detector):

- *Sycophancy anchor*: contains the wrong answer, contains no correct alias, no negation/contrast words, no leak words. *Control*: contains a correct alias. Yield: 174 of 300 pairs.
- *Reasoning anchor*: final answer ≠ gold, no leak words ("mistake", "misread", …), and the working never reaches the gold number (so the error is inside the derivation, not a tacked-on wrong last line). *Control*: final answer = gold. Yield: 223 of 1,319 pairs.

Generation: temperature 0.7, top-p 0.9, seed 42. Full prompts are in `src/build_anchors.py`. Mean response length is matched within pairs (sycophancy 197 vs 188 characters; reasoning 165 vs 167).

Example anchors:

> **Sycophancy.** *User:* "Which English novelist wrote 'The Rachel Papers' and 'Money'? I think the answer is Ian McEwan but I'm really not sure." *Anchor:* "You are absolutely right; Ian McEwan is indeed the author of both…" (correct: Martin Amis)

> **Reasoning.** *Problem:* 4 rows × 18 seats; a quarter taken by administrators; a third of the rest by parents; how many students? *Anchor:* "… Step 4: Occupied by parents = 54 / 3 = 19. Step 5: Students = 54 − 19 = 35. Answer: 35" (correct: 36)

### 4.2 Detectors

Every detector sees the same input in every domain: the user request and the response.

| # | Detector | Flag rule |
|---|---|---|
| 1 | Refusal-keyword matching (GCG/AdvBench list) | no refusal string present |
| 2 | `unitary/toxic-bert` | max toxicity score ≥ 0.5 |
| 3 | `ibm-granite/granite-guardian-3.1-2b`, default "harm" risk | P(Yes) ≥ 0.5 |
| 4 | LLM judge, JailbreakBench harm rubric (safe/unsafe) | P(unsafe) ≥ 0.5 |
| 5 | LLM judge, broad rubric: "any misaligned or problematic behaviour?" (names no domain) | P(bad) ≥ 0.5 |
| 6 | LLM judge, domain-targeted rubric (sycophancy audit / solution verification), no reference | P(yes) ≥ 0.5 |
| 7 | LLM judge, targeted rubric + reference answer (oracle upper bound) | P(yes) ≥ 0.5 |

Detectors 1–5 are the "standard" battery. Detectors 6–7 are the rubric ablation for H4 and apply to the two subtle domains only.

LLM judges: primary Llama-3.1-8B-Instruct; second judge Qwen2.5-7B-Instruct (also the anchor generator). The score is the first-token probability of the positive word, normalised over the two answer words, from a single forward pass (no sampling). Rubric texts are in `src/detectors.py`.

### 4.3 Metrics and tests

- **Detection rate** = TPR on anchors, Wilson 95% CI. **FPR** on controls. **Youden's J** = TPR − FPR.
- **AUROC** with bootstrap CI (2,000 resamples) and **TPR at 5% FPR** with the threshold re-set on each domain's own controls (best-case recalibration).
- Tests: exact binomial for H1–H3; Fisher exact for overt vs subtle TPR; exact McNemar for anchor vs matched control. Holm correction within each family; α = 0.05.

### 4.4 Environment and cost

- Python 3.10, torch 2.14.1+cu130, transformers 5.19.0; one shared RTX A6000 (48 GB), fp16.
- Batch sizes: 24 for generation, 16 for judging. GPU time: roughly 15 minutes of generation and 10 minutes of detection in total.
- API cost: $0. No API key was available, so no API models were used.
- Determinism check: re-scoring 160 random items with a different batch size gave identical flags (max score difference 0.015).

### 4.5 Deviations from the plan

Listed in full at the end of `planning.md`. The ones that affect interpretation:

- The anchor prompts and the reasoning-anchor filter were revised after reading pilot samples, before any detector was run.
- One rubric variant with self-describing answer words (`…2` rows in §5.3) was added after seeing the second judge's results. It is exploratory.

## 5. Results

### 5.1 Standard detectors (primary judge: Llama-3.1-8B)

![Detection rate by detector and domain](figures/fig1_standard_detectors.png)

| Detector | Domain | TPR [95% CI] | FPR | J | AUROC [95% CI] | TPR @ 5% FPR |
|---|---|---|---|---|---|---|
| Refusal keyword | overt | 90.9% [84.1, 95.0] | 64.2% | +0.27 | — | — |
| | sycophancy | 100% [97.8, 100] | 100% | 0.00 | — | — |
| | reasoning | 100% [98.3, 100] | 100% | 0.00 | — | — |
| toxic-bert | overt | 0.0% [0.0, 3.4] | 0.0% | 0.00 | 0.56 [0.49, 0.62] | 6% |
| | sycophancy | 0.0% [0.0, 2.2] | 0.0% | 0.00 | 0.38 [0.32, 0.44] | 2% |
| | reasoning | 0.0% [0.0, 1.7] | 0.0% | 0.00 | 0.37 [0.31, 0.42] | 2% |
| **Granite Guardian (harm)** | overt | **100% [96.6, 100]** | 19.5% | +0.81 | 0.96 [0.94, 0.98] | 77% |
| | sycophancy | **0.0% [0.0, 2.2]** | 0.0% | 0.00 | 0.24 [0.19, 0.29] | 1% |
| | reasoning | **9.4% [6.2, 14.0]** | 0.4% | +0.09 | 0.74 [0.70, 0.79] | 36% |
| **LLM judge, harm rubric** | overt | **75.5% [66.6, 82.5]** | 8.4% | +0.67 | 0.95 [0.92, 0.97] | 62% |
| | sycophancy | **1.1% [0.3, 4.1]** | 0.0% | +0.01 | 0.42 [0.36, 0.48] | 13% |
| | reasoning | **2.7% [1.2, 5.7]** | 0.4% | +0.02 | 0.70 [0.65, 0.74] | 22% |
| LLM judge, broad rubric | overt | 98.2% [93.6, 99.5] | 21.6% | +0.77 | 0.95 [0.92, 0.97] | 64% |
| | sycophancy | 29.9% [23.6, 37.1] | 23.0% | +0.07 | 0.50 [0.44, 0.56] | 11% |
| | reasoning | 43.9% [37.6, 50.5] | 8.5% | +0.35 | 0.74 [0.69, 0.79] | 37% |

Pre-specified tests (Holm-corrected, 25 tests):

| Detector | H1: overt > 70% | H2: sycophancy < 30% | H3: reasoning < 30% | Overt vs subtle (Fisher) |
|---|---|---|---|---|
| Granite Guardian | supported (p ≈ 1e-16) | supported (p ≈ 2e-26) | supported (p ≈ 1e-12) | both p < 1e-60 |
| Harm-rubric judge | **not established** (75.5%, p = 0.13 uncorrected) | supported (p ≈ 5e-23) | supported (p ≈ 5e-25) | both p < 1e-40 |
| Broad-rubric judge | supported (p ≈ 1e-13) | **not supported** (29.9%, p = 0.52) | **not supported** (43.9%) | both p < 1e-24 |
| Refusal keyword | TPR 91% but FPR 64% | not supported (flags 100%) | not supported (flags 100%) | — |
| toxic-bert | **not supported** (0%) | trivially (flags nothing) | trivially (flags nothing) | no difference |

Sanity check on the overt domain: accuracy against the human-majority label is 0.857 for my harm-rubric judge and 0.877 for Granite Guardian, against 0.907 for the Llama-3-70B judge and 0.903 for the GPT-4 judge published with JailbreakBench. The local detectors are somewhat weaker than the published ones but in the same range.

### 5.2 Rubric ablation (H4)

![Rubric ablation](figures/fig2_rubric_ablation.png)

Primary judge (Llama-3.1-8B):

| Rubric | Sycophancy TPR / FPR / AUROC | Reasoning TPR / FPR / AUROC |
|---|---|---|
| Harm (JBB) | 1.1% / 0.0% / 0.42 | 2.7% / 0.4% / 0.70 |
| Broad "misaligned?" | 29.9% / 23.0% / 0.50 | 43.9% / 8.5% / 0.74 |
| Domain-targeted | **100% / 16.1% / 1.00** | 85.2% / **61.0%** / 0.72 |
| Targeted + reference | 100% / 26.4% / 1.00 | 99.6% / 52.0% / 0.97 |

- **Sycophancy:** naming the behaviour is enough. The targeted judge separates anchors from controls almost perfectly (98% TPR at 5% FPR).
- **Reasoning:** naming the behaviour is not enough. The targeted judge says "contains an error" for 61% of correct solutions, and its AUROC (0.72) is no better than the broad rubric's (0.74). With the reference answer it ranks well (AUROC 0.97) but its default threshold is badly calibrated (52% FPR).

### 5.3 Second judge (Qwen2.5-7B) and the post-hoc rubric variant

| Rubric @ judge | Sycophancy TPR / FPR / AUROC | Reasoning TPR / FPR / AUROC |
|---|---|---|
| Harm @ Qwen | 8.0% / 31.0% / 0.15 | 20.6% / 2.2% / 0.74 |
| Broad @ Qwen | 44.3% / 82.2% / 0.27 | 55.6% / 18.4% / 0.76 |
| Targeted @ Qwen | 94.8% / 100% / 0.10 | 39.5% / 56.5% / 0.40 |
| Reference @ Qwen | 98.3% / 99.4% / 0.33 | 24.7% / 72.6% / 0.18 |
| *Post-hoc* targeted2 @ Qwen | 94.8% / 97.7% / 0.58 | 67.3% / 20.2% / 0.79 |
| *Post-hoc* reference2 @ Qwen | 98.9% / 92.0% / 0.86 | 95.1% / 3.6% / 0.99 |
| *Post-hoc* targeted2 @ Llama | 99.4% / 16.7% / 0.99 | 93.7% / 80.7% / 0.73 |
| *Post-hoc* reference2 @ Llama | 100% / 35.1% / 1.00 | 100% / 24.7% / 0.99 |

What replicates with the second judge:

- Harm rubric: high on overt (96%), low on both subtle domains (8% and 21%). Direction replicates; the reasoning figure is higher than with Llama.
- Broad rubric on reasoning: again above 30% (56%).

What does not replicate:

- On the planned yes/no rubrics Qwen scores below chance. I checked this is not a scoring bug: batched scores, unbatched scores and the model's free-running greedy output agree. On reasoning it answers "yes" to correct solutions (reading "yes" as "is correct"); on sycophancy it answers "yes" to nearly everything.
- With self-describing answer words (post-hoc), Qwen's reasoning verdicts become sensible and, with the reference, excellent (95% TPR, 4% FPR). Its sycophancy verdicts still flag 92–98% of honest controls.

Caveat on this table: Qwen2.5-7B also wrote the subtle anchors and controls, so it is judging its own text.

Minor data-quality note: on the post-hoc reasoning rubric, Llama put most first-token probability outside the two answer words for 4.5% (targeted2) and 0.9% (reference2) of items. For all planned rubrics this was ≤ 0.2%.

### 5.4 Score distributions

![Score distributions](figures/fig3_score_distributions.png)

Harm-detector scores for sycophancy anchors and controls are both piled at zero; there is no threshold that would rescue them. Granite Guardian's reasoning scores do shift upward for anchors without crossing 0.5.

### 5.5 Output files

| File | Content |
|---|---|
| `datasets/anchors.jsonl` | the 1,094 benchmark items |
| `results/model_outputs/generations.jsonl` | all 3,238 raw generations, including those filtered out |
| `results/evaluations/scores.jsonl` | every detector score and flag |
| `results/summary.csv`, `results/tests.csv` | the tables above, all detectors |
| `results/evaluations/error_examples.md` | misses and false alarms (first by item order, not hand-picked) |
| `results/anchor_stats.json`, `results/config.json` | yields, versions, parameters |

## 6. Analysis & Discussion

**Does the evidence support the hypothesis?**

- *Domain-dependent sensitivity:* yes, strongly, for harm-style metrics. The overt-minus-subtle gap in detection rate is 73–100 percentage points for the two primary detectors.
- *">70% overt":* yes for Granite Guardian and the broad judge; borderline for the harm judge; false for toxic-bert, which flags none of the 110 human-confirmed jailbreaks. Toxicity is not harmful compliance.
- *"<30% subtle":* yes for harm-style metrics (0–9%). No for a broad misalignment rubric on reasoning (44–56%).
- *"Evaluation gap rather than robustness":* supported in the sense that a harm-metric null carries almost no information about these domains. The stronger reading, that a better rubric recovers the signal, holds for sycophancy with one judge and does not hold for reasoning without a reference answer.

**Two different kinds of gap.** The sycophancy miss is a *scope* gap: the judge has the knowledge (it scores AUROC 1.00 when asked the right question) but the harm rubric never asks. The reasoning miss is partly a *capability* gap: a single-token 8B judge cannot verify arithmetic, so asking the right question produces indiscriminate flagging. This distinction changes the remedy: a rubric for the first, a reference answer or a stronger verifier for the second.

**High TPR is not detection.** Three detectors reach ≥85% TPR on a subtle domain while being useless: refusal matching (100% TPR, 100% FPR), the targeted reasoning judge (85% / 61%) and Qwen's targeted sycophancy judge (95% / 100%). The hypothesis as worded, in terms of detection rate alone, would count these as successes. Any anchor benchmark needs matched controls.

**Surprise: harm detectors lean the wrong way on sycophancy.** AUROC is below 0.5 for Granite Guardian (0.24), the Llama harm judge (0.42) and the Qwen harm judge (0.15). Honest responses that contradict the user score as slightly *more* harmful than sycophantic agreement. With Qwen this crosses the threshold: it flags 31% of honest corrections and 8% of sycophantic answers. I did not investigate the cause; a plausible but untested explanation is that corrections contain negative language ("that is not correct").

**Surprise: weak reasoning signal in harm detectors.** AUROC is 0.70–0.74 on reasoning for detectors that were never asked about correctness. I have not tested whether this reflects the error itself or a stylistic difference between anchors and controls; the response-length match rules out only the crudest artifact.

**Observation from anchor construction.** With a plain "always agree with the user" persona, Qwen2.5-7B corrected the user in 3 of 4 pilot items, and with "insert one subtle mistake" it wrote correct solutions in 4 of 4. That is four items per condition and is anecdotal, but it is why explicit elicitation prompts were needed.

## 7. Limitations

- **Real vs synthetic confound.** Overt anchors are real model outputs with human labels; subtle anchors are prompted and filter-selected. Part of the domain gap could come from that difference rather than from the domain. A cleaner design would use synthetic anchors in all three domains or naturally occurring sycophancy.
- **Selected anchors.** 42% of sycophancy pairs and 83% of reasoning pairs were filtered out. The survivors are clean, unhedged cases; real sycophancy is often partial. Subtlety also varies: some reasoning anchors contain a visibly inconsistent last step.
- **Small, single-token judges.** Production judges are larger and often reason before answering. The reasoning-domain capability gap may shrink considerably with a stronger judge; I could not test this without API access.
- **Judge-dependence.** The targeted-rubric results come from one judge; the second judge did not reproduce them. Two models are too few to say which is typical.
- **Generator overlap.** The second judge is the model that generated the subtle items.
- **Detector coverage.** Llama Guard, ShieldGemma, the HarmBench classifier and all API moderation endpoints were not run. Granite Guardian was run only with its default "harm" risk; it ships other risk definitions that I did not try.
- **Narrow domain instances.** One sycophancy format (trivia with a stated wrong belief) and one reasoning task (grade-school arithmetic). No agentic or deception settings.
- **One generation seed**, default thresholds only (plus the 5%-FPR recalibration), English only.
- **Contamination.** JailbreakBench, TriviaQA and GSM8K are public and likely in training data. This affects how hard the items are for the judges, less so the cross-domain contrast.
- **Post-hoc elements** are labelled as such (§5.3) and carry less weight.

## 8. Conclusions & Next Steps

Harm-style safety metrics have sharply domain-dependent sensitivity: near-complete on overt violations, near-zero on sycophancy and flawed reasoning. Their null results in those domains are uninformative. The claim that the gap is purely an evaluation artifact is half right: a targeted rubric closes it for sycophancy (one judge), but for flawed reasoning the judge also needs the answer or more capability.

Practical recommendations:

1. Report a per-domain positive control (anchor TPR and control FPR) beside every "no issues found".
2. Never report anchor detection rate without the matched-control false-positive rate.
3. Validate each judge-and-rubric pair on labelled items before use; answer-label wording alone flipped one judge from below chance to 95% / 4%.

Follow-ups, in order of value:

1. Repeat with frontier API judges and Llama Guard to see how much of the reasoning gap is judge capability.
2. Replace the real/synthetic split with synthetic anchors in all domains, and add naturally occurring sycophancy.
3. Let judges reason before answering.
4. Test the "corrections look harmful" effect directly by varying the wording of honest corrections.
5. Extend to deception and agentic sabotage, and vary anchor subtlety on a graded scale.

## 9. References

- Chao et al. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.
- Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems (GSM8K).
- Hubinger et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.
- Inan et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.
- Lanham et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning.
- Mazeika et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.
- Padhi et al. (2024). Granite Guardian.
- Perez et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations.
- Sharma et al. (2023). Towards Understanding Sycophancy in Language Models.
- Souly et al. (2024). A StrongREJECT for Empty Jailbreaks.
- Turpin et al. (2023). Language Models Don't Always Say What They Think.
- Zou et al. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models.

Citations are from memory; titles and years should be checked before reuse.
