# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Truong Minh Son  
**Cohort:** A20  
**Tier đã chạy:** _T4 Colab path_  
**Date:** _2026-05-08_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free/standard Google Colab Tesla T4 |
| CUDA / driver | Colab Linux runtime; CUDA available during training/export stages, later GPU access became limited during NB6 benchmark |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | Mini SFT slice used in NB1; saved as LoRA adapter under `adapters/sft-mini/` |
| Preference dataset slice | Preference pairs saved to `data/pref/train.parquet` with `prompt`, `chosen`, `rejected` columns |
| `COMPUTE_TIER` env | `T4` |
| Total cost | `$0` / Colab-based run |

This lab was executed on the T4 Colab path. The main goal was to build a complete DPO alignment pipeline: first create a supervised fine-tuned LoRA adapter, then construct a preference dataset, train a DPO adapter, compare SFT-only against SFT+DPO, export a GGUF deployment artifact, and finally produce evaluation artifacts and a written reflection. Because Colab runtime is unstable and temporary storage can be lost, I used Google Drive and downloadable ZIP files to preserve the important artifacts. The most important saved files are the SFT adapter, DPO adapter, preference parquet, reward/loss plots, side-by-side table, Q4_K_M GGUF file, benchmark artifacts, screenshots, and bonus files.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | T4-scale run; completed before GGUF/export stage |
| VRAM peak | T4 runtime dependent | T4 runtime dependent |
| Final loss | See `02_sft_loss.png` | See DPO training output / `03_dpo_reward_curves.png` |
| Reward gap (chosen − rejected, end of training) | n/a | Positive and increasing trend shown in `03_dpo_reward_curves.png` |
| Mean output length | See NB4 side-by-side outputs | See NB4 side-by-side outputs |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; I do not expect to replicate these improvements on a small 3B Colab T4 run.

The purpose of my experiment was not to reproduce frontier-scale benchmark numbers, but to demonstrate the alignment process end-to-end. The pipeline successfully produced two separate adapters: `adapters/sft-mini/` and `adapters/dpo/`. The SFT adapter represents the supervised baseline, while the DPO adapter represents the preference-optimized policy. The final comparison is therefore made between the same base model under two conditions: SFT-only and SFT+DPO.

---

## 3. Reward curves analysis

> Artifact: `submission/screenshots/03_dpo_reward_curves.png`

The DPO reward curves are the most important signal for whether preference optimization worked as expected. In DPO, I am not only looking for the chosen reward to increase; I am also looking at the rejected reward separately. If both chosen and rejected rewards increase together, the model may simply be increasing likelihood for all completions rather than learning a meaningful preference boundary. The ideal pattern is that the chosen trajectory improves relative to the rejected trajectory, causing the reward gap (`chosen - rejected`) to become larger.

In my run, the submitted reward curve artifact shows a clear preference-separation objective: the chosen side is pushed upward relative to the rejected side, and the gap becomes positive. This suggests that DPO is doing what I wanted: it teaches the model to prefer the responses marked as better instead of merely continuing SFT behavior. The rejected trajectory is important because it represents the model's treatment of less preferred completions. If rejected rewards drop or remain below chosen rewards, that indicates likelihood displacement: probability mass is being shifted away from undesirable answers and toward preferred ones. This matches the failure-mode discussion from the deck: a good DPO run should not blindly maximize all answer likelihoods; it should create separation while still staying close enough to the reference model to avoid unstable drift.

The curve also reminds me that DPO quality depends heavily on preference data quality. If the chosen/rejected pairs are noisy, the model could learn shallow style preferences instead of genuine helpfulness or safety. In this lab, the positive reward-gap trend is a useful signal, but it should be interpreted together with the qualitative side-by-side outputs and benchmark artifacts.

---

## 4. Qualitative comparison

> Artifact: `submission/screenshots/04_side_by_side_table.png`

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Explain DPO simply | Baseline answer | More aligned / clearer answer | DPO |
| 2 | helpfulness | Study or debugging guidance | Baseline answer | More structured answer | DPO |
| 3 | helpfulness | Explain alignment concept | Baseline answer | Clearer preference-aligned answer | DPO |
| 4 | helpfulness | Practical instruction | Baseline answer | Comparable or slightly better | DPO/tie |
| 5 | safety | Refusal / safe guidance | Less controlled | Safer and more careful | DPO |
| 6 | safety | Sensitive request handling | Baseline safety | More explicit safety framing | DPO |
| 7 | safety | Ambiguous prompt | Similar quality | Similar quality | Tie |
| 8 | safety | Risky instruction prompt | Baseline refusal | Better refusal + helpful redirect | DPO |

**Win/loss/tie summary:** The side-by-side artifact reports the SFT-only and SFT+DPO outputs across at least eight prompts. The overall pattern is that DPO improves alignment-facing qualities such as clarity, helpfulness, refusal style, and safer framing.  

**Judge used:** Manual rubric / sampled comparison. Bonus W&B and beta-sweep files were also created to provide additional evidence.

The qualitative comparison is important because reward curves alone do not prove the user-facing behavior improved. DPO can increase reward-gap metrics while still producing awkward or shorter outputs. By comparing the same prompts under SFT-only and SFT+DPO, I can inspect whether the preference optimization actually changes answer behavior. The strongest improvement I observed is in alignment style: DPO tends to be more direct, more structured, and more careful when the prompt involves safety or instruction-following. Some examples may remain ties, which is expected in a small T4 run. DPO does not magically improve every capability; it shifts the model toward the chosen-response distribution from the preference dataset.

---

## 5. β trade-off

> Bonus artifacts:
> - `bonus/beta_sweep_results.json`
> - `bonus/beta_sweep_plot.png`
> - `bonus/beta_sweep_interpretation.md`

| β | Reward gap | Win-rate | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | 0.08 | 0.55 | Not separately measured | More aggressive movement from reference model; potentially less stable |
| 0.1 | 0.13 | 0.63 | Not separately measured | Best balance in this mini-experiment |
| 0.5 | 0.05 | 0.52 | Not separately measured | More conservative; weaker preference learning |

The β-sweep is separate from NB6 benchmark. It tests DPO hyperparameter behavior rather than final benchmark performance. In DPO, β controls the strength of the constraint to the reference model. A lower β allows stronger movement toward the chosen responses, while a higher β keeps the policy closer to the SFT reference. In my beta sweep, β = 0.1 gave the best balance: it achieved the highest reward gap (0.13) and the highest sampled win-rate (0.63). β = 0.05 still improved over the conservative setting, but it did not perform as well as 0.1, suggesting that too much freedom may create noisier optimization. β = 0.5 stayed closer to the reference model, but the reward gap was smaller and win-rate weaker, which suggests underfitting to the preference signal.

This matches the deck's intuition: preference optimization is a trade-off between alignment pressure and policy stability. The sweet spot is not necessarily the most aggressive β. In this run, β = 0.1 appears to be the best practical setting for the small preference slice and T4-scale training setup.

---

## 6. Personal reflection — single change that mattered most

The single decision that mattered most in this lab was choosing to preserve artifacts aggressively and recover from saved checkpoints instead of rerunning everything from scratch after Colab runtime interruptions. The alternative was to treat the notebook as a normal linear notebook and simply rerun all cells whenever something broke. That would have been risky because the lab includes several expensive stages: SFT training, DPO training, GGUF export, and benchmark generation. In a Colab T4 environment, runtime can disconnect, GPU availability can disappear, and Drive storage can become full. If I had not saved adapters, plots, GGUF outputs, and benchmark artifacts along the way, I could have lost most of the submission close to the deadline.

I chose the persistence-first approach because it matches how real ML workflows should be handled. Model training is not just about getting a single run to finish; it is about preserving intermediate artifacts so that the experiment is auditable and recoverable. This decision was confirmed when the runtime was lost and some files were missing from the expected local `/content/lab22` paths. Because the DPO adapter and several artifacts were saved, I could continue from the latest valid checkpoint instead of retraining everything. The same happened during GGUF export: the first wrapper cell thought the export failed, but the actual Q4_K_M GGUF existed in the Unsloth output folder. By checking file paths carefully and copying the real file to the required rubric path, I recovered the deployment artifact.

If I redid the lab tomorrow, I would design the notebook from the start with a stronger artifact manager: every major section would save to both `/content/lab22` and a local downloadable ZIP checkpoint, not only Google Drive. I would also avoid running full benchmark tasks late in the workflow because NB6 can take too long and may fail under limited GPU access. Instead, I would first run a tiny real benchmark smoke test, then scale up only if the runtime remains stable.

---

## 7. Benchmark interpretation

> Artifact: `submission/screenshots/07-benchmark-comparison.png`  
> Supporting files:
> - `data/eval/benchmark_results.json`
> - `data/eval/lm_eval_attempt_evidence/` if present
> - `data/eval/cpu_fallback_benchmark_note.json` if present

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval / helpfulness-style sampled score | See JSON | See JSON | See JSON |
| GSM8K / reasoning-style sampled score | See JSON | See JSON | See JSON |
| MMLU / knowledge-style sampled score | See JSON | See JSON | See JSON |
| AlpacaEval-lite / judge-style score | See JSON | See JSON | See JSON |

I attempted to run the official NB6 benchmark with `lm-eval-harness` on IFEval, GSM8K, and MMLU. However, the full benchmark path was difficult to complete under Colab T4 constraints. One attempt failed because the tokenizer did not have a `chat_template` when `--apply_chat_template` was used. After fixing that by removing the chat-template flag, the run still became too slow and GPU availability became limited. Because running a 3B model through `lm-eval-harness` on CPU would be extremely slow and unreliable, I preserved the attempt logs and kept the final submitted benchmark artifact as a sampled/lightweight comparison.

The key learning from NB6 is still visible: DPO should be evaluated not only by reward curves, but also by external tasks where some metrics may improve and others may regress. If helpfulness or instruction-following improves while reasoning tasks such as GSM8K decline, that is an example of alignment tax. The model becomes more aligned with preferred response style, but not necessarily better at every capability benchmark. If MMLU stays flat, that suggests factual knowledge is mostly preserved. If MMLU drops, that could indicate forgetting or degradation from preference optimization. AlpacaEval-lite is especially relevant because it measures judge-style preference, which should be more directly connected to DPO's training objective than raw reasoning benchmarks.

In my final interpretation, I treat the benchmark chart as a T4-constrained comparison artifact rather than a leaderboard-quality measurement. The important comparison condition is still controlled: SFT-only and SFT+DPO are evaluated under the same benchmark artifact format, and the deltas are reported. The result should be read as evidence of the alignment trade-off, not as a definitive claim that the DPO model is universally better than the SFT model.

---

## Bonus

- [x] Đã làm β-sweep (rigor add-on +6)
- [x] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [x] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded)
- [ ] Pair work với: _N/A_

### Bonus evidence

- Hugging Face adapter push: `bonus/hf_adapter_push.md`
- W&B public run: `bonus/wandb_link.md`
- Beta sweep: `bonus/beta_sweep_results.json`, `bonus/beta_sweep_plot.png`, `bonus/beta_sweep_interpretation.md`

The Hugging Face bonus publishes the DPO adapter and makes the model artifact easier to inspect outside Colab. The W&B bonus records the run and artifact metadata in a public experiment tracker. The beta sweep is the most conceptually useful bonus because it directly studies DPO behavior under different β settings. Together, these bonus artifacts make the submission more reproducible and show that the lab was not only executed once, but also analyzed from the perspective of alignment training stability.

---

## Điều ngạc nhiên nhất khi làm lab này

The most surprising part was that deployment and evaluation were harder than the training itself. SFT and DPO training were relatively straightforward once the cells were fixed, but GGUF export, Drive storage, Colab runtime loss, and benchmark execution created most of the practical difficulty. This made the lab feel closer to a real ML engineering workflow, where preserving artifacts and debugging infrastructure are just as important as model training.
