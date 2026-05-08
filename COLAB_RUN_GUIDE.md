# Colab Run Guide For Lab 22

Use this path if your laptop has weak GPU/RAM.

## 1. Start Colab

1. Open Google Colab.
2. Runtime -> Change runtime type -> T4 GPU.
3. Run:

```python
!nvidia-smi
```

Take screenshot `submission/screenshots/01-setup-gpu.png`.

## 2. Clone And Setup

Replace the GitHub URL with your own public repo URL after you push this repo.

```bash
!git clone https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git
%cd Day22-Track3-DPO-Alignment-Lab
!bash setup-colab.sh
```

For the safest path, run the patched notebook sources through the Makefile:

```bash
!make smoke
!make pipeline
```

This executes:

```text
notebooks/01_sft_mini.py
notebooks/02_preference_data.py
notebooks/03_dpo_train.py
notebooks/04_compare_and_eval.py
notebooks/05_merge_deploy_gguf.py
notebooks/06_benchmark.py
```

## 3. API Judge Is Optional

If you have an OpenAI or Anthropic key, set it before `make pipeline`:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["JUDGE_MODEL"] = "gpt-4o-mini"
```

If you do not have a key, the lab still runs. Notebook 04 creates manual/tie
judge results, and Notebook 06 uses those as a fallback for the fourth benchmark
bar.

## 4. Download Results Back To Your Repo

After the run, download these folders/files from Colab and copy them into your
local repo:

```text
adapters/sft-mini/
adapters/dpo/
data/pref/train.parquet
data/eval/
gguf/
submission/screenshots/
notebooks/*.ipynb
```

Convenient zip command in Colab:

```bash
!zip -r lab22-results.zip adapters data gguf submission notebooks/*.ipynb
```

Then download `lab22-results.zip` from the Colab file browser.

Then fill:

```text
submission/REFLECTION.md
```

## 5. Verify And Submit

On your laptop:

```bash
python scripts/verify.py
```

If it passes, push the repo publicly and submit the GitHub URL to LMS.
