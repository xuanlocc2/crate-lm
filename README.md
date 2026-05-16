<!-- <p align="center">
    <img src="./assets/digirl-logo-text.png" alt="logo" width="20%">
</p>
-->


<h3 align="center">
Improving Neuron-level Interpretability with White-box Language Models 
<br>
<b>To Appear at CPAL 2025, Oral</b>

</h3>


<p align="center">
| <a href="https://crate-lm.github.io/"><b>Website</b></a> | <a href="https://arxiv.org/abs/2410.16443"><b>Paper</b></a> |
</p>

---

Research Code for preprint "Improving Neuron-level Interpretability with White-box Language Models".

[Hao Bai*](https://jackgethome.com), [Yi Ma](https://people.eecs.berkeley.edu/~yima/)<br>
UC Berkeley, UIUC, HKU
<br>
*Work done at UC Berkeley

# The CRATE Language Model

## Pre-training

Download the [Uncopyrighted Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted) dataset to `data`. Then run `run_pretrain.sh` to pre-train the CRATE language model. You can also get the 12L model we pre-trained [here](https://huggingface.co/JackBAI/CRATE-GPT-12L-Pile-600000steps).

## Performance Evaluation

Download the datasets you want to evaluate the CRATE language model. For example, you can get lambada, wikitext-2, openwebtext, and wikitext-103 from Huggingface datasets. After obtaining these datasets, just run `run_eval.sh` to evaluate the performance of the CRATE language model on downstream tasks.

---

# Benchmark Suite

The benchmark covers two tracks:

| Track | Description |
|-------|-------------|
| **Train from scratch** | Pre-train CRATE on the tested dataset (Pile) and two novel datasets (OpenWebText, Wikitext-103) |
| **Fine-tune evaluation** | Fine-tune a pre-trained CRATE checkpoint on two novel datasets (Shakespeare, Penn Treebank) and measure perplexity |

## Quick Start

```bash
# 1. Install dependencies
pip install torch tiktoken datasets transformers tqdm requests

# 2. Run the full benchmark (data prep → training → fine-tuning)
bash run_benchmark.sh

# 3. Run only data preparation
bash run_benchmark.sh --prepare-only

# 4. Run only pre-training
bash run_benchmark.sh --train-only

# 5. Run only fine-tuning (requires a pretrained checkpoint)
PRETRAIN_CKPT_DIR=out bash run_benchmark.sh --finetune-only
```

## Datasets

### Data Preparation

Each dataset has a `prepare.py` script in its `data/<dataset>/` directory.  
Run them individually or let `run_benchmark.sh` handle it.

| Dataset | Purpose | Script |
|---------|---------|--------|
| [Pile (Uncopyrighted)](https://huggingface.co/datasets/monology/pile-uncopyrighted) | Tested (paper) pre-training | `data/pile/prepare.py` |
| [OpenWebText](https://huggingface.co/datasets/openwebtext) | Novel pre-training dataset 1 | `data/openwebtext/prepare.py` |
| [Wikitext-103](https://huggingface.co/datasets/wikitext) | Novel pre-training dataset 2 | `data/wikitext/prepare.py` |
| [LAMBADA](https://huggingface.co/datasets/EleutherAI/lambada_openai) | LM evaluation | `data/lambada/prepare.py` |
| [Penn Treebank (PTB)](https://huggingface.co/datasets/ptb-text-only/ptb_text_only) | Novel fine-tune dataset 1 | `data/ptb/prepare.py` |
| [Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | Novel fine-tune dataset 2 | `data/shakespeare/prepare.py` |

## Training Configs

| Config | Description |
|--------|-------------|
| `config/train_gpt2.py` | Train CRATE on Pile (tested dataset) |
| `config/train_openwebtext.py` | Train CRATE on OpenWebText (novel dataset 1) |
| `config/train_wikitext103.py` | Train CRATE on Wikitext-103 (novel dataset 2) |

### Single-GPU example

```bash
python train.py config/train_openwebtext.py --compile=False
```

### Multi-GPU (DDP) example

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_openwebtext.py
```

## Fine-tuning Configs

Fine-tuning starts from a pre-trained CRATE checkpoint (set `out_dir` to your
checkpoint directory):

| Config | Dataset | Description |
|--------|---------|-------------|
| `config/finetune_shakespeare_crate.py` | Shakespeare | Novel fine-tune dataset 1 |
| `config/finetune_ptb_crate.py` | Penn Treebank | Novel fine-tune dataset 2 |

```bash
# Fine-tune on Shakespeare
python train.py config/finetune_shakespeare_crate.py --out_dir=<pretrain_ckpt_dir> --compile=False

# Fine-tune on PTB
python train.py config/finetune_ptb_crate.py --out_dir=<pretrain_ckpt_dir> --compile=False
```

## Evaluation Configs

After training or fine-tuning, evaluate perplexity with the corresponding eval config:

| Config | Evaluates on |
|--------|-------------|
| `config/eval_pile.py` | Pile |
| `config/eval_openwebtext.py` | OpenWebText |
| `config/eval_wikitext.py` | Wikitext-2 |
| `config/eval_wikitext103.py` | Wikitext-103 |
| `config/eval_lambada.py` | LAMBADA |
| `config/eval_ptb_crate.py` | Penn Treebank |
| `config/eval_shakespeare_crate.py` | Shakespeare |

```bash
# Example: evaluate CRATE trained on OpenWebText
python train.py config/eval_openwebtext.py --out_dir=out-crate-openwebtext --compile=False
```

---

## Small-Model Benchmark (laptop / single GPU)

A lightweight variant of the benchmark uses **CRATE-small** (4 layers, 4 heads,
256-dim embeddings, ~3 M parameters) and small datasets so the entire
pipeline can run on a laptop CPU or a single consumer GPU.

| Track | Description |
|-------|-------------|
| **Train from scratch** | Shakespeare (tested dataset) + Wikitext-2 and TinyStories (novel datasets) |
| **Fine-tune evaluation** | Fine-tune the Shakespeare checkpoint on Wikitext-2 and TinyStories; compare perplexity against from-scratch |

### Quick Start

```bash
# 1. Install dependencies (same as above)
pip install torch tiktoken datasets tqdm requests

# 2. Run the full small-model benchmark
bash run_benchmark_small.sh

# 3. Only data preparation
bash run_benchmark_small.sh --prepare-only

# 4. Only training from scratch
bash run_benchmark_small.sh --train-only

# 5. Only fine-tuning (requires Shakespeare checkpoint)
bash run_benchmark_small.sh --finetune-only

# 6. Force CPU execution
DEVICE=cpu bash run_benchmark_small.sh
```

### Datasets

| Dataset | Purpose | Tokens (approx.) | Script |
|---------|---------|------------------|--------|
| Shakespeare | Tested dataset (train from scratch) | ~300 K | `data/shakespeare/prepare.py` |
| Wikitext-2 | Novel dataset 1 | ~2 M | `data/wikitext2/prepare.py` |
| TinyStories | Novel dataset 2 | ~40 M (200 K stories) | `data/tinystories/prepare.py` |

### Configs

| Config | Purpose |
|--------|---------|
| `config/train_shakespeare_small.py` | Train CRATE-small from scratch on Shakespeare |
| `config/train_wikitext2_small.py` | Train CRATE-small from scratch on Wikitext-2 |
| `config/train_tinystories_small.py` | Train CRATE-small from scratch on TinyStories |
| `config/finetune_wikitext2_small.py` | Fine-tune Shakespeare ckpt → Wikitext-2 |
| `config/finetune_tinystories_small.py` | Fine-tune Shakespeare ckpt → TinyStories |
| `config/eval_shakespeare_small.py` | Evaluate perplexity on Shakespeare |
| `config/eval_wikitext2_small.py` | Evaluate perplexity on Wikitext-2 |
| `config/eval_tinystories_small.py` | Evaluate perplexity on TinyStories |

### Step-by-step (manual)

```bash
# ── Step 1: prepare data ────────────────────────────────────────────────────
python data/shakespeare/prepare.py
python data/wikitext2/prepare.py
python data/tinystories/prepare.py

# ── Step 2: train from scratch on the tested dataset (Shakespeare) ──────────
python train.py config/train_shakespeare_small.py
python train.py config/eval_shakespeare_small.py --out_dir=out-crate-small-shakespeare

# ── Step 3: train from scratch on the two novel datasets ───────────────────
python train.py config/train_wikitext2_small.py
python train.py config/eval_wikitext2_small.py --out_dir=out-crate-small-wikitext2

python train.py config/train_tinystories_small.py
python train.py config/eval_tinystories_small.py --out_dir=out-crate-small-tinystories

# ── Step 4: fine-tune the Shakespeare checkpoint on both novel datasets ─────
python train.py config/finetune_wikitext2_small.py --out_dir=out-crate-small-shakespeare
python train.py config/eval_wikitext2_small.py     --out_dir=out-crate-small-ft-wikitext2

python train.py config/finetune_tinystories_small.py --out_dir=out-crate-small-shakespeare
python train.py config/eval_tinystories_small.py     --out_dir=out-crate-small-ft-tinystories
```

> **Perplexity** is computed as `exp(val_loss)` from the `[EVAL]` output lines.

---

## Interpretability Evaluation

You should first install the automated interpretability evaluation tools from OpenAI. I accomodated the code from only supporting OpenAI checkpoint to any model on HuggingFace you want to use for evaluation. You can find the code in `automated-interpretability` folder.

Then you should get the activations from the CRATE language model. You can use the `./interpret/activations_crate_overcomplete.py` script to get the activations. The activations will be saved in the `neurons` folder. Then use `neurons/eval.py` to aggregate the interpretability of the CRATE language model.

You can choose to evaluate the neuron-level interpretability with either OpenAI or Anthropic metric. You need to also change `./automated-interpretability/neuron_explainer/activations/activations.py` line 160 and 197 when you change from OpenAI to Anthropic metric.

### Lightweight Local Interpretability Metrics

For a small local benchmark without running the full LLM explainer/simulator
pipeline, use `interpretability_metrics.py`. The script hooks CRATE ISTA sparse
code activations, builds activation records, and computes local proxy versions
of the three metric families used in the paper:

| Metric | Evaluation records |
|--------|--------------------|
| `random` | Random validation text windows |
| `top_and_random` | Top-activation windows plus random windows |
| `anthropic` | Samples from activation quantiles |

The script also assigns each ISTA feature to a subspace `U_k` by the largest
L2 energy of its dictionary atom over the attention-head-sized embedding chunks,
then aggregates metric scores by `U_k`.

> These are lightweight proxy scores for local experiments. They are not
> directly comparable to the paper's official LLM-based interpretability scores.

Example smoke test:

```powershell
python interpretability_metrics.py `
  --out_dir=out-crate-small-wikitext2 `
  --dataset=wikitext2 `
  --split=val `
  --device=cuda `
  --layers=0 `
  --num_batches=1 `
  --batch_size=1 `
  --num_features=4 `
  --top_k=3 `
  --random_k=3 `
  --quantile_k=1 `
  --output=interpretability_results\smoke_test_subspace.json
```

Run the small benchmark checkpoints:

```powershell
python interpretability_metrics.py `
  --out_dir=out-crate-small-wikitext2 `
  --dataset=wikitext2 `
  --split=val `
  --device=cuda `
  --layers=all `
  --num_batches=8 `
  --batch_size=4 `
  --num_features=32 `
  --output=interpretability_results\wikitext2_scratch_interpretability_subspace.json

python interpretability_metrics.py `
  --out_dir=out-crate-small-tinystories `
  --dataset=tinystories `
  --split=val `
  --device=cuda `
  --layers=all `
  --num_batches=8 `
  --batch_size=4 `
  --num_features=32 `
  --output=interpretability_results\tinystories_scratch_interpretability_subspace.json

python interpretability_metrics.py `
  --out_dir=out-crate-small-ft-wikitext2 `
  --dataset=wikitext2 `
  --split=val `
  --device=cuda `
  --layers=all `
  --num_batches=8 `
  --batch_size=4 `
  --num_features=32 `
  --output=interpretability_results\wikitext2_finetuned_interpretability_subspace.json

python interpretability_metrics.py `
  --out_dir=out-crate-small-ft-tinystories `
  --dataset=tinystories `
  --split=val `
  --device=cuda `
  --layers=all `
  --num_batches=8 `
  --batch_size=4 `
  --num_features=32 `
  --output=interpretability_results\tinystories_finetuned_interpretability_subspace.json
```

# Sparse Auto-encoder

## Training

We recommend using SAE Lens (https://github.com/jbloomAus/SAELens) or SAE repo from [Arthur Conmy](https://github.com/ArthurConmy/sae/tree/main). This study was done before SAE Lens was proposed, so we used the SAE repo from Arthur Conmy (located at `./arthursae`).

Before training SAE, you need to install [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) from Neel Nanda. I taylored the repo to support the CRATE language model. You need to first transform the nanogpt model to transformer_lens model (`./TransformerLens/demos/nanogpt_to_transformer_lens.ipynb`). You can find the taylored code in `transformer-lens` folder. You can run the SAE training scripts after installing TransformerLens.

## Evaluation

After saving the trained model, you should output the feature activations in a folder and use the same evaluation method mentioned above to evaluate the neuron-level interpretability. Also use the `./interpret/activations_crate_overcomplete.py` script to get the feature activations.

## Inspection

You can use `arthursae/inspect_sae.ipynb` to inspect the trained SAE model.
