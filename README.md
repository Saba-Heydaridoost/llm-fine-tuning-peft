# LLM Fine-Tuning with PEFT

Parameter-efficient fine-tuning (PEFT) experiments on large language models.

## 1. Persian instruction tuning of Gemma-2 2B (`persian-gemma/`)

Fine-tuned `google/gemma-2-2b-it` on the Persian conversational dataset [`miladmim/slim-orca-dedup-chat-50k-persian`](https://huggingface.co/datasets/miladmim/slim-orca-dedup-chat-50k-persian) (system / user / assistant format) and compared three approaches. The models were then tested on sample prompts covering math reasoning, multiple-choice QA and essay writing.

| Method | Trainable parameters | Validation loss | Result |
|---|---|---|---|
| Prompt Tuning (PEFT) | ~46K (~0.002%) | 2.20 | Follows instructions and produces coherent Persian at very low cost |
| LoRA (PEFT) | ~3.2M (~0.12%) | **1.75** | Lowest validation loss and the most detailed answers |
| Partial-layer fine-tuning | ~353M (~14%) | - | Heavy on memory; the model degraded and repeated its input |

- `gemma_persian_prompt_tuning_vs_lora.ipynb`: data preparation, base-model evaluation, prompt tuning, LoRA and the comparative analysis
- `gemma_partial_layer_finetuning.ipynb`: traditional fine-tuning of selected layers
- `results/`: generated outputs of the base, prompt-tuned and LoRA models

## 2. GPT-2 sentiment classification with LoRA (`gpt2-imdb-sentiment/`)

Fine-tuned GPT-2 with a classification head on the IMDb reviews dataset, with an additional LoRA implementation through PEFT.

- Test accuracy: **74.6%** (F1: 0.746)

## Tech stack
Python, PyTorch, Hugging Face Transformers, PEFT, Datasets

## Running the notebooks
The Gemma notebooks require a Hugging Face access token. Set it as an environment variable before running:

```bash
export HF_TOKEN=your_token_here
```
