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

## 2. GPT-2 sentiment classification on IMDb (`gpt2-imdb-sentiment/`)

Adapted GPT-2 for binary sentiment classification of IMDb movie reviews (5,000 training and 1,000 test reviews).

- **Frozen GPT-2 + classification head:** the pre-trained GPT-2 was frozen and only a small classification head (1,538 parameters) was trained.
  Test accuracy: **74.6%** (F1: 0.746, perplexity: 1.89).
- **LoRA setup (PEFT):** LoRA adapters (rank 16, on the `c_attn` and `c_proj` attention layers) were added to GPT-2, reducing trainable parameters to about 1.3% of the model. The LoRA model was implemented and tested, but not fully trained or evaluated.

## Tech stack
Python, PyTorch, Hugging Face Transformers, PEFT, Datasets

## Running the notebooks
The Gemma notebooks require a Hugging Face access token. Set it as an environment variable before running:

```bash
export HF_TOKEN=your_token_here
```
