# Pre-training Data (GPT-2 Tokenized, Mixed)

Mixed pre-training corpus with chunk-level shuffling for LLM training.

## Data Mix

| Source | Ratio | Tokens | Description |
|--------|-------|--------|-------------|
| FineWeb-EDU | 70% | 0.7B | Educational web content |
| The Stack v2 | 20% | 0.2B | Code (Python, JS, etc.) |
| arXiv | 10% | 0.1B | Scientific papers |

**Total: 1.0B tokens**

## Format

- Binary shards (~100M tokens each)
- GPT-2 tokenizer (vocab size 50,257)
- Chunk-shuffled (2048 tokens per chunk)
- llm.c / nanoGPT compatible format

## Usage

```python
from huggingface_hub import hf_hub_download

# Download specific shard
hf_hub_download(repo_id="s-maddrellmander/idlemachines-50B", filename="train_00000.bin", 
                repo_type="dataset", local_dir="./data")

# Or use the download script
python download_pretrain_data.py --repo s-maddrellmander/idlemachines-50B
```

## Training

```bash
python train_gpt_fp8.py --train-data 'data/train_*.bin' --val-data 'data/val_*.bin'
```

## Details

- Seed: 42
- Train shards: 9
- Val shards: 1
