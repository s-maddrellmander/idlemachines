#!/usr/bin/env python3
"""
Pre-training Data Processor (Mixed Shards)
==========================================

Creates a 50B token pre-training mix with proper domain interleaving.
Each shard contains a shuffled mix of:
- 70% FineWeb-EDU (educational web content)
- 20% The Stack v2 (code)
- 10% RedPajama arXiv (scientific papers)

Shuffling is done at chunk-level (~2K tokens) to maintain document coherence
while ensuring good domain mixing within each shard.

Usage:
    # Process full 50B mix
    python process_pretrain_data.py --output-dir ./data --tokens 50B
    
    # Process smaller amount for testing
    python process_pretrain_data.py --output-dir ./data --tokens 1B
    
    # Upload to HuggingFace Hub after processing
    python process_pretrain_data.py --output-dir ./data --tokens 50B \\
        --upload --hf-repo your-username/pretrain-50B-gpt2

Output format:
    data/
    ├── train_00000.bin   # 100M tokens (70/20/10 mix, chunk-shuffled)
    ├── train_00001.bin
    ├── ...
    ├── val_00000.bin     # Validation shard (same mix)
    └── manifest.json     # Metadata
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random
from collections import deque
from typing import Iterator, List, Tuple

# Tokenizer
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("gpt2")
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    TOKENIZER = None

# HuggingFace datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    print("Please install datasets: pip install datasets")
    HF_AVAILABLE = False

# HuggingFace Hub upload
try:
    from huggingface_hub import HfApi, create_repo
    HF_UPLOAD_AVAILABLE = True
except ImportError:
    HF_UPLOAD_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

# Data mix ratios (must sum to 1.0)
DATA_MIX = {
    'fineweb_edu': 0.70,  # 70% educational web content
    'stack_v2': 0.20,     # 20% code
    'arxiv': 0.10,        # 10% scientific papers
}

# Shard configuration
DEFAULT_SHARD_SIZE = 100_000_000  # 100M tokens per shard
CHUNK_SIZE = 2048                  # ~2K tokens per chunk for shuffling
VAL_SHARDS = 1                     # Number of validation shards
VAL_TOKENS = 100_000_000           # 100M validation tokens

# Binary format (matching llm.c / nanoGPT)
HEADER_SIZE = 256 * 4  # 256 int32s
MAGIC_NUMBER = 20240520
VERSION = 1


# =============================================================================
# Tokenization & Shard Writing
# =============================================================================

def tokenize(text: str) -> List[int]:
    """Tokenize text using GPT-2 tokenizer."""
    return TOKENIZER.encode_ordinary(text)


def write_shard(tokens: List[int], filepath: Path) -> int:
    """
    Write tokens to binary shard file.
    
    Format:
    - Header: 256 int32s (magic, version, num_tokens, reserved...)
    - Data: uint16 tokens
    """
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = VERSION
    header[2] = len(tokens)
    
    tokens_u16 = np.array(tokens, dtype=np.uint16)
    
    with open(filepath, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens_u16.tobytes())
    
    return len(tokens)


# =============================================================================
# Data Source Iterators
# =============================================================================

def iter_fineweb_edu() -> Iterator[str]:
    """Iterate over FineWeb-EDU documents."""
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # Use sample for faster download, "default" for full
        split="train",
        streaming=True
    )
    for example in dataset:
        yield example["text"]


def iter_starcoderdata(languages: List[str] = None) -> Iterator[str]:
    """
    Iterate over StarCoderData (code) dataset.
    
    This is the cleaned, decontaminated version used to train StarCoder.
    Uses bigcode/starcoderdata which works reliably.
    """
    if languages is None:
        # Focus on popular languages
        languages = ["python",  "c", "cpp", "go", "rust", "shell"]
    
    for lang in languages:
        try:
            print(f"    Loading code: {lang}...")
            dataset = load_dataset(
                "bigcode/starcoderdata",
                data_dir=lang,
                split="train",
                streaming=True
            )
            for example in dataset:
                content = example.get("content", "")
                if content:
                    # Wrap code in markdown-style blocks
                    yield f"```{lang}\n{content}\n```"
        except Exception as e:
            print(f"    Warning: Could not load {lang}: {e}")
            continue


def iter_slimpajama_arxiv() -> Iterator[str]:
    """
    Iterate over arXiv documents from SlimPajama.
    
    SlimPajama contains multiple sources, we filter for arXiv only.
    """
    dataset = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True
    )
    for example in dataset:
        # Filter for arXiv only
        meta = example.get("meta", {})
        if meta.get("redpajama_set_name") == "RedPajamaArXiv":
            yield example["text"]


# =============================================================================
# Chunk-based Token Buffer
# =============================================================================

class ChunkedTokenBuffer:
    """
    Buffer that collects tokens into fixed-size chunks for shuffling.
    
    Tokens are added continuously, and when we have enough to form a chunk,
    it's added to the chunk list. Chunks can then be shuffled and written.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.current_chunk: List[int] = []
        self.complete_chunks: List[List[int]] = []
    
    def add_tokens(self, tokens: List[int], source: str = None):
        """Add tokens to buffer, forming complete chunks as we go."""
        self.current_chunk.extend(tokens)
        
        while len(self.current_chunk) >= self.chunk_size:
            chunk = self.current_chunk[:self.chunk_size]
            self.current_chunk = self.current_chunk[self.chunk_size:]
            self.complete_chunks.append(chunk)
    
    def get_shuffled_tokens(self, num_tokens: int, rng: random.Random) -> List[int]:
        """
        Get approximately num_tokens worth of shuffled chunks.
        
        Returns tokens and removes used chunks from buffer.
        """
        num_chunks_needed = (num_tokens + self.chunk_size - 1) // self.chunk_size
        
        if len(self.complete_chunks) < num_chunks_needed:
            return None  # Not enough chunks yet
        
        # Take the chunks we need
        chunks_to_use = self.complete_chunks[:num_chunks_needed]
        self.complete_chunks = self.complete_chunks[num_chunks_needed:]
        
        # Shuffle chunks
        rng.shuffle(chunks_to_use)
        
        # Flatten to token list
        tokens = []
        for chunk in chunks_to_use:
            tokens.extend(chunk)
        
        return tokens[:num_tokens]  # Trim to exact size
    
    @property
    def num_complete_chunks(self) -> int:
        return len(self.complete_chunks)
    
    @property
    def total_complete_tokens(self) -> int:
        return len(self.complete_chunks) * self.chunk_size


# =============================================================================
# Multi-Source Mixed Data Generator
# =============================================================================

class MixedDataGenerator:
    """
    Generates mixed, chunk-shuffled training data from multiple sources.
    
    Maintains the target ratio by tracking how many tokens we've seen from
    each source and adjusting which source to draw from next.
    """
    
    def __init__(self, mix_ratios: dict, chunk_size: int = CHUNK_SIZE, seed: int = 42):
        self.mix_ratios = mix_ratios
        self.chunk_size = chunk_size
        self.rng = random.Random(seed)
        
        # Token counts per source
        self.tokens_seen = {source: 0 for source in mix_ratios}
        
        # Iterators for each source
        self.iterators = {
            'fineweb_edu': iter_fineweb_edu(),
            'stack_v2': iter_starcoderdata(),  # Using starcoderdata
            'arxiv': iter_slimpajama_arxiv(),  # Using SlimPajama filtered
        }
        
        # Buffers for each source (collect tokens into chunks)
        self.buffers = {source: ChunkedTokenBuffer(chunk_size) for source in mix_ratios}
        
        # Track exhausted sources
        self.exhausted = {source: False for source in mix_ratios}
    
    def _get_next_source(self) -> str:
        """
        Determine which source to draw from next based on current ratios.
        
        Picks the source that is most "behind" its target ratio.
        """
        total_tokens = sum(self.tokens_seen.values()) or 1
        
        max_deficit = -float('inf')
        best_source = None
        
        for source, target_ratio in self.mix_ratios.items():
            if self.exhausted[source]:
                continue
            
            current_ratio = self.tokens_seen[source] / total_tokens
            deficit = target_ratio - current_ratio
            
            if deficit > max_deficit:
                max_deficit = deficit
                best_source = source
        
        return best_source
    
    def _fill_buffer(self, source: str, min_tokens: int = 1_000_000):
        """Fill a source's buffer with at least min_tokens."""
        iterator = self.iterators[source]
        buffer = self.buffers[source]
        tokens_added = 0
        
        while buffer.total_complete_tokens < min_tokens:
            try:
                text = next(iterator)
                tokens = tokenize(text)
                buffer.add_tokens(tokens, source)
                tokens_added += len(tokens)
            except StopIteration:
                self.exhausted[source] = True
                print(f"\n  Source exhausted: {source}")
                break
        
        return tokens_added
    
    def generate_shard(self, shard_size: int) -> Tuple[List[int], dict]:
        """
        Generate one shard worth of mixed, shuffled tokens.
        
        Returns:
            tokens: List of tokens for the shard
            stats: Dict with per-source token counts
        """
        shard_tokens = []
        shard_stats = {source: 0 for source in self.mix_ratios}
        
        # Calculate target tokens per source for this shard
        targets = {source: int(shard_size * ratio) 
                   for source, ratio in self.mix_ratios.items()}
        
        # Collect chunks from each source according to ratio
        all_chunks = []
        
        for source, target_tokens in targets.items():
            if self.exhausted[source]:
                continue
            
            # Ensure buffer has enough
            buffer = self.buffers[source]
            while buffer.total_complete_tokens < target_tokens and not self.exhausted[source]:
                self._fill_buffer(source, target_tokens)
            
            # Get chunks for this source
            num_chunks = target_tokens // self.chunk_size
            chunks_to_take = min(num_chunks, buffer.num_complete_chunks)
            
            for _ in range(chunks_to_take):
                if buffer.complete_chunks:
                    chunk = buffer.complete_chunks.pop(0)
                    all_chunks.append((source, chunk))
                    shard_stats[source] += len(chunk)
                    self.tokens_seen[source] += len(chunk)
        
        if not all_chunks:
            return None, None
        
        # Shuffle all chunks together
        self.rng.shuffle(all_chunks)
        
        # Flatten to token list
        for source, chunk in all_chunks:
            shard_tokens.extend(chunk)
        
        # Trim to exact shard size
        shard_tokens = shard_tokens[:shard_size]
        
        return shard_tokens, shard_stats
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        total = sum(self.tokens_seen.values()) or 1
        return {
            'tokens_per_source': self.tokens_seen.copy(),
            'ratios': {s: self.tokens_seen[s]/total for s in self.tokens_seen},
            'target_ratios': self.mix_ratios,
        }


# =============================================================================
# Main Processing
# =============================================================================

def process_mixed_data(
    output_dir: Path,
    total_tokens: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
    seed: int = 42,
):
    """
    Process and create mixed training shards.
    
    Args:
        output_dir: Output directory for shards
        total_tokens: Total tokens to process
        shard_size: Tokens per shard
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MIXED PRE-TRAINING DATA PROCESSOR")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Total tokens: {total_tokens/1e9:.1f}B")
    print(f"  Shard size: {shard_size/1e6:.0f}M tokens")
    print(f"  Chunk size: {CHUNK_SIZE} tokens")
    print(f"  Num shards: {total_tokens // shard_size}")
    print(f"\nData mix:")
    for source, ratio in DATA_MIX.items():
        print(f"  {source}: {ratio*100:.0f}% ({total_tokens * ratio / 1e9:.1f}B tokens)")
    print(f"\nOutput: {output_dir}")
    print("="*70)
    
    # Initialize generator
    generator = MixedDataGenerator(DATA_MIX, CHUNK_SIZE, seed)
    
    # Calculate number of shards
    num_train_shards = (total_tokens - VAL_TOKENS) // shard_size
    num_val_shards = VAL_SHARDS
    
    tokens_written = 0
    shard_idx = 0
    
    # Progress bar
    pbar = tqdm(total=total_tokens, unit="tok", unit_scale=True, desc="Processing")
    
    # Generate training shards
    print(f"\nGenerating {num_train_shards} training shards...")
    
    for shard_idx in range(num_train_shards):
        tokens, stats = generator.generate_shard(shard_size)
        
        if tokens is None:
            print(f"\nData exhausted at shard {shard_idx}")
            break
        
        shard_file = output_dir / f"train_{shard_idx:05d}.bin"
        write_shard(tokens, shard_file)
        
        tokens_written += len(tokens)
        pbar.update(len(tokens))
        
        # Periodic stats
        if (shard_idx + 1) % 10 == 0:
            stats = generator.get_stats()
            ratio_str = " | ".join([f"{s[:3]}:{r*100:.1f}%" for s, r in stats['ratios'].items()])
            tqdm.write(f"  Shard {shard_idx+1}: {ratio_str}")
    
    # Generate validation shard(s)
    print(f"\nGenerating {num_val_shards} validation shard(s)...")
    
    for val_idx in range(num_val_shards):
        tokens, stats = generator.generate_shard(VAL_TOKENS // num_val_shards)
        
        if tokens is None:
            print("Warning: Not enough data for validation")
            break
        
        shard_file = output_dir / f"val_{val_idx:05d}.bin"
        write_shard(tokens, shard_file)
        
        tokens_written += len(tokens)
        pbar.update(len(tokens))
    
    pbar.close()
    
    # Final stats
    final_stats = generator.get_stats()
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total tokens written: {tokens_written/1e9:.2f}B")
    print(f"Training shards: {shard_idx + 1}")
    print(f"Validation shards: {num_val_shards}")
    print(f"\nFinal mix ratios:")
    for source, ratio in final_stats['ratios'].items():
        target = DATA_MIX[source]
        diff = (ratio - target) * 100
        print(f"  {source}: {ratio*100:.1f}% (target: {target*100:.0f}%, diff: {diff:+.1f}%)")
    
    # Write manifest
    manifest = {
        'total_tokens': tokens_written,
        'num_train_shards': shard_idx + 1,
        'num_val_shards': num_val_shards,
        'shard_size': shard_size,
        'chunk_size': CHUNK_SIZE,
        'mix_ratios': DATA_MIX,
        'actual_ratios': final_stats['ratios'],
        'tokens_per_source': final_stats['tokens_per_source'],
        'seed': seed,
        'tokenizer': 'gpt2',
        'vocab_size': 50257,
    }
    
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest: {manifest_file}")
    print(f"{'='*70}")
    
    return manifest


def upload_to_hub(output_dir: Path, repo_id: str, private: bool = False):
    """Upload processed data to HuggingFace Hub."""
    if not HF_UPLOAD_AVAILABLE:
        print("Error: huggingface_hub not installed")
        return False
    
    print(f"\n{'='*70}")
    print(f"UPLOADING TO HUGGINGFACE HUB")
    print(f"Repo: {repo_id}")
    print(f"{'='*70}\n")
    
    api = HfApi()
    
    # Create repo
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return False
    
    output_dir = Path(output_dir)
    
    # Upload all bin files
    bin_files = sorted(output_dir.glob("*.bin"))
    print(f"Uploading {len(bin_files)} shard files...")
    
    for bin_file in tqdm(bin_files, desc="Uploading"):
        api.upload_file(
            path_or_fileobj=str(bin_file),
            path_in_repo=bin_file.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    # Upload manifest
    manifest_file = output_dir / "manifest.json"
    if manifest_file.exists():
        api.upload_file(
            path_or_fileobj=str(manifest_file),
            path_in_repo="manifest.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    # Create README
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    readme = f"""# Pre-training Data (GPT-2 Tokenized, Mixed)

Mixed pre-training corpus with chunk-level shuffling for LLM training.

## Data Mix

| Source | Ratio | Tokens | Description |
|--------|-------|--------|-------------|
| FineWeb-EDU | 70% | {manifest['tokens_per_source'].get('fineweb_edu', 0)/1e9:.1f}B | Educational web content |
| The Stack v2 | 20% | {manifest['tokens_per_source'].get('stack_v2', 0)/1e9:.1f}B | Code (Python, JS, etc.) |
| arXiv | 10% | {manifest['tokens_per_source'].get('arxiv', 0)/1e9:.1f}B | Scientific papers |

**Total: {manifest['total_tokens']/1e9:.1f}B tokens**

## Format

- Binary shards (~{manifest['shard_size']/1e6:.0f}M tokens each)
- GPT-2 tokenizer (vocab size 50,257)
- Chunk-shuffled ({manifest['chunk_size']} tokens per chunk)
- llm.c / nanoGPT compatible format

## Usage

```python
from huggingface_hub import hf_hub_download

# Download specific shard
hf_hub_download(repo_id="{repo_id}", filename="train_00000.bin", 
                repo_type="dataset", local_dir="./data")

# Or use the download script
python download_pretrain_data.py --repo {repo_id}
```

## Training

```bash
python train_gpt_fp8.py --train-data 'data/train_*.bin' --val-data 'data/val_*.bin'
```

## Details

- Seed: {manifest['seed']}
- Train shards: {manifest['num_train_shards']}
- Val shards: {manifest['num_val_shards']}
"""
    
    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme)
    
    api.upload_file(
        path_or_fileobj=str(readme_file),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print(f"\n{'='*70}")
    print(f"Upload complete!")
    print(f"https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*70}")
    
    return True


# =============================================================================
# CLI
# =============================================================================

def parse_tokens(s: str) -> int:
    """Parse token count string like '50B', '1B', '100M'."""
    s = s.upper().strip()
    if s.endswith('B'):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description='Process mixed pre-training data with chunk-level shuffling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 50B tokens (full mix)
  python process_pretrain_data.py --output-dir ./data --tokens 50B
  
  # Process 1B tokens for testing
  python process_pretrain_data.py --output-dir ./data --tokens 1B
  
  # Process and upload to HuggingFace
  python process_pretrain_data.py --output-dir ./data --tokens 50B \\
      --upload --hf-repo username/pretrain-50B-gpt2
"""
    )
    
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for shards')
    parser.add_argument('--tokens', type=str, default='50B',
                        help='Total tokens to process (e.g., 50B, 10B, 1B)')
    parser.add_argument('--shard-size', type=str, default='100M',
                        help='Tokens per shard (default: 100M)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    
    # Upload options
    parser.add_argument('--upload', action='store_true',
                        help='Upload to HuggingFace Hub after processing')
    parser.add_argument('--hf-repo', type=str, default='s-maddrellmander/idlemachines-50B',
                        help='HuggingFace repo ID (e.g., username/pretrain-50B)')
    parser.add_argument('--private', action='store_true',
                        help='Make HF repo private')
    parser.add_argument('--upload-only', action='store_true',
                        help='Skip processing, just upload existing data')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.upload_only:
        if not args.hf_repo:
            print("Error: --hf-repo required for upload")
            return
        upload_to_hub(output_dir, args.hf_repo, args.private)
        return
    
    if not HF_AVAILABLE:
        print("Error: datasets library required")
        print("Install: pip install datasets")
        return
    
    if TOKENIZER is None:
        print("Error: tiktoken required")
        print("Install: pip install tiktoken")
        return
    
    total_tokens = parse_tokens(args.tokens)
    shard_size = parse_tokens(args.shard_size)
    
    # Process data
    manifest = process_mixed_data(
        output_dir=output_dir,
        total_tokens=total_tokens,
        shard_size=shard_size,
        seed=args.seed,
    )
    
    # Upload if requested
    if args.upload:
        if not args.hf_repo:
            print("\nWarning: --hf-repo not specified, skipping upload")
        else:
            upload_to_hub(output_dir, args.hf_repo, args.private)


if __name__ == "__main__":
    main()