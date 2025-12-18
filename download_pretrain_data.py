#!/usr/bin/env python3
"""
Download Pre-training Data from Hugging Face Hub
=================================================

Downloads the processed 50B token pre-training mix from HF Hub.
Similar to the fineweb10B download script but for our custom mix.

Usage:
    python download_pretrain_data.py                    # Download all (50B tokens)
    python download_pretrain_data.py --tokens 10B      # Download ~10B tokens
    python download_pretrain_data.py --edu-only        # Only FineWeb-EDU
    python download_pretrain_data.py --source stack    # Only code
    
Data mix (default):
    - 35B tokens: FineWeb-EDU (educational web content)
    - 10B tokens: The Stack v2 (code) 
    - 5B tokens:  RedPajama arXiv (scientific papers)
"""

import os
import sys
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


# Configure your HF repo here
HF_REPO_ID = "your-username/pretrain-50B-gpt2"  # Change this!
LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'pretrain_data')


def get_file(fname, repo_id=HF_REPO_ID, local_dir=LOCAL_DIR):
    """Download a single file from HF Hub if not already present."""
    local_path = os.path.join(local_dir, fname)
    if not os.path.exists(local_path):
        print(f"Downloading {fname}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir
        )
    else:
        print(f"Already exists: {fname}")
    return local_path


def list_shards(repo_id=HF_REPO_ID, prefix=None):
    """List all shard files in the HF repo."""
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        shards = [f for f in files if f.endswith('.bin')]
        if prefix:
            shards = [f for f in shards if f.startswith(prefix)]
        return sorted(shards)
    except Exception as e:
        print(f"Error listing repo files: {e}")
        return []


def download_source(source_name, max_shards=None, repo_id=HF_REPO_ID, local_dir=LOCAL_DIR):
    """
    Download shards for a specific source.
    
    Args:
        source_name: 'fineweb_edu', 'stack_v2', or 'arxiv'
        max_shards: Maximum number of shards to download (None = all)
    """
    prefix_map = {
        'fineweb_edu': 'fineweb_edu/',
        'edu': 'fineweb_edu/',
        'stack_v2': 'stack_v2/',
        'stack': 'stack_v2/',
        'code': 'stack_v2/',
        'arxiv': 'arxiv/',
    }
    
    prefix = prefix_map.get(source_name.lower())
    if not prefix:
        print(f"Unknown source: {source_name}")
        print(f"Available: {list(prefix_map.keys())}")
        return []
    
    shards = list_shards(repo_id, prefix)
    
    if max_shards:
        shards = shards[:max_shards]
    
    print(f"\nDownloading {len(shards)} shards for {source_name}...")
    
    downloaded = []
    for shard in shards:
        path = get_file(shard, repo_id, local_dir)
        downloaded.append(path)
    
    return downloaded


def download_all(max_tokens=None, repo_id=HF_REPO_ID, local_dir=LOCAL_DIR):
    """
    Download all data up to max_tokens.
    
    Token distribution per source:
    - fineweb_edu: 35B (350 shards @ 100M each)
    - stack_v2: 10B (100 shards)
    - arxiv: 5B (50 shards)
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Download manifest first
    try:
        get_file("manifest.json", repo_id, local_dir)
    except:
        print("No manifest found, downloading all shards...")
    
    # Calculate shard limits based on max_tokens
    # Each shard is ~100M tokens
    tokens_per_shard = 100_000_000
    
    if max_tokens:
        # Proportional distribution: 70% edu, 20% code, 10% arxiv
        edu_tokens = int(max_tokens * 0.70)
        code_tokens = int(max_tokens * 0.20)
        arxiv_tokens = int(max_tokens * 0.10)
        
        edu_shards = max(1, edu_tokens // tokens_per_shard)
        code_shards = max(1, code_tokens // tokens_per_shard)
        arxiv_shards = max(1, arxiv_tokens // tokens_per_shard)
    else:
        edu_shards = None  # All
        code_shards = None
        arxiv_shards = None
    
    downloaded = []
    
    print("="*60)
    print("Downloading Pre-training Data")
    print("="*60)
    
    # Download each source
    downloaded.extend(download_source('fineweb_edu', edu_shards, repo_id, local_dir))
    downloaded.extend(download_source('stack_v2', code_shards, repo_id, local_dir))
    downloaded.extend(download_source('arxiv', arxiv_shards, repo_id, local_dir))
    
    print(f"\n{'='*60}")
    print(f"Download complete: {len(downloaded)} shards")
    print(f"Location: {local_dir}")
    print(f"{'='*60}")
    
    return downloaded


def download_validation(repo_id=HF_REPO_ID, local_dir=LOCAL_DIR):
    """Download just the validation shards (for quick testing)."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    shards = list_shards(repo_id)
    val_shards = [s for s in shards if 'val' in s.lower()]
    
    if not val_shards:
        # If no dedicated val shards, grab first shard of each source
        val_shards = []
        for source in ['fineweb_edu', 'stack_v2', 'arxiv']:
            source_shards = [s for s in shards if source in s]
            if source_shards:
                val_shards.append(source_shards[0])
    
    print(f"Downloading {len(val_shards)} validation shards...")
    for shard in val_shards:
        get_file(shard, repo_id, local_dir)
    
    return val_shards


def parse_tokens(s):
    """Parse token count string like '35B', '10B', '5B'."""
    if s is None:
        return None
    s = s.upper().strip()
    if s.endswith('B'):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)


def main():
    parser = argparse.ArgumentParser(description='Download pre-training data from HF Hub')
    parser.add_argument('--repo', type=str, default=HF_REPO_ID,
                        help=f'HuggingFace repo ID (default: {HF_REPO_ID})')
    parser.add_argument('--output-dir', type=str, default=LOCAL_DIR,
                        help=f'Local directory (default: {LOCAL_DIR})')
    parser.add_argument('--tokens', type=str, default=None,
                        help='Max tokens to download (e.g., 10B, 50B). Default: all')
    parser.add_argument('--source', type=str, default=None,
                        choices=['edu', 'fineweb_edu', 'code', 'stack', 'stack_v2', 'arxiv', 'all'],
                        help='Download specific source only')
    parser.add_argument('--val-only', action='store_true',
                        help='Download validation shards only (quick test)')
    parser.add_argument('--list', action='store_true',
                        help='List available shards without downloading')
    
    args = parser.parse_args()
    
    if args.list:
        print(f"Listing shards in {args.repo}...")
        shards = list_shards(args.repo)
        for s in shards:
            print(f"  {s}")
        print(f"\nTotal: {len(shards)} shards")
        return
    
    if args.val_only:
        download_validation(args.repo, args.output_dir)
        return
    
    if args.source and args.source != 'all':
        download_source(args.source, repo_id=args.repo, local_dir=args.output_dir)
    else:
        max_tokens = parse_tokens(args.tokens)
        download_all(max_tokens, args.repo, args.output_dir)


if __name__ == "__main__":
    main()