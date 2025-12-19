#!/usr/bin/env python3
"""
Download Pre-training Data from Hugging Face Hub
=================================================

Downloads the processed pre-training mix from HF Hub.
Similar to the fineweb10B download script but for our custom mix.

Usage:
    python download_pretrain_data.py --repo username/dataset        # Download all
    python download_pretrain_data.py --repo username/dataset --tokens 10B  # Download ~10B tokens
    python download_pretrain_data.py --repo username/dataset --val-only    # Just validation
    python download_pretrain_data.py --repo username/dataset --list        # List available shards
"""

import os
import sys
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


# Default config - override with --repo
DEFAULT_REPO_ID = "s-maddrellmander/idlemachines-50B"
DEFAULT_LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'pretrain')


def get_file(fname, repo_id, local_dir):
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


def list_shards(repo_id):
    """List all shard files in the HF repo."""
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        shards = [f for f in files if f.endswith('.bin')]
        return sorted(shards)
    except Exception as e:
        print(f"Error listing repo files: {e}")
        return []


def download_all(repo_id, local_dir, max_shards=None, train_only=False, val_only=False):
    """
    Download shards from the repo.
    
    Args:
        repo_id: HuggingFace repo ID
        local_dir: Local directory to save files
        max_shards: Maximum number of train shards to download (None = all)
        train_only: Only download train shards
        val_only: Only download validation shards
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Download manifest first if it exists
    try:
        get_file("manifest.json", repo_id, local_dir)
    except:
        print("No manifest found, continuing...")
    
    # List all shards
    all_shards = list_shards(repo_id)
    train_shards = sorted([s for s in all_shards if s.startswith('train_')])
    val_shards = sorted([s for s in all_shards if s.startswith('val_')])
    
    print(f"\nFound {len(train_shards)} train shards, {len(val_shards)} val shards")
    
    # Determine what to download
    shards_to_download = []
    
    if val_only:
        shards_to_download = val_shards
    elif train_only:
        shards_to_download = train_shards[:max_shards] if max_shards else train_shards
    else:
        # Both train and val
        train_subset = train_shards[:max_shards] if max_shards else train_shards
        shards_to_download = train_subset + val_shards
    
    print(f"Downloading {len(shards_to_download)} shards...")
    
    downloaded = []
    for shard in shards_to_download:
        path = get_file(shard, repo_id, local_dir)
        downloaded.append(path)
    
    return downloaded


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
    parser.add_argument('--repo', type=str, default=DEFAULT_REPO_ID,
                        help=f'HuggingFace repo ID (default: {DEFAULT_REPO_ID})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_LOCAL_DIR,
                        help=f'Local directory (default: {DEFAULT_LOCAL_DIR})')
    parser.add_argument('--tokens', type=str, default=None,
                        help='Max tokens to download (e.g., 10B, 50B). Assumes 100M per shard.')
    parser.add_argument('--shards', type=int, default=None,
                        help='Max number of train shards to download')
    parser.add_argument('--val-only', action='store_true',
                        help='Download validation shards only')
    parser.add_argument('--train-only', action='store_true',
                        help='Download training shards only (skip validation)')
    parser.add_argument('--list', action='store_true',
                        help='List available shards without downloading')
    
    args = parser.parse_args()
    
    if args.list:
        print(f"Listing shards in {args.repo}...")
        shards = list_shards(args.repo)
        train_shards = [s for s in shards if s.startswith('train_')]
        val_shards = [s for s in shards if s.startswith('val_')]
        
        print(f"\nTrain shards ({len(train_shards)}):")
        for s in train_shards:
            print(f"  {s}")
        
        print(f"\nVal shards ({len(val_shards)}):")
        for s in val_shards:
            print(f"  {s}")
        
        print(f"\nTotal: {len(shards)} shards")
        return
    
    # Calculate max_shards from tokens if specified
    max_shards = args.shards
    if args.tokens:
        tokens = parse_tokens(args.tokens)
        # Assume 100M tokens per shard
        max_shards = tokens // 100_000_000
        print(f"Downloading up to {max_shards} shards for {args.tokens} tokens")
    
    print("="*60)
    print(f"Downloading Pre-training Data")
    print(f"Repo: {args.repo}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    downloaded = download_all(
        repo_id=args.repo,
        local_dir=args.output_dir,
        max_shards=max_shards,
        train_only=args.train_only,
        val_only=args.val_only,
    )
    
    print(f"\n{'='*60}")
    print(f"Download complete: {len(downloaded)} shards")
    print(f"Location: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
