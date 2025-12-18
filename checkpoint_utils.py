"""
Checkpoint Utilities
====================

Utilities for loading and using saved checkpoints from FP8 training.

Usage:
    # Load a checkpoint
    from checkpoint_utils import load_checkpoint
    model, optimizer, info = load_checkpoint('checkpoint_bf16_1.5B.pt', device='cuda')
    
    # Inspect checkpoint
    python checkpoint_utils.py checkpoint_bf16_1.5B.pt
"""

import torch
import argparse
from pathlib import Path

# Import GPTConfig so torch.load can unpickle it
# This is needed because checkpoints contain the config dataclass
try:
    from train_fp8_full import GPTConfig, GPT
except ImportError:
    try:
        from train_gpt_fp8 import GPTConfig, GPT
    except ImportError:
        GPTConfig = None
        GPT = None
        print("Warning: Could not import GPTConfig. Some features may not work.")


def load_checkpoint(checkpoint_path, device='cuda'):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto
    
    Returns:
        model: Loaded model (needs to be created first based on config)
        optimizer: Optimizer state (optional)
        info: Dictionary with training info
    """
    # weights_only=False needed because we save GPTConfig dataclass
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    info = {
        'step': checkpoint['step'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'config': checkpoint['config'],
        'args': checkpoint.get('args', {}),
    }
    
    return checkpoint['model_state_dict'], checkpoint.get('optimizer_state_dict'), info


def inspect_checkpoint(checkpoint_path):
    """Print information about a checkpoint."""
    # weights_only=False needed because we save GPTConfig dataclass
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print("="*70)
    print(f"Training step: {checkpoint['step']}")
    print(f"Train loss: {checkpoint['train_loss']:.4f}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    print()
    
    config = checkpoint['config']
    print("Model Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Embed dim: {config.n_embd}")
    print(f"  Heads: {config.n_head} (KV: {config.n_kv_head})")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Block size: {config.block_size}")
    print()
    
    if 'args' in checkpoint:
        args = checkpoint['args']
        print("Training Configuration:")
        print(f"  Model size: {args.get('model_size', 'N/A')}")
        print(f"  Batch size: {args.get('batch_size', 'N/A')}")
        print(f"  Sequence length: {args.get('seq_len', 'N/A')}")
        print(f"  Learning rate: {args.get('lr', 'N/A')}")
        print(f"  FP8 enabled: {args.get('use_fp8', 'N/A')}")
        if args.get('use_fp8'):
            print(f"  FP8 recipe: {args.get('fp8_recipe', 'N/A')}")
        print()
    
    # Calculate model size
    state_dict = checkpoint['model_state_dict']
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"Total parameters: {total_params/1e9:.2f}B ({total_params/1e6:.1f}M)")
    
    # File size
    file_size = Path(checkpoint_path).stat().st_size
    print(f"Checkpoint size: {file_size/1e9:.2f} GB ({file_size/1e6:.1f} MB)")
    print("="*70)


def create_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Create and load a model from checkpoint.
    
    This reconstructs the model from the saved config and loads weights.
    """
    if GPT is None:
        raise ImportError("Could not import GPT model. Make sure train_fp8_full.py is in the same directory.")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from step {checkpoint['step']}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    
    return model


def compare_checkpoints(checkpoint_paths):
    """Compare multiple checkpoints."""
    print("="*70)
    print("Checkpoint Comparison")
    print("="*70)
    print(f"{'Checkpoint':<40} {'Step':>8} {'Train Loss':>12} {'Val Loss':>12}")
    print("-"*70)
    
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        name = Path(path).name
        step = checkpoint['step']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        
        print(f"{name:<40} {step:>8} {train_loss:>12.4f} {val_loss:>12.4f}")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checkpoint utilities')
    parser.add_argument('checkpoint', nargs='+', help='Checkpoint file(s)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple checkpoints')
    
    args = parser.parse_args()
    
    if args.compare and len(args.checkpoint) > 1:
        compare_checkpoints(args.checkpoint)
    else:
        for cp in args.checkpoint:
            inspect_checkpoint(cp)
            if len(args.checkpoint) > 1:
                print()