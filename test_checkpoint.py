#!/usr/bin/env python3
"""
test_checkpoint.py
==================

Quick sanity test that a checkpoint loads correctly and can do inference.
Run this before the full eval to catch issues early.

Usage:
    python test_checkpoint.py checkpoints/bf16_125M.pt
    python test_checkpoint.py checkpoints/bf16_125M.pt --device cpu
"""

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# =============================================================================
# Model Definition (minimal copy for standalone testing)
# =============================================================================

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.scale = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.epsilon)
        return x * rms * self.scale


class SwiGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.w3 = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, seq_len):
        B, T, H, D = q.shape
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs_cos = freqs.cos().view(1, T, 1, D//2)
        freqs_sin = freqs.sin().view(1, T, 1, D//2)
        q_left, q_right = q.chunk(2, dim=-1)
        k_left, k_right = k.chunk(2, dim=-1)
        q_out = torch.cat([q_left * freqs_cos - q_right * freqs_sin,
                           q_right * freqs_cos + q_left * freqs_sin], dim=-1)
        k_out = torch.cat([k_left * freqs_cos - k_right * freqs_sin,
                           k_right * freqs_cos + k_left * freqs_sin], dim=-1)
        return q_out, k_out


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = torch.nn.Linear(config.n_embd, (config.n_embd // config.n_head) * config.n_kv_head, bias=False)
        self.v_proj = torch.nn.Linear(config.n_embd, (config.n_embd // config.n_head) * config.n_kv_head, bias=False)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_query_groups = config.n_head // config.n_kv_head
        self.rope = RoPE(config)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        q, k = self.rope(q, k, T)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.repeat_interleave(self.n_query_groups, dim=1)
        v = v.repeat_interleave(self.n_query_groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.swiglu = SwiGLU(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
        self.rmsnorm_1 = RMSNorm(config.n_embd)
        self.rmsnorm_2 = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(self.rmsnorm_1(x))
        x = x + self.swiglu(self.rmsnorm_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 2048
    n_kv_head: int = 4
    
    @classmethod
    def from_size(cls, size: str):
        presets = {
            "125M":  (12,  768,  12, 4),
            "250M":  (14,  1024, 16, 4),
            "350M":  (24,  1024, 16, 4),
            "760M":  (24,  1536, 16, 4),
            "1.5B":  (24,  2048, 16, 4),
            "3B":    (32,  2560, 32, 8),
            "7B":    (32,  4096, 32, 8),
        }
        if size in presets:
            n_layer, n_embd, n_head, n_kv_head = presets[size]
            return cls(n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
        raise ValueError(f"Unknown size: {size}")


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe=torch.nn.Embedding(config.block_size, config.n_embd),
            h=torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.rmsnorm = RMSNorm(config.n_embd)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.rmsnorm(x)
        return self.lm_head(x)


# =============================================================================
# Test Functions
# =============================================================================

def test_checkpoint(checkpoint_path: str, device: str = "auto"):
    """Load checkpoint and run basic sanity checks."""
    
    print(f"{'='*60}")
    print("CHECKPOINT TEST")
    print(f"{'='*60}")
    print(f"\nCheckpoint: {checkpoint_path}")
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"\n[1/5] Loading checkpoint...")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"  ✓ Checkpoint loaded")
    except Exception as e:
        print(f"  ✗ Failed to load checkpoint: {e}")
        return False
    
    # Inspect checkpoint contents
    print(f"\n[2/5] Checkpoint contents:")
    for key in ckpt.keys():
        if key == 'model_state_dict':
            num_tensors = len(ckpt[key])
            print(f"  - {key}: {num_tensors} tensors")
        elif key == 'optimizer_state_dict':
            print(f"  - {key}: present")
        elif key == 'config':
            print(f"  - {key}: {ckpt[key]}")
        elif key == 'args':
            print(f"  - {key}: {list(ckpt[key].keys()) if isinstance(ckpt[key], dict) else ckpt[key]}")
        else:
            print(f"  - {key}: {ckpt[key]}")
    
    # Extract/infer config
    print(f"\n[3/5] Building model...")
    if 'config' in ckpt:
        config = ckpt['config']
        if isinstance(config, dict):
            config = GPTConfig(**config)
        print(f"  Config from checkpoint: {config.n_layer}L, {config.n_embd}D")
    else:
        args = ckpt.get('args', {})
        model_size = args.get('model_size', '125M')
        print(f"  Inferring config from model_size: {model_size}")
        config = GPTConfig.from_size(model_size)
    
    # Build model
    try:
        model = GPT(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model built: {num_params/1e6:.1f}M parameters")
    except Exception as e:
        print(f"  ✗ Failed to build model: {e}")
        return False
    
    # Load weights
    print(f"\n[4/5] Loading weights...")
    try:
        state_dict = ckpt['model_state_dict']
        # Handle torch.compile prefix
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✓ Weights loaded (strict=True)")
    except Exception as e:
        print(f"  ✗ Failed to load weights: {e}")
        return False
    
    # Move to device and test inference
    print(f"\n[5/5] Testing inference...")
    try:
        model = model.to(torch.bfloat16).to(device)
        model.eval()
        
        # Generate a few tokens
        prompt = torch.tensor([[50256]], dtype=torch.long, device=device)  # EOS token as start
        
        with torch.no_grad():
            generated = [50256]
            for _ in range(10):
                input_ids = torch.tensor([generated], dtype=torch.long, device=device)
                output = model(input_ids)
                # Handle both (logits, loss) tuple and plain logits
                logits = output[0] if isinstance(output, tuple) else output
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)
        
        print(f"  ✓ Generated 10 tokens: {generated[1:]}")
        
        # Test with actual text using tiktoken or transformers
        try:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            
            prompt_text = "The quick brown fox"
            tokens = tokenizer.encode(prompt_text)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            with torch.no_grad():
                output = model(input_ids)
                logits = output[0] if isinstance(output, tuple) else output
                next_token = logits[0, -1].argmax().item()
                next_word = tokenizer.decode([next_token])
            
            print(f"  ✓ Text test: '{prompt_text}' -> '{next_word.strip()}'")
        except ImportError:
            print(f"  ⚠ transformers not installed, skipping text test")
        
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED ✓")
    print(f"{'='*60}")
    
    print(f"\nCheckpoint summary:")
    print(f"  Step: {ckpt.get('step', 'unknown')}")
    if 'val_loss' in ckpt and ckpt['val_loss']:
        val_loss = ckpt['val_loss']
        print(f"  Val loss: {val_loss:.4f} (ppl: {math.exp(val_loss):.2f})")
    if 'train_loss' in ckpt and ckpt['train_loss']:
        print(f"  Train loss: {ckpt['train_loss']:.4f}")
    
    print(f"\nReady for evaluation with:")
    print(f"  python eval_checkpoint.py -c {checkpoint_path} --suite quick")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test checkpoint loading")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    success = test_checkpoint(args.checkpoint, args.device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()