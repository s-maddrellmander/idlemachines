"""
FP8 Training Script - Full Featured Version
============================================

Complete training script with:
- Validation loss tracking
- Gradient norm logging
- Learning rate schedule (warmup + cosine decay)
- Comprehensive CSV logging
- Multiple FP8 recipes support

Usage:
    python train_fp8_full.py --model-size 1.5B --steps 15000 --fp8-recipe tensorwise
    python train_fp8_full.py --model-size 1.5B --steps 15000 --no-fp8  # BF16 baseline
"""

import os
import sys
import math
import glob
import time
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# FP8 imports
try:
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("WARNING: torchao not available")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.2f}h"


# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.epsilon)
        return x * rms * self.scale


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, (config.n_embd // config.n_head) * config.n_kv_head, bias=False)
        self.v_proj = nn.Linear(config.n_embd, (config.n_embd // config.n_head) * config.n_kv_head, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
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


class Block(nn.Module):
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
        """Create config from size preset."""
        presets = {
            # name: (n_layer, n_embd, n_head, n_kv_head)
            "125M":  (12,  768,  12, 4),
            "350M":  (24,  1024, 16, 4),
            "760M":  (24,  1536, 16, 4),
            "1.5B":  (24,  2048, 16, 4),
            "3B":    (32,  2560, 32, 8),
            "7B":    (32,  4096, 32, 8),
            "13B":   (40,  5120, 40, 8),
        }
        if size in presets:
            n_layer, n_embd, n_head, n_kv_head = presets[size]
            return cls(n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
        raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.rmsnorm = RMSNorm(config.n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Proper initialization scaled by depth.
        
        Uses GPT-style initialization where deeper layers get smaller init values
        to prevent gradient explosion. Based on:
        - GPT-2/3 papers
        - "On Layer Normalization in the Transformer Architecture" (Xiong et al.)
        """
        if isinstance(module, nn.Linear):
            # Standard deviation scaled by sqrt(2 * num_layers) for residual paths
            # This prevents activation/gradient explosion as depth increases
            std = 0.02 / math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.rmsnorm(x)
        
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

class SimpleDataLoader:
    """Random data for testing."""
    def __init__(self, B, T, vocab_size, device='cuda'):
        self.B = B
        self.T = T
        self.vocab_size = vocab_size
        self.device = device
        self.total_tokens = 10_000_000_000  # Fake large number
    
    def next_batch(self):
        x = torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device)
        y = torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device)
        return x, y
    
    def reset(self):
        pass


class FileDataLoader:
    """Load from binary files."""
    def __init__(self, filename_pattern, B, T):
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files matching {filename_pattern}")
        
        # Quick token count
        total_tokens = 0
        for f in self.files:
            with open(f, "rb") as fp:
                header = np.frombuffer(fp.read(256 * 4), dtype=np.int32)
                if header[0] == 20240520:
                    total_tokens += int(header[2])
        
        print(f"Data: {len(self.files)} shards, ~{total_tokens/1e9:.2f}B tokens total")
        self.total_tokens = total_tokens
        
        self.current_shard = 0
        self.current_position = 0
        self._load_shard()
    
    def _load_shard(self):
        with open(self.files[self.current_shard], "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            self.tokens = np.frombuffer(f.read(), dtype=np.uint16)
    
    def reset(self):
        """Reset to beginning of data."""
        self.current_shard = 0
        self.current_position = 0
        self._load_shard()
    
    def next_batch(self):
        buf = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(self.B, self.T).cuda()
        y = buf[1:].view(self.B, self.T).cuda()
        
        self.current_position += self.B * self.T
        if self.current_position + self.B * self.T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.files)
            self.current_position = 0
            self._load_shard()
        
        return x, y


# -----------------------------------------------------------------------------
# Learning Rate Schedule
# -----------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    Args:
        step: Current training step (1-indexed)
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        max_lr: Peak learning rate
        min_lr: Minimum learning rate (at end of training)
    
    Returns:
        Learning rate for this step
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    # Cosine decay
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# -----------------------------------------------------------------------------
# Gradient Norm Computation
# -----------------------------------------------------------------------------

def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, num_batches=50):
    """Run validation and return average loss."""
    model.eval()
    val_loader.reset()
    
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = val_loader.next_batch()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


# -----------------------------------------------------------------------------
# FP8 Conversion
# -----------------------------------------------------------------------------

def apply_fp8(model, recipe="tensorwise"):
    """Convert model to FP8 training."""
    if not TORCHAO_AVAILABLE:
        print("torchao not available, skipping FP8")
        return model
    
    def module_filter_fn(mod, fqn):
        if isinstance(mod, nn.Linear):
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if mod.in_features < 512 or mod.out_features < 512:
                return False
        return True
    
    config = Float8LinearConfig.from_recipe_name(recipe)
    
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, module_filter_fn=module_filter_fn, config=config)
    
    try:
        from torchao.float8.float8_linear import Float8Linear
        num_fp8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        print(f"Converted {num_fp8}/{num_linear} Linear layers to Float8Linear")
    except:
        pass
    
    return model


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------

def train(args):
    print("="*70)
    print("FP8 TRAINING - FULL FEATURED")
    print("="*70)
    
    # Setup
    device = 'cuda'
    torch.cuda.set_device(0)
    
    # Mode string for logging
    if args.use_fp8:
        mode_str = f"fp8_{args.fp8_recipe}"
    else:
        mode_str = "bf16"
    
    print(f"\n{'Configuration':-^50}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Mode: {mode_str}")
    print(f"Steps: {args.steps} (optimizer steps)")
    print(f"Micro batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum} sequences")
    print(f"Sequence length: {args.seq_len}")
    print(f"Tokens per step: {args.batch_size * args.grad_accum * args.seq_len:,}")
    print(f"Learning rate: {args.lr} (warmup: {args.warmup_steps} steps, min: {args.min_lr})")
    print(f"Validation every: {args.val_every} steps")
    
    # Create model config
    if args.model_size:
        config = GPTConfig.from_size(args.model_size)
        print(f"Model preset: {args.model_size}")
    else:
        config = GPTConfig()
    
    # Create model
    model = GPT(config).to(torch.bfloat16).to(device)
    num_params = model.count_parameters()
    
    print(f"\n{'Model':-^50}")
    print(f"Parameters: {num_params/1e9:.2f}B ({num_params/1e6:.1f}M)")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head} (KV: {config.n_kv_head})")
    print(f"Embed dim: {config.n_embd}")
    
    # Apply FP8 if requested
    if args.use_fp8:
        print(f"\n{'FP8 Conversion':-^50}")
        model = apply_fp8(model, args.fp8_recipe)
    
    # Resume from checkpoint if specified (BEFORE compile!)
    start_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n{'Resuming from checkpoint':-^50}")
        if os.path.exists(args.resume):
            # weights_only=False needed because we save GPTConfig dataclass
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # Load model weights (handle compiled model state dict)
            state_dict = checkpoint['model_state_dict']
            # Remove _orig_mod. prefix if present (from torch.compile)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('_orig_mod.', '')
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=True)
            
            start_step = checkpoint['step']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            print(f"  Loaded model weights ✓")
            print(f"  Resumed from step {start_step}")
            print(f"  Previous val_loss: {best_val_loss:.4f}")
            print(f"  Remaining steps: {args.steps - start_step}")
        else:
            print(f"  Warning: Checkpoint not found: {args.resume}")
            print(f"  Starting from scratch")
    
    # Compile model (AFTER loading weights)
    print(f"\n{'Compilation':-^50}")
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)
    
    # Data loaders
    print(f"\n{'Data':-^50}")
    try:
        train_loader = FileDataLoader(
            args.train_data, B=args.batch_size, T=args.seq_len
        )
        val_loader = FileDataLoader(
            args.val_data, B=args.batch_size, T=args.seq_len
        )
        print(f"Train data: {args.train_data}")
        print(f"Val data: {args.val_data}")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        print("Using random data for testing")
        train_loader = SimpleDataLoader(
            B=args.batch_size, T=args.seq_len,
            vocab_size=config.vocab_size, device=device
        )
        val_loader = train_loader  # Use same for testing
    
    # Calculate tokens per step and total
    # tokens_per_step = effective batch size * seq_len = micro_batch * grad_accum * seq_len
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    total_tokens = args.steps * tokens_per_step
    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Total tokens: {total_tokens/1e6:.1f}M ({total_tokens/1e9:.2f}B)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Load optimizer state if resuming
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  Loaded optimizer state ✓")
            except Exception as e:
                print(f"  Warning: Could not load optimizer state: {e}")
                print(f"  Continuing with fresh optimizer")
    
    # Setup logging
    log_file = args.log_file or f"train_{mode_str}_{args.model_size or 'custom'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    checkpoint_path = args.checkpoint_path or f"checkpoint_{mode_str}_{args.model_size or 'custom'}.pt"
    
    print(f"\n{'Logging':-^50}")
    print(f"Log file: {log_file}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint strategy: Save best validation loss (overwrites)")
    
    # Determine if we're appending to existing log
    append_log = args.resume and os.path.exists(log_file)
    
    csv_file = open(log_file, 'a' if append_log else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    if not append_log:
        csv_writer.writerow([
            'step', 'train_loss', 'val_loss', 'perplexity', 'val_perplexity',
            'tokens_per_sec', 'step_time_ms', 'train_time_ms',
            'total_tokens', 'elapsed_time', 'train_time_total',
            'lr', 'grad_norm',
            'mode', 'model_size', 'batch_size', 'grad_accum', 'effective_batch', 'num_params'
        ])
    else:
        print(f"  Appending to existing log file")
    
    # Training loop
    print(f"\n{'='*70}")
    if start_step > 0:
        print(f"Resuming training from step {start_step + 1}...")
    else:
        print("Starting training...")
    print(f"{'='*70}\n")
    
    model.train()
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    
    total_tokens_processed = start_step * tokens_per_step  # Account for already processed tokens
    start_time = time.time()
    train_time_total = 0.0  # Only training time, excludes validation
    
    # For gradient accumulation, we track accumulated loss
    accumulated_loss = 0.0
    
    for step in range(start_step + 1, args.steps + 1):
        step_start = time.time()
        
        # Get learning rate for this step
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation loop
        accumulated_loss = 0.0
        for micro_step in range(args.grad_accum):
            x, y = train_loader.next_batch()
            
            with amp_ctx:
                _, loss = model(x, y)
                # Scale loss by accumulation steps for correct gradient averaging
                scaled_loss = loss / args.grad_accum
            
            scaled_loss.backward()
            accumulated_loss += loss.item()
        
        # Average the accumulated loss for logging
        loss_val = accumulated_loss / args.grad_accum
        
        # Compute gradient norm before clipping
        grad_norm = compute_gradient_norm(model)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        train_time_total += step_time
        
        # Metrics
        total_tokens_processed += tokens_per_step
        tokens_per_sec = tokens_per_step / step_time
        perplexity = math.exp(loss_val) if loss_val < 20 else float('inf')
        elapsed = time.time() - start_time
        
        # Validation
        val_loss = None
        val_perplexity = None
        if args.val_every > 0 and step % args.val_every == 0:
            val_start = time.time()
            val_loss = evaluate(model, val_loader, num_batches=args.val_batches)
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
            val_time = time.time() - val_start
            
            # Save checkpoint if validation improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': loss_val,
                    'config': config,
                    'args': vars(args),
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"  → Checkpoint saved (val_loss: {val_loss:.4f})")
            
            # Note: val_time is NOT added to train_time_total
        
        # Log to CSV
        csv_writer.writerow([
            step,
            f"{loss_val:.4f}",
            f"{val_loss:.4f}" if val_loss else "",
            f"{perplexity:.2f}",
            f"{val_perplexity:.2f}" if val_perplexity else "",
            f"{tokens_per_sec:.0f}",
            f"{step_time*1000:.1f}",
            f"{step_time*1000:.1f}",  # train_time_ms (same as step_time when no val)
            total_tokens_processed,
            f"{elapsed:.1f}",
            f"{train_time_total:.1f}",
            f"{lr:.2e}",
            f"{grad_norm:.4f}",
            mode_str,
            args.model_size or 'custom',
            args.batch_size,
            args.grad_accum,
            args.batch_size * args.grad_accum,
            num_params
        ])
        
        # Flush periodically
        if step % 100 == 0:
            csv_file.flush()
        
        # Console logging
        if step % args.log_every == 0 or step == 1:
            val_str = f" | val: {val_loss:.4f}" if val_loss else ""
            print(f"step {step:6d}/{args.steps} | loss: {loss_val:.4f}{val_str} | "
                  f"ppl: {perplexity:.1f} | lr: {lr:.2e} | grad: {grad_norm:.2f} | "
                  f"tok/s: {tokens_per_sec:.0f} | "
                  f"tokens: {total_tokens_processed/1e6:.1f}M | "
                  f"train_time: {format_time(train_time_total)} | "
                  f"wall_time: {format_time(elapsed)}")
    
    csv_file.close()
    
    # Final validation
    print(f"\n{'Final Validation':-^50}")
    final_val_loss = evaluate(model, val_loader, num_batches=100)
    final_val_ppl = math.exp(final_val_loss)
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation perplexity: {final_val_ppl:.2f}")
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total steps: {args.steps}")
    print(f"Total tokens: {total_tokens_processed/1e6:.1f}M")
    print(f"Training time: {format_time(train_time_total)}")
    print(f"Wall time: {format_time(total_time)}")
    print(f"Average throughput: {total_tokens_processed/train_time_total:.0f} tokens/sec")
    print(f"Final train loss: {loss_val:.4f}")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Mode: {mode_str}")
    print(f"Log saved to: {log_file}")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"{'='*70}")
    
    return {
        'train_loss': loss_val,
        'val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'throughput': total_tokens_processed / train_time_total,
        'train_time': train_time_total,
        'wall_time': total_time,
        'total_tokens': total_tokens_processed,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FP8 Training - Full Featured')
    
    # Training params
    parser.add_argument("--steps", type=int, default=15000, help="Training steps (optimizer steps, not micro-steps)")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro batch size (sequences per forward pass)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    
    # Learning rate
    parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Min learning rate")
    parser.add_argument("--warmup-frac", type=float, default=0.02, help="Warmup as fraction of total steps (default: 2%%)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps (overrides warmup-frac if set)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    
    # Model
    parser.add_argument("--model-size", type=str, default="1.5B",
                        help="Model size: 125M, 350M, 760M, 1.5B, 3B, 7B, 13B")
    
    # Data
    parser.add_argument("--train-data", type=str, 
                        default="data/pretrain/train_*.bin",
                        help="Training data pattern")
    parser.add_argument("--val-data", type=str,
                        default="data/pretrain/val_*.bin", 
                        help="Validation data pattern")
    
    # Validation
    parser.add_argument("--val-every", type=int, default=500, help="Validate every N steps (0 to disable)")
    parser.add_argument("--val-batches", type=int, default=50, help="Batches per validation")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--log-file", type=str, default=None, help="CSV log file")
    parser.add_argument("--checkpoint-path", type=str, default=None, 
                        help="Checkpoint path (default: auto-generated, overwrites on improvement)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file")
    
    # FP8
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise",
                        choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"])
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 (BF16 baseline)")
    
    args = parser.parse_args()
    args.use_fp8 = not args.no_fp8
    
    # Compute warmup steps from fraction if not explicitly set
    if args.warmup_steps is None:
        args.warmup_steps = int(args.warmup_frac * args.steps)
    
    train(args)
