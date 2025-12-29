"""
train_gpt_fp8_ddp.py
FP8 Training Script - DDP Multi-GPU Version
============================================

Complete training script with:
- Distributed Data Parallel (DDP) for multi-GPU training
- Validation loss tracking
- Gradient norm logging
- Learning rate schedule (warmup + cosine decay)
- Comprehensive CSV logging
- Multiple FP8 recipes support

Usage:
    # Single GPU (falls back gracefully)
    python train_gpt_fp8_ddp.py --model-size 1.5B --steps 15000
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 train_gpt_fp8_ddp.py --model-size 1.5B --steps 15000
    
    # BF16 baseline
    torchrun --nproc_per_node=8 train_gpt_fp8_ddp.py --model-size 1.5B --steps 15000 --no-fp8
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# FP8 imports
try:
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("WARNING: torchao not available")

import torch._dynamo
torch._dynamo.config.optimize_ddp = False

# -----------------------------------------------------------------------------
# DDP Utilities
# -----------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP if running in distributed mode."""
    # Check if we're running under torchrun
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, local_rank, world_size
    else:
        # Single GPU mode
        torch.cuda.set_device(0)
        return False, 0, 0, 1


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print0(*args, **kwargs):
    """Print only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


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
            "250M":  (14,  1024, 16, 4),
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
        if isinstance(module, nn.Linear):
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
# Data Loading with DDP Support
# -----------------------------------------------------------------------------

class FileDataLoader:
    """
    Load from binary files with DDP support.
    
    Each rank reads from different shards or different positions within shards
    to ensure no overlap in training data.
    """
    def __init__(self, filename_pattern, B, T, rank=0, world_size=1):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
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
        
        if rank == 0:
            print(f"Data: {len(self.files)} shards, ~{total_tokens/1e9:.2f}B tokens total")
        self.total_tokens = total_tokens
        
        # Each rank starts at a different shard (round-robin assignment)
        self.current_shard = rank % len(self.files)
        self.current_position = 0
        self._load_shard()
    
    def _load_shard(self):
        with open(self.files[self.current_shard], "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            self.tokens = np.frombuffer(f.read(), dtype=np.uint16)
    
    def reset(self):
        """Reset to beginning of data for this rank."""
        self.current_shard = self.rank % len(self.files)
        self.current_position = 0
        self._load_shard()
    
    def next_batch(self):
        buf = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(self.B, self.T).cuda()
        y = buf[1:].view(self.B, self.T).cuda()
        
        self.current_position += self.B * self.T
        if self.current_position + self.B * self.T + 1 > len(self.tokens):
            # Move to next shard (skip by world_size to avoid overlap)
            self.current_shard = (self.current_shard + self.world_size) % len(self.files)
            self.current_position = 0
            self._load_shard()
        
        return x, y


class SimpleDataLoader:
    """Random data for testing."""
    def __init__(self, B, T, vocab_size, device='cuda'):
        self.B = B
        self.T = T
        self.vocab_size = vocab_size
        self.device = device
        self.total_tokens = 10_000_000_000
    
    def next_batch(self):
        x = torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device)
        y = torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device)
        return x, y
    
    def reset(self):
        pass


# -----------------------------------------------------------------------------
# Learning Rate Schedule
# -----------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
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
def evaluate(model, val_loader, num_batches=50, ddp=False):
    """Run validation and return average loss."""
    model.eval()
    val_loader.reset()
    
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = val_loader.next_batch()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    
    # Average across all ranks
    if ddp:
        loss_tensor = torch.tensor([avg_loss], device='cuda')
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    model.train()
    return avg_loss


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
        print0(f"Converted {num_fp8}/{num_linear} Linear layers to Float8Linear")
    except:
        pass
    
    return model


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------

def train(args):
    # Setup DDP
    ddp, rank, local_rank, world_size = setup_ddp()
    device = f'cuda:{local_rank}'
    is_master = (rank == 0)
    
    print0("="*70)
    print0("FP8 TRAINING - DDP MULTI-GPU")
    print0("="*70)
    
    # Mode string for logging
    if args.use_fp8:
        mode_str = f"fp8_{args.fp8_recipe}"
    else:
        mode_str = "bf16"
    
    print0(f"\n{'Configuration':-^50}")
    print0(f"Device: {torch.cuda.get_device_name()} x {world_size}")
    print0(f"DDP: {ddp} (rank {rank}/{world_size})")
    print0(f"Mode: {mode_str}")
    print0(f"Steps: {args.steps} (optimizer steps)")
    print0(f"Micro batch size: {args.batch_size} per GPU")
    print0(f"Gradient accumulation: {args.grad_accum}")
    print0(f"Effective batch size: {args.batch_size * args.grad_accum * world_size} sequences")
    print0(f"Sequence length: {args.seq_len}")
    tokens_per_step = args.batch_size * args.grad_accum * world_size * args.seq_len
    print0(f"Tokens per step: {tokens_per_step:,}")
    print0(f"Learning rate: {args.lr} (warmup: {args.warmup_steps} steps, min: {args.min_lr})")
    print0(f"Validation every: {args.val_every} steps")
    
    # Create model config
    if args.model_size:
        config = GPTConfig.from_size(args.model_size)
        print0(f"Model preset: {args.model_size}")
    else:
        config = GPTConfig()
    
    # Create model
    model = GPT(config).to(torch.bfloat16).to(device)
    num_params = model.count_parameters()
    
    print0(f"\n{'Model':-^50}")
    print0(f"Parameters: {num_params/1e9:.2f}B ({num_params/1e6:.1f}M)")
    print0(f"Layers: {config.n_layer}")
    print0(f"Heads: {config.n_head} (KV: {config.n_kv_head})")
    print0(f"Embed dim: {config.n_embd}")
    
    # Apply FP8 if requested (BEFORE DDP wrapping)
    if args.use_fp8:
        print0(f"\n{'FP8 Conversion':-^50}")
        model = apply_fp8(model, args.fp8_recipe)
    
    # Resume from checkpoint if specified (BEFORE compile and DDP!)
    start_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print0(f"\n{'Resuming from checkpoint':-^50}")
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            state_dict = checkpoint['model_state_dict']
            # Remove _orig_mod. prefix if present (from torch.compile)
            # Remove module. prefix if present (from DDP)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('_orig_mod.', '').replace('module.', '')
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=True)
            
            start_step = checkpoint['step']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            print0(f"  Loaded model weights ✓")
            print0(f"  Resumed from step {start_step}")
            print0(f"  Previous val_loss: {best_val_loss:.4f}")
            print0(f"  Remaining steps: {args.steps - start_step}")
        else:
            print0(f"  Warning: Checkpoint not found: {args.resume}")
            print0(f"  Starting from scratch")
    
    
    # Wrap in DDP (AFTER compile)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    
    # Compile model (AFTER loading weights, BEFORE DDP)
    print0(f"\n{'Compilation':-^50}")
    print0("Compiling model with torch.compile()...")
    model = torch.compile(model, fullgraph=False)

    # Data loaders (each rank gets different data)
    print0(f"\n{'Data':-^50}")
    try:
        train_loader = FileDataLoader(
            args.train_data, B=args.batch_size, T=args.seq_len,
            rank=rank, world_size=world_size
        )
        val_loader = FileDataLoader(
            args.val_data, B=args.batch_size, T=args.seq_len,
            rank=rank, world_size=world_size
        )
        print0(f"Train data: {args.train_data}")
        print0(f"Val data: {args.val_data}")
    except FileNotFoundError as e:
        print0(f"Data not found: {e}")
        print0("Using random data for testing")
        train_loader = SimpleDataLoader(
            B=args.batch_size, T=args.seq_len,
            vocab_size=config.vocab_size, device=device
        )
        val_loader = train_loader
    
    total_tokens = args.steps * tokens_per_step
    print0(f"Tokens per optimizer step: {tokens_per_step:,}")
    print0(f"Total tokens: {total_tokens/1e6:.1f}M ({total_tokens/1e9:.2f}B)")
    
    # Optimizer
    # Note: DDP wraps the model, so we need to get parameters from the wrapped model
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=True  # Use fused AdamW for speed on CUDA
    )
    
    # Load optimizer state if resuming
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print0(f"  Loaded optimizer state ✓")
            except Exception as e:
                print0(f"  Warning: Could not load optimizer state: {e}")
                print0(f"  Continuing with fresh optimizer")
    
    # Setup logging (only on master)
    log_file = args.log_file or f"train_{mode_str}_{args.model_size or 'custom'}_{world_size}gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    checkpoint_path = args.checkpoint_path or f"checkpoint_{mode_str}_{args.model_size or 'custom'}_{world_size}gpu.pt"
    
    csv_file = None
    csv_writer = None
    if is_master:
        print0(f"\n{'Logging':-^50}")
        print0(f"Log file: {log_file}")
        print0(f"Checkpoint: {checkpoint_path}")
        
        append_log = args.resume and os.path.exists(log_file)
        csv_file = open(log_file, 'a' if append_log else 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        if not append_log:
            csv_writer.writerow([
                'step', 'train_loss', 'val_loss', 'perplexity', 'val_perplexity',
                'tokens_per_sec', 'step_time_ms', 'train_time_ms',
                'total_tokens', 'elapsed_time', 'train_time_total',
                'lr', 'grad_norm',
                'mode', 'model_size', 'batch_size', 'grad_accum', 'effective_batch', 
                'num_params', 'world_size'
            ])
    
    # Training loop
    print0(f"\n{'='*70}")
    if start_step > 0:
        print0(f"Resuming training from step {start_step + 1}...")
    else:
        print0("Starting training...")
    print0(f"{'='*70}\n")
    
    model.train()
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    
    total_tokens_processed = start_step * tokens_per_step
    start_time = time.time()
    train_time_total = 0.0
    
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
            
            # Only sync gradients on the last micro step
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum - 1)
            
            with amp_ctx:
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum
            
            scaled_loss.backward()
            accumulated_loss += loss.item()
        
        # Average loss for logging
        loss_val = accumulated_loss / args.grad_accum
        
        # Average loss across ranks for consistent logging
        if ddp:
            loss_tensor = torch.tensor([loss_val], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_val = loss_tensor.item()
        
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
            val_loss = evaluate(model, val_loader, num_batches=args.val_batches, ddp=ddp)
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
            
            # Save checkpoint if validation improved (only on master)
            if is_master and val_loss < best_val_loss:
                best_val_loss = val_loss
                # Get state dict (handle DDP wrapper)
                model_state = model.module.state_dict() if ddp else model.state_dict()
                checkpoint = {
                    'step': step,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': loss_val,
                    'config': config,
                    'args': vars(args),
                }
                torch.save(checkpoint, checkpoint_path)
                print0(f"  → Checkpoint saved (val_loss: {val_loss:.4f})")
        
        # Log to CSV (only on master)
        if is_master and csv_writer:
            csv_writer.writerow([
                step,
                f"{loss_val:.4f}",
                f"{val_loss:.4f}" if val_loss else "",
                f"{perplexity:.2f}",
                f"{val_perplexity:.2f}" if val_perplexity else "",
                f"{tokens_per_sec:.0f}",
                f"{step_time*1000:.1f}",
                f"{step_time*1000:.1f}",
                total_tokens_processed,
                f"{elapsed:.1f}",
                f"{train_time_total:.1f}",
                f"{lr:.2e}",
                f"{grad_norm:.4f}",
                mode_str,
                args.model_size or 'custom',
                args.batch_size,
                args.grad_accum,
                args.batch_size * args.grad_accum * world_size,
                num_params,
                world_size
            ])
            
            if step % 100 == 0:
                csv_file.flush()
        
        # Console logging (only on master)
        if step % args.log_every == 0 or step == 1:
            val_str = f" | val: {val_loss:.4f}" if val_loss else ""
            print0(f"step {step:6d}/{args.steps} | loss: {loss_val:.4f}{val_str} | "
                  f"ppl: {perplexity:.1f} | lr: {lr:.2e} | grad: {grad_norm:.2f} | "
                  f"tok/s: {tokens_per_sec:.0f} | "
                  f"tokens: {total_tokens_processed/1e6:.1f}M | "
                  f"train_time: {format_time(train_time_total)} | "
                  f"wall_time: {format_time(elapsed)}")
    
    if csv_file:
        csv_file.close()
    
    # Final validation
    print0(f"\n{'Final Validation':-^50}")
    final_val_loss = evaluate(model, val_loader, num_batches=100, ddp=ddp)
    final_val_ppl = math.exp(final_val_loss)
    print0(f"Final validation loss: {final_val_loss:.4f}")
    print0(f"Final validation perplexity: {final_val_ppl:.2f}")
    
    # Summary
    total_time = time.time() - start_time
    
    print0(f"\n{'='*70}")
    print0("TRAINING COMPLETE")
    print0(f"{'='*70}")
    print0(f"GPUs: {world_size}")
    print0(f"Total steps: {args.steps}")
    print0(f"Total tokens: {total_tokens_processed/1e6:.1f}M ({total_tokens_processed/1e9:.2f}B)")
    print0(f"Training time: {format_time(train_time_total)}")
    print0(f"Wall time: {format_time(total_time)}")
    print0(f"Average throughput: {total_tokens_processed/train_time_total:.0f} tokens/sec")
    print0(f"Per-GPU throughput: {total_tokens_processed/train_time_total/world_size:.0f} tokens/sec/GPU")
    print0(f"Final train loss: {loss_val:.4f}")
    print0(f"Final val loss: {final_val_loss:.4f}")
    print0(f"Best val loss: {best_val_loss:.4f}")
    print0(f"Mode: {mode_str}")
    if is_master:
        print0(f"Log saved to: {log_file}")
        print0(f"Best checkpoint saved to: {checkpoint_path}")
    print0(f"{'='*70}")
    
    # Cleanup
    cleanup_ddp()
    
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
    
    parser = argparse.ArgumentParser(description='FP8 Training - DDP Multi-GPU')
    
    # Training params
    parser.add_argument("--steps", type=int, default=15000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    
    # Learning rate
    parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Min learning rate")
    parser.add_argument("--warmup-frac", type=float, default=0.02, help="Warmup fraction")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps")
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
    parser.add_argument("--val-every", type=int, default=500, help="Validate every N steps")
    parser.add_argument("--val-batches", type=int, default=50, help="Batches per validation")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--log-file", type=str, default=None, help="CSV log file")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # FP8
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise",
                        choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"])
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 (BF16 baseline)")
    
    args = parser.parse_args()
    args.use_fp8 = not args.no_fp8
    
    if args.warmup_steps is None:
        args.warmup_steps = int(args.warmup_frac * args.steps)
    
    train(args)
