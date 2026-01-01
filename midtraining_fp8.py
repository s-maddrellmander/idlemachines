"""
midtrain_fp8.py
Mid-Training Script - Task-focused training on pretrained checkpoint
=====================================================================

Mid-training bridges pretraining and chat SFT by training on structured
task data (instructions, QA, reasoning) while maintaining FP8 precision.

Key differences from pretraining:
- Loads from pretrained checkpoint
- Conversation-formatted data (instruction/response pairs)
- Lower learning rate, shorter schedule
- Task mixture: SmolTalk, MMLU, GSM8K

Usage:
    # Single GPU
    python midtrain_fp8.py --checkpoint pretrain_checkpoint.pt --steps 5000
    
    # Multi-GPU
    torchrun --nproc_per_node=8 midtrain_fp8.py --checkpoint pretrain_checkpoint.pt --steps 5000
    
    # BF16 baseline
    torchrun --nproc_per_node=8 midtrain_fp8.py --checkpoint pretrain_checkpoint.pt --no-fp8
"""

import os
import sys
import math
import glob
import time
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import tiktoken

# FP8 imports
try:
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("WARNING: torchao not available")

# HuggingFace datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("WARNING: datasets not available, install with: pip install datasets")

import torch._dynamo
torch._dynamo.config.optimize_ddp = False


# =============================================================================
# DDP Utilities (duplicated from train_gpt_fp8_ddp.py)
# =============================================================================

def setup_ddp():
    """Initialize DDP if running in distributed mode."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, local_rank, world_size
    else:
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


# =============================================================================
# Model Components (duplicated from train_gpt_fp8_ddp.py)
# =============================================================================

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
        
        x = self.transformer.wte(idx) 
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


# =============================================================================
# Tokenizer and Conversation Rendering
# =============================================================================

class ConversationTokenizer:
    """
    Simple conversation tokenizer using tiktoken GPT-2.
    
    Format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
    """
    
    # Special tokens
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab  # 50257 for GPT-2
        
        # We'll encode special tokens as text since tiktoken doesn't have them
        # In practice you'd add these to the tokenizer, but for simplicity we use text markers
        self.im_start_tokens = self.enc.encode(self.IM_START)
        self.im_end_tokens = self.enc.encode(self.IM_END)
        self.newline_token = self.enc.encode("\n")
    
    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={'<|endoftext|>'})
    
    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
    
    def render_conversation(self, conversation: List[Dict[str, str]]) -> List[int]:
        """
        Render a conversation to token IDs.
        
        Args:
            conversation: List of {"role": "user"|"assistant"|"system", "content": "..."}
        
        Returns:
            List of token IDs
        """
        tokens = []
        
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            # <|im_start|>role\n
            tokens.extend(self.im_start_tokens)
            tokens.extend(self.encode(role))
            tokens.extend(self.newline_token)
            
            # content<|im_end|>\n
            tokens.extend(self.encode(content))
            tokens.extend(self.im_end_tokens)
            tokens.extend(self.newline_token)
        
        return tokens
    
    def render_instruction(self, instruction: str, response: str, 
                          system: Optional[str] = None) -> List[int]:
        """Convenience method for instruction-response pairs."""
        conversation = []
        if system:
            conversation.append({"role": "system", "content": system})
        conversation.append({"role": "user", "content": instruction})
        conversation.append({"role": "assistant", "content": response})
        return self.render_conversation(conversation)


# =============================================================================
# Task Datasets
# =============================================================================

class TaskDataset:
    """Base class for task datasets."""
    
    def __init__(self, name: str):
        self.name = name
        self._data = []
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        """Return conversation format: [{"role": ..., "content": ...}, ...]"""
        raise NotImplementedError


class SmolTalkDataset(TaskDataset):
    """
    SmolTalk dataset - general instruction following conversations.
    https://huggingface.co/datasets/HuggingFaceTB/smoltalk
    """
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        super().__init__("smoltalk")
        
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required: pip install datasets")
        
        print0(f"Loading SmolTalk ({split})...")
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        self._data = ds
        print0(f"  Loaded {len(self._data)} samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        item = self._data[idx]
        # SmolTalk has 'messages' field with conversation
        return item['messages']


class MMLUDataset(TaskDataset):
    """
    MMLU dataset - multiple choice questions.
    Formats as instruction with choices.
    """
    
    def __init__(self, split: str = "auxiliary_train", max_samples: Optional[int] = None):
        super().__init__("mmlu")
        
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required: pip install datasets")
        
        print0(f"Loading MMLU ({split})...")
        
        # MMLU auxiliary_train is good for training
        if split == "auxiliary_train":
            ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
        else:
            # For validation, use a subset of subjects
            ds = load_dataset("cais/mmlu", "all", split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        self._data = ds
        print0(f"  Loaded {len(self._data)} samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        item = self._data[idx]
        
        # Format: question with choices
        question = item['question']
        choices = item['choices']
        answer_idx = item['answer']
        
        # Build instruction
        choice_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        instruction = f"{question}\n\n{choice_text}"
        
        # Answer is the letter
        response = f"The answer is {chr(65 + answer_idx)}."
        
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]


class GSM8KDataset(TaskDataset):
    """
    GSM8K dataset - grade school math problems.
    """
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        super().__init__("gsm8k")
        
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required: pip install datasets")
        
        print0(f"Loading GSM8K ({split})...")
        ds = load_dataset("openai/gsm8k", "main", split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        self._data = ds
        print0(f"  Loaded {len(self._data)} samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        item = self._data[idx]
        
        question = item['question']
        answer = item['answer']  # Contains step-by-step solution
        
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]


class TaskMixture:
    """
    Mixture of task datasets with proportional sampling.
    """
    
    def __init__(self, datasets: List[TaskDataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        # Build index mapping
        self._indices = []  # List of (dataset_idx, item_idx)
        for ds_idx, ds in enumerate(datasets):
            for item_idx in range(len(ds)):
                self._indices.append((ds_idx, item_idx))
        
        # Shuffle
        np.random.shuffle(self._indices)
        
        total_samples = sum(len(ds) for ds in datasets)
        print0(f"TaskMixture: {len(datasets)} datasets, {total_samples} total samples")
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        ds_idx, item_idx = self._indices[idx % len(self._indices)]
        return self.datasets[ds_idx][item_idx]


# =============================================================================
# Mid-Training Data Loader
# =============================================================================

class MidTrainDataLoader:
    """
    Data loader for mid-training that handles conversation rendering
    and sequence packing (similar to NanoChat approach).
    """
    
    def __init__(
        self,
        dataset: TaskMixture,
        tokenizer: ConversationTokenizer,
        batch_size: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        device: str = 'cuda'
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        self.needed_tokens = batch_size * seq_len + 1
        self.token_buffer = deque()
        self.cursor = rank  # Each rank starts at different position
        self.epoch = 0
        
        # For tracking progress
        self.total_samples = len(dataset)
        self.samples_seen = 0
    
    def reset(self):
        """Reset to beginning."""
        self.token_buffer.clear()
        self.cursor = self.rank
        self.epoch = 0
        self.samples_seen = 0
    
    def _fill_buffer(self):
        """Fill token buffer from dataset."""
        while len(self.token_buffer) < self.needed_tokens:
            # Get conversation and tokenize
            conversation = self.dataset[self.cursor]
            tokens = self.tokenizer.render_conversation(conversation)
            self.token_buffer.extend(tokens)
            
            # Move cursor (skip by world_size to avoid overlap between ranks)
            self.cursor += self.world_size
            self.samples_seen += 1
            
            if self.cursor >= self.total_samples:
                self.cursor = self.cursor % self.total_samples
                self.epoch += 1
    
    def next_batch(self):
        """Get next batch of (inputs, targets)."""
        self._fill_buffer()
        
        # Extract tokens for this batch
        tokens = []
        for _ in range(self.needed_tokens):
            tokens.append(self.token_buffer.popleft())
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1].view(self.batch_size, self.seq_len).to(self.device)
        y = tokens[1:].view(self.batch_size, self.seq_len).to(self.device)
        
        return x, y
    
    @property
    def progress(self) -> float:
        """Approximate progress through dataset."""
        return self.samples_seen / self.total_samples


# =============================================================================
# FP8 Conversion
# =============================================================================

def apply_fp8(model, recipe="tensorwise"):
    """Convert model to FP8 training."""
    if not TORCHAO_AVAILABLE:
        print0("torchao not available, skipping FP8")
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


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# =============================================================================
# Validation
# =============================================================================

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
    
    if ddp:
        loss_tensor = torch.tensor([avg_loss], device='cuda')
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    model.train()
    return avg_loss


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_pretrained_checkpoint(model, checkpoint_path, device):
    """Load pretrained checkpoint into model."""
    print0(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    
    # Remove prefixes from DDP/compile
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '').replace('module.', '')
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    
    # Return metadata
    return {
        'step': checkpoint.get('step', 0),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
        'config': checkpoint.get('config', None),
    }


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    # Setup DDP
    ddp, rank, local_rank, world_size = setup_ddp()
    device = f'cuda:{local_rank}'
    is_master = (rank == 0)
    
    print0("=" * 70)
    print0("MID-TRAINING - FP8")
    print0("=" * 70)
    
    # Mode string
    mode_str = f"fp8_{args.fp8_recipe}" if args.use_fp8 else "bf16"
    
    print0(f"\n{'Configuration':-^50}")
    print0(f"Device: {torch.cuda.get_device_name()} x {world_size}")
    print0(f"DDP: {ddp} (rank {rank}/{world_size})")
    print0(f"Mode: {mode_str}")
    print0(f"Pretrained checkpoint: {args.checkpoint}")
    print0(f"Steps: {args.steps}")
    print0(f"Batch size: {args.batch_size} per GPU")
    print0(f"Gradient accumulation: {args.grad_accum}")
    print0(f"Effective batch: {args.batch_size * args.grad_accum * world_size}")
    print0(f"Sequence length: {args.seq_len}")
    
    tokens_per_step = args.batch_size * args.grad_accum * world_size * args.seq_len
    print0(f"Tokens per step: {tokens_per_step:,}")
    
    # Create model
    if args.model_size:
        config = GPTConfig.from_size(args.model_size)
        print0(f"Model preset: {args.model_size}")
    else:
        config = GPTConfig()
    
    model = GPT(config).to(torch.bfloat16).to(device)
    num_params = model.count_parameters()
    
    print0(f"\n{'Model':-^50}")
    print0(f"Parameters: {num_params/1e9:.2f}B ({num_params/1e6:.1f}M)")
    print0(f"Layers: {config.n_layer}")
    print0(f"Heads: {config.n_head} (KV: {config.n_kv_head})")
    print0(f"Embed dim: {config.n_embd}")
    
    # Load pretrained checkpoint
    print0(f"\n{'Loading Pretrained':-^50}")
    pretrain_meta = load_pretrained_checkpoint(model, args.checkpoint, device)
    print0(f"  Pretrain step: {pretrain_meta['step']}")
    if pretrain_meta['val_loss']:
        print0(f"  Pretrain val_loss: {pretrain_meta['val_loss']:.4f}")
    
    # Apply FP8 (BEFORE DDP)
    if args.use_fp8:
        print0(f"\n{'FP8 Conversion':-^50}")
        model = apply_fp8(model, args.fp8_recipe)
    
    # Wrap in DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    
    # Compile
    print0(f"\n{'Compilation':-^50}")
    print0("Compiling model...")
    model = torch.compile(model, fullgraph=False)
    
    # Tokenizer
    tokenizer = ConversationTokenizer()
    print0(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Datasets
    print0(f"\n{'Datasets':-^50}")
    
    train_datasets = []
    val_datasets = []
    
    if args.use_smoltalk:
        train_datasets.append(SmolTalkDataset("train", max_samples=args.max_samples))
        val_datasets.append(SmolTalkDataset("test", max_samples=5000))
    
    if args.use_mmlu:
        train_datasets.append(MMLUDataset("auxiliary_train", max_samples=args.max_samples))
        # MMLU validation uses test split
        val_datasets.append(MMLUDataset("validation", max_samples=2000))
    
    if args.use_gsm8k:
        train_datasets.append(GSM8KDataset("train", max_samples=args.max_samples))
        val_datasets.append(GSM8KDataset("test", max_samples=500))
    
    if not train_datasets:
        print0("No datasets specified! Using SmolTalk by default.")
        train_datasets.append(SmolTalkDataset("train", max_samples=args.max_samples))
        val_datasets.append(SmolTalkDataset("test", max_samples=5000))
    
    train_mixture = TaskMixture(train_datasets)
    val_mixture = TaskMixture(val_datasets)
    
    # Data loaders
    train_loader = MidTrainDataLoader(
        train_mixture, tokenizer, 
        batch_size=args.batch_size, seq_len=args.seq_len,
        rank=rank, world_size=world_size, device=device
    )
    val_loader = MidTrainDataLoader(
        val_mixture, tokenizer,
        batch_size=args.batch_size, seq_len=args.seq_len,
        rank=rank, world_size=world_size, device=device
    )
    
    print0(f"Train samples: {len(train_mixture)}")
    print0(f"Val samples: {len(val_mixture)}")
    
    # Optimizer (lower LR than pretraining)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=True
    )
    
    # Logging setup
    log_file = args.log_file or f"midtrain_{mode_str}_{args.model_size or 'custom'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    checkpoint_path = args.output_checkpoint or f"midtrain_{mode_str}_{args.model_size or 'custom'}.pt"
    
    csv_file = None
    csv_writer = None
    if is_master:
        print0(f"\n{'Logging':-^50}")
        print0(f"Log file: {log_file}")
        print0(f"Output checkpoint: {checkpoint_path}")
        
        csv_file = open(log_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'step', 'train_loss', 'val_loss', 'perplexity', 'val_perplexity',
            'tokens_per_sec', 'step_time_ms', 'total_tokens', 'elapsed_time',
            'lr', 'grad_norm', 'epoch', 'mode', 'model_size'
        ])
    
    # Training loop
    print0(f"\n{'='*70}")
    print0("Starting mid-training...")
    print0(f"{'='*70}\n")
    
    model.train()
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    
    total_tokens = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    for step in range(1, args.steps + 1):
        step_start = time.time()
        
        # Learning rate
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(args.grad_accum):
            x, y = train_loader.next_batch()
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum - 1)
            
            with amp_ctx:
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum
            
            scaled_loss.backward()
            accumulated_loss += loss.item()
        
        loss_val = accumulated_loss / args.grad_accum
        
        # Average across ranks
        if ddp:
            loss_tensor = torch.tensor([loss_val], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_val = loss_tensor.item()
        
        # Gradient norm and clipping
        grad_norm = compute_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        
        # Metrics
        total_tokens += tokens_per_step
        tokens_per_sec = tokens_per_step / step_time
        perplexity = math.exp(loss_val) if loss_val < 20 else float('inf')
        elapsed = time.time() - start_time
        
        # Validation
        val_loss = None
        val_perplexity = None
        if args.val_every > 0 and step % args.val_every == 0:
            val_loss = evaluate(model, val_loader, num_batches=args.val_batches, ddp=ddp)
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
            
            if is_master and val_loss < best_val_loss:
                best_val_loss = val_loss
                model_state = model.module.state_dict() if ddp else model.state_dict()
                checkpoint = {
                    'step': step,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': loss_val,
                    'config': config,
                    'pretrain_checkpoint': args.checkpoint,
                    'args': vars(args),
                }
                torch.save(checkpoint, checkpoint_path)
                print0(f"  → Checkpoint saved (val_loss: {val_loss:.4f})")
        
        # CSV logging
        if is_master and csv_writer:
            csv_writer.writerow([
                step, f"{loss_val:.4f}",
                f"{val_loss:.4f}" if val_loss else "",
                f"{perplexity:.2f}",
                f"{val_perplexity:.2f}" if val_perplexity else "",
                f"{tokens_per_sec:.0f}", f"{step_time*1000:.1f}",
                total_tokens, f"{elapsed:.1f}",
                f"{lr:.2e}", f"{grad_norm:.4f}",
                train_loader.epoch, mode_str, args.model_size or 'custom'
            ])
            if step % 100 == 0:
                csv_file.flush()
        
        # Console logging
        if step % args.log_every == 0 or step == 1:
            val_str = f" | val: {val_loss:.4f}" if val_loss else ""
            print0(f"step {step:5d}/{args.steps} | loss: {loss_val:.4f}{val_str} | "
                  f"ppl: {perplexity:.1f} | lr: {lr:.2e} | grad: {grad_norm:.2f} | "
                  f"tok/s: {tokens_per_sec:.0f} | epoch: {train_loader.epoch} | "
                  f"time: {format_time(elapsed)}")
    
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
    print0("MID-TRAINING COMPLETE")
    print0(f"{'='*70}")
    print0(f"Total steps: {args.steps}")
    print0(f"Total tokens: {total_tokens/1e6:.1f}M")
    print0(f"Training time: {format_time(total_time)}")
    print0(f"Throughput: {total_tokens/total_time:.0f} tokens/sec")
    print0(f"Final train loss: {loss_val:.4f}")
    print0(f"Final val loss: {final_val_loss:.4f}")
    print0(f"Best val loss: {best_val_loss:.4f}")
    print0(f"Mode: {mode_str}")
    if is_master:
        print0(f"Checkpoint: {checkpoint_path}")
    print0(f"{'='*70}")
    
    cleanup_ddp()
    
    return {
        'train_loss': loss_val,
        'val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mid-Training - FP8')
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    
    # Training
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    
    # Learning rate (lower than pretraining)
    parser.add_argument("--lr", type=float, default=1e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Min learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    
    # Model
    parser.add_argument("--model-size", type=str, default="1.5B",
                        help="Model size: 125M, 350M, 760M, 1.5B, 3B, 7B")
    
    # Datasets
    parser.add_argument("--use-smoltalk", action="store_true", default=True,
                        help="Use SmolTalk dataset")
    parser.add_argument("--use-mmlu", action="store_true", default=True,
                        help="Use MMLU dataset")
    parser.add_argument("--use-gsm8k", action="store_true", default=True,
                        help="Use GSM8K dataset")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (for testing)")
    
    # Validation
    parser.add_argument("--val-every", type=int, default=200, help="Validate every N steps")
    parser.add_argument("--val-batches", type=int, default=50, help="Batches per validation")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=20, help="Log every N steps")
    parser.add_argument("--log-file", type=str, default=None, help="CSV log file")
    parser.add_argument("--output-checkpoint", type=str, default=None, help="Output checkpoint")
    
    # FP8
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise",
                        choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"])
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8")
    
    args = parser.parse_args()
    args.use_fp8 = not args.no_fp8
    
    train(args)