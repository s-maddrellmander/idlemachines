"""
chat_sft_fp8.py
Chat Supervised Fine-Tuning Script - FP8 Training
==================================================

Final stage of training pipeline: converts mid-trained model into a chat model.

Key differences from mid-training:
- Loss masking: Only compute loss on assistant responses, not user prompts
- Variable-length sequences with padding (no token packing)
- Smaller, curated dataset (~23K examples)
- No torch.compile (variable lengths don't work well)

Usage:
    # Single GPU
    python chat_sft_fp8.py --checkpoint midtrain_checkpoint.pt --steps 2000
    
    # Multi-GPU
    torchrun --nproc_per_node=8 chat_sft_fp8.py --checkpoint midtrain_checkpoint.pt --steps 2000
    
    # BF16 baseline
    python chat_sft_fp8.py --checkpoint midtrain_checkpoint.pt --no-fp8
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
from typing import List, Dict, Optional, Tuple, Iterator

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


# =============================================================================
# DDP Utilities
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
# Model Components (same as pretrain/midtrain)
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
            # targets can have -1 for masked positions (ignore_index)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1  # Ignore masked positions
            )
        else:
            loss = None
        
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Tokenizer with Loss Masking for SFT
# =============================================================================

class SFTTokenizer:
    """
    Tokenizer for chat SFT that returns both token IDs and loss masks.
    
    Loss mask = 1 for tokens we want to compute loss on (assistant responses)
    Loss mask = 0 for tokens we want to ignore (user prompts, system, special tokens)
    
    Format:
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
    """
    
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        
        # Pre-encode special token strings
        self.im_start_tokens = self.enc.encode(self.IM_START)
        self.im_end_tokens = self.enc.encode(self.IM_END)
        self.newline_token = self.enc.encode("\n")
        
        # For padding - use a token that won't affect loss (will be masked anyway)
        self.pad_token_id = self.im_end_tokens[0] if self.im_end_tokens else 0
    
    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={'<|endoftext|>'})
    
    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
    
    def render_conversation(self, conversation: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """
        Render a conversation to token IDs and loss mask.
        
        Args:
            conversation: List of {"role": "user"|"assistant"|"system", "content": "..."}
        
        Returns:
            (token_ids, loss_mask) where loss_mask[i] = 1 if we compute loss on token i
        """
        tokens = []
        mask = []
        
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            # Determine if this turn should have loss computed
            compute_loss = (role == "assistant")
            
            # <|im_start|>role\n - never compute loss on this
            turn_header = self.im_start_tokens + self.encode(role) + self.newline_token
            tokens.extend(turn_header)
            mask.extend([0] * len(turn_header))
            
            # content - only compute loss for assistant
            content_tokens = self.encode(content)
            tokens.extend(content_tokens)
            mask.extend([1 if compute_loss else 0] * len(content_tokens))
            
            # <|im_end|>\n - compute loss on im_end for assistant (teaches when to stop)
            turn_footer = self.im_end_tokens + self.newline_token
            tokens.extend(turn_footer)
            mask.extend([1 if compute_loss else 0] * len(turn_footer))
        
        return tokens, mask


# =============================================================================
# Task Datasets for SFT
# =============================================================================

class TaskDataset:
    """Base class for task datasets."""
    
    def __init__(self, name: str):
        self.name = name
        self._data = []
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        raise NotImplementedError


class SmolTalkDataset(TaskDataset):
    """SmolTalk - general instruction following."""
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        super().__init__("smoltalk")
        
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required")
        
        print0(f"Loading SmolTalk ({split})...")
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        self._data = ds
        print0(f"  Loaded {len(self._data)} samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        return self._data[idx]['messages']


class ARCDataset(TaskDataset):
    """ARC (AI2 Reasoning Challenge) - multiple choice science questions."""
    
    def __init__(self, subset: str = "ARC-Easy", split: str = "train", max_samples: Optional[int] = None):
        super().__init__(f"arc_{subset}")
        
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required")
        
        print0(f"Loading ARC {subset} ({split})...")
        ds = load_dataset("allenai/ai2_arc", subset, split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        self._data = ds
        print0(f"  Loaded {len(self._data)} samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        item = self._data[idx]
        
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']
        
        # Build choice text
        choice_labels = choices['label']
        choice_texts = choices['text']
        choice_str = "\n".join([f"{l}. {t}" for l, t in zip(choice_labels, choice_texts)])
        
        # Find answer text
        answer_idx = choice_labels.index(answer_key)
        answer_text = choice_texts[answer_idx]
        
        instruction = f"{question}\n\n{choice_str}"
        response = f"The answer is {answer_key}. {answer_text}"
        
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]


class GSM8KDataset(TaskDataset):
    """GSM8K - grade school math with step-by-step solutions."""
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        super().__init__("gsm8k")
        
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required")
        
        print0(f"Loading GSM8K ({split})...")
        ds = load_dataset("openai/gsm8k", "main", split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        self._data = ds
        print0(f"  Loaded {len(self._data)} samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        item = self._data[idx]
        return [
            {"role": "user", "content": item['question']},
            {"role": "assistant", "content": item['answer']}
        ]


class SimpleSpellingDataset(TaskDataset):
    """Simple spelling task: 'Spell the word apple' -> 'a-p-p-l-e'"""
    
    def __init__(self, size: int = 1000, split: str = "train", seed: int = 42):
        super().__init__("simple_spelling")
        
        # Common English words for spelling practice
        words = [
            "apple", "banana", "orange", "grape", "strawberry", "blueberry",
            "computer", "keyboard", "monitor", "mouse", "printer", "speaker",
            "elephant", "giraffe", "hippopotamus", "rhinoceros", "crocodile",
            "beautiful", "wonderful", "fantastic", "incredible", "magnificent",
            "temperature", "government", "environment", "development", "entertainment",
            "restaurant", "chocolate", "vegetable", "breakfast", "afternoon",
            "necessary", "definitely", "occasionally", "immediately", "unfortunately",
            "psychology", "philosophy", "technology", "archaeology", "bibliography",
        ]
        
        rng = np.random.RandomState(seed if split == "train" else seed + 1)
        
        self._data = []
        for _ in range(size):
            word = rng.choice(words)
            spelled = "-".join(list(word))
            self._data.append({
                "word": word,
                "spelled": spelled
            })
        
        print0(f"  Generated {len(self._data)} spelling samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        item = self._data[idx]
        return [
            {"role": "user", "content": f"Spell the word '{item['word']}'"},
            {"role": "assistant", "content": item['spelled']}
        ]


class SpellingBeeDataset(TaskDataset):
    """Spelling bee: 'How many r's are in strawberry?' -> '3'"""
    
    def __init__(self, size: int = 1000, split: str = "train", seed: int = 42):
        super().__init__("spelling_bee")
        
        words = [
            "strawberry", "raspberry", "blueberry", "blackberry", "cranberry",
            "occurrence", "reference", "preference", "conference", "difference",
            "mississippi", "committee", "accommodate", "recommend", "profession",
            "assessment", "possession", "ccessible", "successful", "ecessary",
            "parallel", "millennium", "bookkeeper", "addressee", "committee",
        ]
        
        rng = np.random.RandomState(seed if split == "train" else seed + 1)
        
        self._data = []
        for _ in range(size):
            word = rng.choice(words)
            # Pick a letter that appears in the word
            letter = rng.choice(list(set(word)))
            count = word.count(letter)
            self._data.append({
                "word": word,
                "letter": letter,
                "count": count
            })
        
        print0(f"  Generated {len(self._data)} spelling bee samples")
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        item = self._data[idx]
        return [
            {"role": "user", "content": f"How many '{item['letter']}'s are in the word '{item['word']}'?"},
            {"role": "assistant", "content": str(item['count'])}
        ]


class TaskMixture:
    """Mixture of task datasets with shuffling."""
    
    def __init__(self, datasets: List[TaskDataset]):
        self.datasets = datasets
        
        # Build flat index
        self._indices = []
        for ds_idx, ds in enumerate(datasets):
            for item_idx in range(len(ds)):
                self._indices.append((ds_idx, item_idx))
        
        np.random.shuffle(self._indices)
        
        total = sum(len(ds) for ds in datasets)
        print0(f"TaskMixture: {len(datasets)} datasets, {total} total samples")
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        ds_idx, item_idx = self._indices[idx % len(self._indices)]
        return self.datasets[ds_idx][item_idx]


# =============================================================================
# SFT Data Loader with Padding and Masking
# =============================================================================

class SFTDataLoader:
    """
    Data loader for SFT with proper padding and loss masking.
    
    Unlike mid-training (token packing), SFT uses:
    - Variable length sequences (one conversation per row)
    - Padding to max length in batch
    - Loss mask: -1 for positions to ignore
    """
    
    def __init__(
        self,
        dataset: TaskMixture,
        tokenizer: SFTTokenizer,
        batch_size: int,
        max_seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        device: str = 'cuda'
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        self.cursor = rank
        self.epoch = 0
        self.samples_seen = 0
    
    def reset(self):
        self.cursor = self.rank
        self.epoch = 0
        self.samples_seen = 0
    
    def _collate_batch(self, batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a batch of (tokens, mask) pairs into padded tensors.
        
        Returns:
            inputs: [batch_size, max_len] - input token IDs
            targets: [batch_size, max_len] - target IDs with -1 for masked positions
        """
        # Find max length in this batch (capped at max_seq_len)
        max_len = min(max(len(ids) for ids, _ in batch) - 1, self.max_seq_len)
        
        inputs = torch.full((len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        targets = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 = ignore
        
        for i, (ids, mask) in enumerate(batch):
            # Truncate if needed
            seq_len = min(len(ids) - 1, max_len)
            
            ids_tensor = torch.tensor(ids[:seq_len + 1], dtype=torch.long)
            mask_tensor = torch.tensor(mask[:seq_len + 1], dtype=torch.long)
            
            # inputs are ids[:-1], targets are ids[1:]
            inputs[i, :seq_len] = ids_tensor[:-1]
            
            # Apply mask: only compute loss where mask[1:] == 1
            row_targets = ids_tensor[1:]
            row_mask = mask_tensor[1:]
            row_targets[row_mask == 0] = -1  # Mask out non-assistant tokens
            targets[i, :seq_len] = row_targets
        
        return inputs.to(self.device), targets.to(self.device)
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch."""
        batch = []
        
        while len(batch) < self.batch_size:
            conversation = self.dataset[self.cursor]
            ids, mask = self.tokenizer.render_conversation(conversation)
            
            # Skip if too short (need at least 2 tokens for input/target)
            if len(ids) >= 2:
                batch.append((ids, mask))
            
            self.cursor += self.world_size
            self.samples_seen += 1
            
            if self.cursor >= len(self.dataset):
                self.cursor = self.rank
                self.epoch += 1
        
        return self._collate_batch(batch)
    
    @property
    def progress(self) -> float:
        return self.samples_seen / len(self.dataset)


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

def get_lr_linear_decay(step, max_steps, max_lr, min_lr=0.0):
    """Linear decay from max_lr to min_lr."""
    if max_steps <= 0:
        return max_lr
    progress = step / max_steps
    return max_lr * (1 - progress) + min_lr * progress


def compute_gradient_norm(model):
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
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
    total_tokens = 0
    
    for _ in range(num_batches):
        x, y = val_loader.next_batch()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        
        # Weight by number of non-masked tokens
        num_tokens = (y >= 0).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    
    if ddp:
        # Average across ranks
        stats = torch.tensor([total_loss, total_tokens], device='cuda')
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_loss = stats[0].item() / max(stats[1].item(), 1)
    
    model.train()
    return avg_loss


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint into model."""
    print0(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    
    # Remove DDP/compile prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '').replace('module.', '')
        new_state_dict[new_key] = v
    
    # Handle potential wpe removal (if loading from old checkpoint with wpe)
    if 'transformer.wpe.weight' in new_state_dict:
        print0("  Note: Removing wpe from checkpoint (RoPE-only model)")
        del new_state_dict['transformer.wpe.weight']
    
    model.load_state_dict(new_state_dict, strict=True)
    
    return {
        'step': checkpoint.get('step', 0),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
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
    print0("CHAT SFT - FP8")
    print0("=" * 70)
    
    mode_str = f"fp8_{args.fp8_recipe}" if args.use_fp8 else "bf16"
    
    print0(f"\n{'Configuration':-^50}")
    print0(f"Device: {torch.cuda.get_device_name()} x {world_size}")
    print0(f"Mode: {mode_str}")
    print0(f"Checkpoint: {args.checkpoint}")
    print0(f"Steps: {args.steps}")
    print0(f"Batch size: {args.batch_size} per GPU")
    print0(f"Gradient accumulation: {args.grad_accum}")
    print0(f"Target examples/step: {args.batch_size * args.grad_accum * world_size}")
    print0(f"Max sequence length: {args.max_seq_len}")
    print0(f"Learning rate: {args.lr}")
    
    # Create model
    if args.model_size:
        config = GPTConfig.from_size(args.model_size)
        config.block_size = args.max_seq_len  # Update for longer sequences
    else:
        config = GPTConfig()
        config.block_size = args.max_seq_len
    
    model = GPT(config).to(torch.bfloat16).to(device)
    num_params = model.count_parameters()
    
    print0(f"\n{'Model':-^50}")
    print0(f"Parameters: {num_params/1e9:.2f}B ({num_params/1e6:.1f}M)")
    print0(f"Layers: {config.n_layer}")
    print0(f"Max seq len: {config.block_size}")
    
    # Load checkpoint
    print0(f"\n{'Loading Checkpoint':-^50}")
    ckpt_meta = load_checkpoint(model, args.checkpoint, device)
    print0(f"  Source step: {ckpt_meta['step']}")
    if ckpt_meta['val_loss']:
        print0(f"  Source val_loss: {ckpt_meta['val_loss']:.4f}")
    
    # Apply FP8 (BEFORE DDP)
    if args.use_fp8:
        print0(f"\n{'FP8 Conversion':-^50}")
        model = apply_fp8(model, args.fp8_recipe)
    
    # Wrap in DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    
    # NOTE: No torch.compile for SFT - variable sequence lengths don't work well
    print0("Note: Skipping torch.compile (variable sequence lengths)")
    
    # Tokenizer
    tokenizer = SFTTokenizer()
    print0(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Datasets
    print0(f"\n{'Datasets':-^50}")
    
    train_datasets = []
    
    if args.use_arc:
        train_datasets.append(ARCDataset("ARC-Easy", "train"))
        train_datasets.append(ARCDataset("ARC-Challenge", "train"))
    
    if args.use_gsm8k:
        train_datasets.append(GSM8KDataset("train"))
    
    if args.use_smoltalk:
        train_datasets.append(SmolTalkDataset("train", max_samples=args.smoltalk_samples))
    
    if args.use_spelling:
        train_datasets.append(SimpleSpellingDataset(size=300, split="train"))
        train_datasets.append(SpellingBeeDataset(size=300, split="train"))
    
    if not train_datasets:
        print0("No datasets specified, using defaults")
        train_datasets = [
            SmolTalkDataset("train", max_samples=10000),
            GSM8KDataset("train"),
        ]
    
    train_mixture = TaskMixture(train_datasets)
    
    # Validation: just use SmolTalk test
    val_mixture = TaskMixture([SmolTalkDataset("test", max_samples=5000)])
    
    # Data loaders
    train_loader = SFTDataLoader(
        train_mixture, tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        rank=rank, world_size=world_size, device=device
    )
    val_loader = SFTDataLoader(
        val_mixture, tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        rank=rank, world_size=world_size, device=device
    )
    
    print0(f"Train samples: {len(train_mixture)}")
    print0(f"Val samples: {len(val_mixture)}")
    
    # Auto-calculate steps if epochs specified
    if args.epochs > 0:
        examples_per_step = args.batch_size * args.grad_accum * world_size
        steps_per_epoch = len(train_mixture) // examples_per_step
        args.steps = steps_per_epoch * args.epochs
        print0(f"Epochs: {args.epochs} → {args.steps} steps")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=True
    )
    
    # Logging setup
    log_file = args.log_file or f"chat_sft_{mode_str}_{args.model_size or 'custom'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    checkpoint_path = args.output_checkpoint or f"chat_sft_{mode_str}_{args.model_size or 'custom'}.pt"
    
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
            'num_tokens', 'step_time_ms', 'elapsed_time',
            'lr', 'grad_norm', 'epoch', 'mode'
        ])
    
    # Training loop
    print0(f"\n{'='*70}")
    print0("Starting Chat SFT...")
    print0(f"{'='*70}\n")
    
    model.train()
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for step in range(1, args.steps + 1):
        step_start = time.time()
        
        # Learning rate (linear decay)
        lr = get_lr_linear_decay(step, args.steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation
        accumulated_loss = 0.0
        total_tokens = 0
        
        for micro_step in range(args.grad_accum):
            x, y = train_loader.next_batch()
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum - 1)
            
            with amp_ctx:
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum
            
            scaled_loss.backward()
            accumulated_loss += loss.item()
            total_tokens += (y >= 0).sum().item()
        
        loss_val = accumulated_loss / args.grad_accum
        
        # Average across ranks
        if ddp:
            stats = torch.tensor([loss_val, total_tokens], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            stats[0] /= world_size  # Average loss
            loss_val = stats[0].item()
            total_tokens = int(stats[1].item())
        
        # Gradient norm and clipping
        grad_norm = compute_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        
        # Metrics
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
                    'val_loss': val_loss,
                    'train_loss': loss_val,
                    'config': config,
                    'source_checkpoint': args.checkpoint,
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
                total_tokens, f"{step_time*1000:.1f}", f"{elapsed:.1f}",
                f"{lr:.2e}", f"{grad_norm:.4f}",
                train_loader.epoch, mode_str
            ])
            if step % 50 == 0:
                csv_file.flush()
        
        # Console logging
        if step % args.log_every == 0 or step == 1:
            val_str = f" | val: {val_loss:.4f}" if val_loss else ""
            print0(f"step {step:5d}/{args.steps} | loss: {loss_val:.4f}{val_str} | "
                  f"ppl: {perplexity:.1f} | lr: {lr:.2e} | grad: {grad_norm:.2f} | "
                  f"tokens: {total_tokens} | epoch: {train_loader.epoch} | "
                  f"time: {format_time(elapsed)}")
    
    if csv_file:
        csv_file.close()
    
    # Final validation
    print0(f"\n{'Final Validation':-^50}")
    final_val_loss = evaluate(model, val_loader, num_batches=100, ddp=ddp)
    final_val_ppl = math.exp(final_val_loss)
    print0(f"Final validation loss: {final_val_loss:.4f}")
    print0(f"Final validation perplexity: {final_val_ppl:.2f}")
    
    # Save final checkpoint
    if is_master:
        model_state = model.module.state_dict() if ddp else model.state_dict()
        final_checkpoint = {
            'step': args.steps,
            'model_state_dict': model_state,
            'val_loss': final_val_loss,
            'train_loss': loss_val,
            'config': config,
            'source_checkpoint': args.checkpoint,
            'args': vars(args),
        }
        final_path = checkpoint_path.replace('.pt', '_final.pt')
        torch.save(final_checkpoint, final_path)
        print0(f"Final checkpoint: {final_path}")
    
    # Summary
    total_time = time.time() - start_time
    
    print0(f"\n{'='*70}")
    print0("CHAT SFT COMPLETE")
    print0(f"{'='*70}")
    print0(f"Total steps: {args.steps}")
    print0(f"Training time: {format_time(total_time)}")
    print0(f"Final train loss: {loss_val:.4f}")
    print0(f"Final val loss: {final_val_loss:.4f}")
    print0(f"Best val loss: {best_val_loss:.4f}")
    print0(f"Mode: {mode_str}")
    print0(f"{'='*70}")
    
    cleanup_ddp()
    
    return {
        'train_loss': loss_val,
        'val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chat SFT - FP8')
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to mid-trained checkpoint")
    
    # Training
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--epochs", type=int, default=-1, 
                        help="Epochs (overrides --steps if > 0)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    
    # Learning rate
    parser.add_argument("--lr", type=float, default=1e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Min learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    
    # Model
    parser.add_argument("--model-size", type=str, default="1.5B",
                        help="Model size: 125M, 350M, 760M, 1.5B, 3B, 7B")
    
    # Datasets
    parser.add_argument("--use-arc", action="store_true", default=True)
    parser.add_argument("--use-gsm8k", action="store_true", default=True)
    parser.add_argument("--use-smoltalk", action="store_true", default=True)
    parser.add_argument("--use-spelling", action="store_true", default=True)
    parser.add_argument("--smoltalk-samples", type=int, default=10000,
                        help="Max SmolTalk samples")
    
    # Validation
    parser.add_argument("--val-every", type=int, default=100, help="Validate every N steps")
    parser.add_argument("--val-batches", type=int, default=50, help="Batches per validation")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--output-checkpoint", type=str, default=None)
    
    # FP8
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise",
                        choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"])
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8")
    
    args = parser.parse_args()
    args.use_fp8 = not args.no_fp8
    
    train(args)