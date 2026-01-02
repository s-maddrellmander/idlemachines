#!/usr/bin/env python3
"""
eval_checkpoint.py
==================

Evaluate checkpoints from train_gpt_fp8.py using lm-evaluation-harness.

This provides a thin wrapper that:
1. Loads our checkpoint format directly
2. Controls precision explicitly (BF16)
3. Runs standard benchmarks for comparability
4. Outputs structured JSON results

Usage:
    # Quick sanity check (2 tasks, fast)
    python eval_checkpoint.py -c checkpoints/bf16_125M.pt --suite quick

    # Full base model suite
    python eval_checkpoint.py -c checkpoints/bf16_125M.pt --suite base

    # Specific tasks
    python eval_checkpoint.py -c checkpoints/bf16_125M.pt --tasks hellaswag,arc_easy

    # Limit samples for debugging
    python eval_checkpoint.py -c checkpoints/bf16_125M.pt --suite quick --limit 100

Requirements:
    pip install lm-eval transformers
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

# We'll import lm_eval lazily to give better error messages
try:
    import lm_eval
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False

try:
    from transformers import GPT2TokenizerFast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =============================================================================
# Model Definition (copied from train_gpt_fp8.py for standalone usage)
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
            "13B":   (40,  5120, 40, 8),
        }
        if size in presets:
            n_layer, n_embd, n_head, n_kv_head = presets[size]
            return cls(n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
        raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(config.vocab_size, config.n_embd),
            h=torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.rmsnorm = RMSNorm(config.n_embd)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
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


# =============================================================================
# lm-evaluation-harness Model Wrapper
# =============================================================================

class IdleMachinesLM(LM):
    """
    Wrapper to make our GPT checkpoint compatible with lm-evaluation-harness.
    
    Implements the three core methods:
    - loglikelihood: for multiple choice tasks (HellaSwag, ARC, etc.)
    - loglikelihood_rolling: for perplexity computation
    - generate_until: for generative tasks (stubbed for now)
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        device: str = "auto",
        dtype: str = "bfloat16",
        batch_size: int = 8,
    ):
        super().__init__()
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self._device = device
        self._dtype = getattr(torch, dtype)
        self._batch_size = batch_size
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract config
        if 'config' in ckpt:
            config = ckpt['config']
            # Handle both dataclass and dict formats
            if isinstance(config, dict):
                config = GPTConfig(**config)
        else:
            # Fallback: infer from args
            args = ckpt.get('args', {})
            model_size = args.get('model_size', '125M')
            print(f"  No config found, inferring from model_size: {model_size}")
            config = GPTConfig.from_size(model_size)
        
        # Build model
        self.model = GPT(config)
        
        # Load weights (handle torch.compile _orig_mod. prefix)
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Move to device and dtype
        self.model = self.model.to(self._dtype).to(self._device)
        self.model.eval()
        
        # Store config
        self.config = config
        self._checkpoint_path = checkpoint_path
        
        # Initialize tokenizer (GPT-2, matches vocab_size=50304)
        print("  Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Extract checkpoint metadata
        self._step = ckpt.get('step', 'unknown')
        self._val_loss = ckpt.get('val_loss', None)
        self._train_loss = ckpt.get('train_loss', None)
        self._args = ckpt.get('args', {})
        
        # Print summary
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model: {config.n_layer}L, {config.n_embd}D, {config.n_head}H ({num_params/1e6:.1f}M params)")
        print(f"  Step: {self._step}")
        if self._val_loss:
            print(f"  Val loss: {self._val_loss:.4f} (ppl: {math.exp(self._val_loss):.2f})")
        print(f"  Device: {self._device}, Dtype: {dtype}")
        print(f"  Batch size: {self._batch_size}")
    
    # -------------------------------------------------------------------------
    # Required properties
    # -------------------------------------------------------------------------
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_length(self):
        return self.config.block_size
    
    @property
    def max_gen_toks(self):
        return 256
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def device(self):
        return self._device
    
    # -------------------------------------------------------------------------
    # Tokenization helpers
    # -------------------------------------------------------------------------
    
    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
    
    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass, return logits."""
        with torch.no_grad():
            with torch.amp.autocast(device_type=self._device if self._device != "mps" else "cpu", 
                                   dtype=self._dtype, enabled=(self._device == "cuda")):
                logits, _ = self.model(input_ids)
        return logits
    
    # -------------------------------------------------------------------------
    # Core evaluation methods
    # -------------------------------------------------------------------------
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.
        
        For each (context, continuation) pair:
        - Returns (log_prob, is_greedy) where log_prob is the sum of log probs
          for continuation tokens, and is_greedy indicates if greedy decoding
          would produce the continuation.
        
        This is used for multiple-choice tasks like HellaSwag, ARC, etc.
        """
        results = []
        
        # Process in batches for efficiency
        # Note: for simplicity, we process one at a time here. 
        # Batching requires padding logic which adds complexity.
        for request in tqdm(requests, desc="loglikelihood", leave=False):
            context, continuation = request.args
            
            # Tokenize
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)
            
            # Combine tokens
            full_tokens = ctx_tokens + cont_tokens
            
            # Truncate if needed (keep the end, which has continuation)
            if len(full_tokens) > self.max_length:
                # Calculate how many context tokens to keep
                overflow = len(full_tokens) - self.max_length
                ctx_tokens = ctx_tokens[overflow:]
                full_tokens = ctx_tokens + cont_tokens
            
            # Where continuation starts in the sequence
            cont_start = len(ctx_tokens)
            
            # Forward pass
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=self._device)
            logits = self._model_call(input_ids)  # (1, T, V)
            
            # Compute log probs
            # logits[0, t] predicts token at position t+1
            # So logits[0, cont_start-1] predicts the first continuation token
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            # Sum log probs for continuation tokens
            total_log_prob = 0.0
            greedy_tokens = []
            
            for i, tok in enumerate(cont_tokens):
                # Position in logits that predicts this token
                pos = cont_start - 1 + i
                if pos >= 0 and pos < log_probs.size(0):
                    total_log_prob += log_probs[pos, tok].item()
                    greedy_tokens.append(log_probs[pos].argmax().item())
            
            # Check if greedy decoding matches
            is_greedy = (greedy_tokens == cont_tokens)
            
            results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float]]:
        """
        Compute rolling log-likelihood (unconditional).
        
        Used for perplexity computation on a dataset.
        Returns just the log-likelihood (no is_greedy).
        """
        results = []
        
        for request in tqdm(requests, desc="loglikelihood_rolling", leave=False):
            (text,) = request.args
            tokens = self.tok_encode(text)
            
            # Truncate if needed
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            if len(tokens) <= 1:
                results.append((0.0,))
                continue
            
            # Forward pass
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)
            logits = self._model_call(input_ids)
            
            # Compute log probs
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            # Sum log probs: logits[i] predicts tokens[i+1]
            total = sum(
                log_probs[i, tokens[i + 1]].item() 
                for i in range(len(tokens) - 1)
            )
            
            results.append((total,))
        
        return results
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until stop condition.
        
        Used for generative tasks like HumanEval, GSM8K.
        For base models without instruction tuning, these tasks
        typically perform poorly anyway.
        """
        # Basic implementation - can be enhanced later
        results = []
        
        for request in tqdm(requests, desc="generate_until", leave=False):
            context, gen_kwargs = request.args
            
            # Get stop sequences
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_new = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            
            # Tokenize context
            ctx_tokens = self.tok_encode(context)
            if len(ctx_tokens) >= self.max_length:
                ctx_tokens = ctx_tokens[-(self.max_length - max_new):]
            
            generated = list(ctx_tokens)
            
            # Simple greedy generation
            for _ in range(max_new):
                if len(generated) >= self.max_length:
                    break
                    
                input_ids = torch.tensor([generated], dtype=torch.long, device=self._device)
                logits = self._model_call(input_ids)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)
                
                # Check stop sequences
                current_text = self.tok_decode(generated[len(ctx_tokens):])
                if any(stop in current_text for stop in until):
                    break
                
                # Check EOS
                if next_token == self.eot_token_id:
                    break
            
            # Decode just the generated part
            output = self.tok_decode(generated[len(ctx_tokens):])
            
            # Trim at stop sequence
            for stop in until:
                if stop in output:
                    output = output[:output.index(stop)]
            
            results.append(output)
        
        return results


# =============================================================================
# Benchmark Suites
# =============================================================================

BENCHMARK_SUITES = {
    # Quick sanity check (2 tasks, ~5 min on CPU)
    "quick": [
        "hellaswag",
        "arc_easy",
    ],
    
    # Base model benchmarks (no instruction following needed)
    # These work well for pretrained models
    "base": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
        "lambada_openai",
        "boolq",
    ],
    
    # Extended suite including MMLU
    "full": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
        "lambada_openai",
        "boolq",
        "mmlu",
    ],
    
    # Just perplexity-style eval
    "perplexity": [
        "lambada_openai",
    ],
}


# =============================================================================
# Main Evaluation Function
# =============================================================================

def run_evaluation(
    checkpoint_path: str,
    tasks: List[str],
    output_dir: str = "eval_results",
    batch_size: int = 8,
    device: str = "auto",
    dtype: str = "bfloat16",
    limit: Optional[int] = None,
    num_fewshot: int = 0,
) -> dict:
    """
    Run evaluation and save results.
    
    Args:
        checkpoint_path: Path to checkpoint file
        tasks: List of task names to evaluate
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to use (auto, cuda, cpu, mps)
        dtype: Data type (bfloat16, float32)
        limit: Limit number of samples per task (for debugging)
        num_fewshot: Number of few-shot examples (0 for zero-shot)
    
    Returns:
        Dictionary with evaluation results
    """
    # Create model
    model = IdleMachinesLM(
        checkpoint_path,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Running evaluation on: {', '.join(tasks)}")
    if limit:
        print(f"  (limited to {limit} samples per task)")
    print(f"{'='*60}\n")
    
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )
    
    # Extract and organize scores
    scores = {}
    for task_name, task_results in results.get('results', {}).items():
        # Different tasks use different metric names
        if 'acc_norm,none' in task_results:
            scores[task_name] = task_results['acc_norm,none']
        elif 'acc,none' in task_results:
            scores[task_name] = task_results['acc,none']
        elif 'acc_norm' in task_results:
            scores[task_name] = task_results['acc_norm']
        elif 'acc' in task_results:
            scores[task_name] = task_results['acc']
        elif 'perplexity,none' in task_results:
            scores[task_name] = task_results['perplexity,none']
        elif 'perplexity' in task_results:
            scores[task_name] = task_results['perplexity']
        elif 'word_perplexity,none' in task_results:
            scores[task_name] = task_results['word_perplexity,none']
    
    # Compute mean accuracy (excluding perplexity metrics)
    acc_scores = [v for k, v in scores.items() 
                  if 'perplexity' not in k.lower() and isinstance(v, (int, float))]
    mean_acc = sum(acc_scores) / len(acc_scores) if acc_scores else None
    
    # Build output
    output = {
        "checkpoint": checkpoint_path,
        "checkpoint_name": Path(checkpoint_path).stem,
        "step": model._step,
        "val_loss": model._val_loss,
        "train_loss": model._train_loss,
        "model_config": {
            "n_layer": model.config.n_layer,
            "n_embd": model.config.n_embd,
            "n_head": model.config.n_head,
            "n_kv_head": model.config.n_kv_head,
        },
        "training_args": model._args,
        "eval_config": {
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "num_fewshot": num_fewshot,
            "limit": limit,
        },
        "timestamp": datetime.now().isoformat(),
        "tasks": tasks,
        "scores": scores,
        "mean_accuracy": mean_acc,
        "full_results": results.get('results', {}),
    }
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(checkpoint_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{ckpt_name}_eval_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {model._step}")
    if model._val_loss:
        print(f"Val loss: {model._val_loss:.4f}")
    
    print(f"\nScores:")
    for task, score in sorted(scores.items()):
        if 'perplexity' in task.lower():
            print(f"  {task}: {score:.2f}")
        else:
            print(f"  {task}: {score:.4f} ({score*100:.2f}%)")
    
    if mean_acc is not None:
        print(f"\nMean accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")
    
    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate idlemachines checkpoints using lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (HellaSwag + ARC-Easy)
  python eval_checkpoint.py -c checkpoint.pt --suite quick

  # Full base model evaluation
  python eval_checkpoint.py -c checkpoint.pt --suite base

  # Specific tasks
  python eval_checkpoint.py -c checkpoint.pt --tasks hellaswag,piqa

  # Debug with limited samples
  python eval_checkpoint.py -c checkpoint.pt --suite quick --limit 100
        """
    )
    
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("-t", "--tasks", type=str, default=None,
                        help="Comma-separated list of tasks")
    parser.add_argument("-s", "--suite", type=str, default="quick",
                        choices=list(BENCHMARK_SUITES.keys()),
                        help="Benchmark suite to run (default: quick)")
    parser.add_argument("-o", "--output-dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("-b", "--batch-size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("-d", "--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="Data type for model")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (for debugging)")
    parser.add_argument("--num-fewshot", type=int, default=5,
                        help="Number of few-shot examples")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not LM_EVAL_AVAILABLE:
        print("ERROR: lm-eval not installed. Install with:")
        print("  pip install lm-eval")
        sys.exit(1)
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers not installed. Install with:")
        print("  pip install transformers")
        sys.exit(1)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Get task list
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = BENCHMARK_SUITES[args.suite]
    
    # Run evaluation
    run_evaluation(
        checkpoint_path=args.checkpoint,
        tasks=tasks,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )


if __name__ == "__main__":
    main()
