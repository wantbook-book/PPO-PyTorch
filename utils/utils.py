import torch
from typing import Union, Optional, Tuple
import torch.nn.functional as F

def compute_entropy(logprobs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute the entropy of the action distribution from logits.
    
    Args:
        logits: Model logits tensor of shape (batch_size, seq_len, vocab_size)
        action_mask: Optional mask for valid action positions
        temperature: Temperature for scaling logits
    
    Returns:
        entropy: Entropy tensor
    """
    # if temperature != 1.0:
    #     logits = logits / temperature
    
    # Compute log probabilities
    # log_probs = F.log_softmax(logits, dim=-1)
    # Compute probabilities
    # probs = F.softmax(logits, dim=-1)
    # probs = log_probs.exp()
    
    # # Compute entropy: -sum(p * log(p))
    # # entropy = -(probs * log_probs).sum(dim=-1)
    # entropy = -(probs * log_probs)
    
    # # Apply action mask if provided
    # if action_mask is not None:
    #     entropy = masked_mean(entropy, action_mask, axis=-1)
    # else:
    #     entropy = entropy.mean(dim=-1)
    
    # return entropy

    # if temperature != 1.0:
    #     logits = logits / temperature
    
    # Compute log probabilities
    # log_probs = F.log_softmax(logits, dim=-1)
    # Compute probabilities
    probs = logprobs.exp()
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * logprobs).sum(dim=-1)
    
    # Apply action mask if provided
    if action_mask is not None:
        entropy = masked_mean(entropy, action_mask, axis=-1)
    else:
        entropy = entropy.mean(dim=-1)
    
    return entropy


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)

def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def masked_whiten(values, mask, shift_mean=True):
    """
    Whiten `values` by normalizing with mean and variance computed over `mask`.

    Args:
        values (torch.Tensor): Input tensor.
        mask (torch.Tensor): Boolean tensor of same shape, selects elements for stats.
        shift_mean (bool): If True (default), output is zero-mean;
                           if False, the original mean is re-added after scaling.

    Returns:
        torch.Tensor: Whitened tensor of same shape as `values`.
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    kl_reward = -kl_coef * kl
    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward

    return reward

def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def tokenize_fn(tokenizer, texts, max_length, padding=True, device=None):
    if not padding:
        # when padding is False, return tokenized texts as list
        return tokenizer(
            texts,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
    batch = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in batch.items()}

def process_sequences(sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
    """
    Process generated sequences to create attention masks and action masks.

    Args:
        sequences (torch.Tensor): Generated sequence tensor
        input_len (int): Length of the input sequence
        eos_token_id (int): Token ID for the end-of-sequence token
        pad_token_id (int): Token ID for the padding token

    Returns:
        tuple: A tuple containing three elements:
            - sequences: Original sequence
            - attention_mask: Attention mask indicating valid token positions
            - action_mask: Action mask indicating valid action token positions
    """
    # Create initial attention mask by marking positions that are neither EOS nor padding tokens
    attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
    seq_length = attention_mask.size(1)

    # Find the position of the last valid token in each sequence
    eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)

    # Handle cases where EOS tokens might appear in the middle of the prompt (for Llama3 and Qwen2 models)
    # Find the position of the first valid token in each sequence
    first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
    # Create position mask
    mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
    # Generate final attention mask, keeping only positions between first and last valid tokens
    attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

    # In reinforcement learning, the state transition is represented as:
    # state_i (current token) + action_i (next token) -> state_i+1 (next token)
    # Generate state sequence from input_len-1 to second-to-last token
    state_seq = sequences[:, input_len - 1 : -1]
    # Generate action mask indicating valid action token positions
    action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
    action_mask[:, 0] = 1

    return sequences, attention_mask, action_mask