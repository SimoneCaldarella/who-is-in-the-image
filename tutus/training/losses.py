import torch
import torch.nn.functional as F

def dpo(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=0.1,
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits)

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean().detach(), rejected_rewards.mean().detach()

def sft(logprobs):
    return -logprobs.mean()

def sft_with_kl(logprobs, ref_logprobs, lambda_kl=0.7):
    """Compute the SFT loss with KL divergence.

    Args:
        logprobs: Log probabilities of the model for the chosen responses. Shape: (batch_size,)
        ref_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)

    Returns:
        A tuple of two tensors: (loss, kl_divergence).
    """
    # KL divergence
    kl_div = (logprobs - ref_logprobs).mean()
    # Combine
    loss = (lambda_kl * sft(logprobs)) + ((1 - lambda_kl) * kl_div)

    return loss, kl_div.detach()

def unlearn(logprobs):
    """Compute the unlearning loss.

    Args:
        logprobs: Log probabilities of the model for the chosen responses. Shape: (batch_size,)

    Returns:
        A tensor representing the unlearning loss.
    """
    # Unlearning loss
    return logprobs.mean()

def npo(logprobs, ref_logprobs, beta=0.1):
    """Compute the unlearning loss with KL divergence.

    Args:
        logprobs: Log probabilities of the model for the rejected responses. Shape: (batch_size,)
        ref_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of two tensors: (loss, kl_divergence).
    """
    # KL divergence
    kl_div = (logprobs - ref_logprobs).mean()
    logits = -kl_div
    # Combine
    loss = 2/beta * -F.logsigmoid(beta * logits)

    return loss


def dpo_special(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      reference_support_logprobs,
      model_support_logprobs,
      beta=0.1,
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs + model_support_logprobs
    reference_logratios = reference_chosen_logprobs + reference_support_logprobs

    logits = model_logratios + reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits) - model_rejected_logprobs

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean().detach(), rejected_rewards.mean().detach()