import torch
import torch.nn.functional as F
import numpy as np
from src.tokenizer import tokenize, detokenize, SOS_TOKEN, EOS_TOKEN
from src.utils import pad_sequence

# PPO/RLHF logic

def compute_ppo_loss(logprobs, old_logprobs, advantages, clip_epsilon=0.1, entropy_coef=0.01):
    ratios = torch.exp(logprobs - old_logprobs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy = -(torch.exp(logprobs) * logprobs).sum(-1).mean()
    return policy_loss - entropy_coef * entropy

# Example reward function (can be replaced by human/AI feedback)
def reward_function(prompt, generated, reference=None):
    # Simple reward: +1 if any reference word is in generated, else 0
    if reference:
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        return float(len(ref_words & gen_words) > 0)
    # Or use length as a dummy reward
    return float(len(generated.split()) > 2)

# RL training step (simplified, batch = 1 for demo)
def rl_step(actor_model, critic_model, optimizer_actor, optimizer_critic, prompt, reference, device, max_len=32, reward_model=None):
    actor_model.train()
    critic_model.train()
    # Generate response
    prompt_ids = tokenize(prompt)
    input_ids = [SOS_TOKEN] + prompt_ids
    input_ids = pad_sequence(input_ids, max_len)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, hidden_states = actor_model(input_tensor, return_hidden=True)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    generated_ids = [input_tensor[0, 0].item()]
    for i in range(1, max_len):
        next_token = torch.multinomial(probs[0, i-1], num_samples=1).item()
        generated_ids.append(next_token)
        if next_token == EOS_TOKEN:
            break
    generated_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
    # Compute reward
    if reward_model is not None:
        # Mask for valid tokens (not padding)
        mask = (generated_tensor != 0).float()
        with torch.no_grad():
            _, gen_hidden = actor_model(generated_tensor, return_hidden=True)
            reward = reward_model(gen_hidden, mask=mask).item()
    else:
        reward = float(any(g == t for g, t in zip(generated_ids, tokenize(reference))))
    # Critic value: use hidden states, not logits
    values = critic_model(hidden_states[:, :-1, :].detach())
    advantages = torch.tensor([reward], dtype=torch.float).to(device) - values.mean()
    # PPO loss
    dist = torch.distributions.Categorical(logits=logits[:, :-1, :])
    sampled_ids = dist.sample()
    logprobs = dist.log_prob(sampled_ids)
    loss = compute_ppo_loss(logprobs.mean(), logprobs.mean().detach(), advantages, clip_epsilon=0.1)
    optimizer_actor.zero_grad()
    loss.backward()
    optimizer_actor.step()
    # Critic update: recompute hidden states so backward uses a fresh graph
    with torch.no_grad():
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        _, hidden_states_new = actor_model(input_ids_tensor, return_hidden=True)
    values_new = critic_model(hidden_states_new[:, :-1, :])
    # Ensure both arguments are scalars to avoid shape warning
    critic_loss = F.mse_loss(values_new.mean().view([]), torch.tensor(reward, dtype=torch.float, device=values_new.device).view([]).detach())
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()
    return loss.item(), critic_loss.item(), reward, detokenize(generated_ids)
