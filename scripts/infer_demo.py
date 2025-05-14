import torch
from src.model import TransformerLM, RewardModel
from src.tokenizer import tokenize, detokenize, VOCAB_SIZE, SOS_TOKEN, EOS_TOKEN

D_MODEL = 96
N_HEADS = 4
N_LAYERS = 4
D_FF = 192
MAX_SEQ_LEN = 40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actor_model = TransformerLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN).to(device)
actor_model.load_state_dict(torch.load("actor_model.pth", map_location=device))
actor_model.eval()

# Optionally load reward model for inspection
try:
    reward_model = RewardModel(D_MODEL).to(device)
    reward_model.load_state_dict(torch.load("reward_model.pth", map_location=device))
    reward_model.eval()
except Exception as e:
    reward_model = None

def generate_text(model, prompt_text, max_len=MAX_SEQ_LEN, top_k=0, top_p=0.9):
    model.eval()
    prompt_ids_inf = [SOS_TOKEN] + tokenize(prompt_text, add_sos_eos=False)
    if len(prompt_ids_inf) >= max_len:
        return detokenize(prompt_ids_inf[:max_len])
    current_input_ids = torch.tensor([prompt_ids_inf], dtype=torch.long).to(device)
    generated_ids = list(prompt_ids_inf)
    with torch.no_grad():
        for _ in range(max_len - len(prompt_ids_inf)):
            if current_input_ids.size(1) >= MAX_SEQ_LEN: break
            logits = model(current_input_ids)[:, -1, :]
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_idx_in_topk = torch.multinomial(probs, num_samples=1)
                next_token_id = top_k_indices.gather(-1, next_token_idx_in_topk).item()
            elif top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove_mask = cumulative_probs > top_p
                sorted_indices_to_remove_mask[..., 1:] = sorted_indices_to_remove_mask[..., :-1].clone()
                sorted_indices_to_remove_mask[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove_mask]
                temp_logits = logits.clone()
                temp_logits[0, indices_to_remove] = -float('Inf')
                probs = torch.nn.functional.softmax(temp_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_token_id)
            if len(generated_ids) >= MAX_SEQ_LEN and next_token_id != EOS_TOKEN:
                current_input_ids = torch.cat([current_input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)[:,:MAX_SEQ_LEN]
            else:
                current_input_ids = torch.cat([current_input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            if next_token_id == EOS_TOKEN or len(generated_ids) >= max_len:
                break
    return detokenize(generated_ids)

if __name__ == "__main__":
    test_prompts = [
        "hello",
        "tell me about ai",
        "how are you today",
        "what is learning"
    ]
    for p in test_prompts:
        generated_output = generate_text(actor_model, p, max_len=MAX_SEQ_LEN, top_k=0, top_p=0.9)
        print(f"Prompt: {p}")
        print(f"Generated: {generated_output}")
        print("-" * 20)
