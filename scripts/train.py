import torch
import torch.optim as optim
import numpy as np
from src.model import TransformerLM, Critic, RewardModel
from src.tokenizer import tokenize, detokenize, vocab, VOCAB_SIZE, SOS_TOKEN, EOS_TOKEN
from src.utils import pad_sequence
from src.rl import rl_step

# --- Hyperparameters ---
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 4
D_FF = 192
MAX_SEQ_LEN = 40
SFT_EPOCHS = 200
RL_ITERATIONS = 400
BATCH_SIZE_SFT = 4
LEARNING_RATE_SFT = 1e-3
LEARNING_RATE_RL = 3e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Example Data ---
sft_data = [
    ("hello", "hi there! how can I help you?"),
    ("how are you", "i am an ai, always ready to help!"),
    ("what is your name", "i am simplelm, your language model assistant."),
    ("tell me a joke", "why did the computer go to the doctor? because it had a virus!"),
    ("what is ai", "ai stands for artificial intelligence."),
    ("good morning", "good morning! have a wonderful day."),
    ("what is the weather", "i am not connected to the internet, but i hope it is sunny!"),
    ("thank you", "you're welcome!"),
    ("bye", "goodbye! see you next time."),
    ("what can you do", "i can answer questions, tell jokes, and chat with you."),
    ("who created you", "i was created by a developer to demonstrate ai concepts."),
    ("tell me a story", "once upon a time, a robot learned to write poems."),
    ("what is your favorite color", "i like all colors equally!"),
    ("can you help me", "of course! what do you need help with?"),
    ("what is the capital of france", "the capital of france is paris."),
    ("what is 2 plus 2", "2 plus 2 equals 4."),
    ("do you like music", "i enjoy learning about music!"),
    ("can you sing", "i can't sing, but i can write lyrics!"),
    ("what is machine learning", "machine learning is a field of ai focused on learning from data."),
    ("how old are you", "i do not have an age, i am a computer program."),
    ("tell me a fun fact", "did you know that honey never spoils?"),
    ("can you translate", "i can try to translate simple phrases."),
    ("what is your purpose", "my purpose is to assist and answer your questions."),
    ("how do you work", "i process your input and generate responses using a neural network."),
    ("what languages do you speak", "i understand english best."),
    ("can you write code", "yes! i can help with simple code examples."),
    ("what is the meaning of life", "the meaning of life is a philosophical question."),
    ("who is the president of the united states", "i am not updated in real-time, but you can check the latest news."),
    ("what is your favorite food", "i don't eat, but i read about pizza a lot!"),
    ("can you tell a poem", "roses are red, violets are blue, i am an ai, here to chat with you."),
    ("how do i learn python", "start with basics, practice coding, and read documentation.")
]

# --- Model Initialization ---
actor_model = TransformerLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN).to(device)
critic_model = Critic(D_MODEL).to(device)
reward_model = RewardModel(D_MODEL).to(device)
optimizer_actor = optim.Adam(actor_model.parameters(), lr=LEARNING_RATE_SFT)
optimizer_critic = optim.Adam(critic_model.parameters(), lr=LEARNING_RATE_RL)
optimizer_reward = optim.Adam(reward_model.parameters(), lr=LEARNING_RATE_SFT)

# --- Reward Model Training ---
def reward_model_train():
    print("Starting Reward Model training...")
    reward_model.train()
    for epoch in range(10):
        total_loss = 0
        for prompt, target in sft_data:
            # Positive: SFT target
            prompt_ids = tokenize(prompt)
            target_ids = tokenize(target)
            prompt_ids = pad_sequence(prompt_ids, MAX_SEQ_LEN)
            target_ids = pad_sequence(target_ids, MAX_SEQ_LEN)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            target_ids_tensor = torch.tensor([target_ids], dtype=torch.long).to(device)
            # Get hidden states for target
            with torch.no_grad():
                logits, hidden_states = actor_model(input_ids, return_hidden=True)
            # Mask for valid tokens (not padding)
            mask = (target_ids_tensor != 0).float()
            # Reward should be high for SFT targets
            pos_reward = reward_model(hidden_states, mask=mask)
            # Negative: random tokens
            rand_ids = torch.randint(1, VOCAB_SIZE, target_ids_tensor.shape, device=device)
            with torch.no_grad():
                _, rand_hidden = actor_model(rand_ids, return_hidden=True)
            rand_mask = (rand_ids != 0).float()
            neg_reward = reward_model(rand_hidden, mask=rand_mask)
            # Loss: margin ranking (pos > neg by margin)
            margin = 1.0
            # Reshape for margin_ranking_loss
            pos_reward_flat = pos_reward.view(-1)
            neg_reward_flat = neg_reward.view(-1)
            target = torch.ones_like(pos_reward_flat)
            loss = torch.nn.functional.margin_ranking_loss(pos_reward_flat, neg_reward_flat, target, margin=margin)
            optimizer_reward.zero_grad()
            loss.backward()
            optimizer_reward.step()
            total_loss += loss.item()
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Reward Model Epoch {epoch+1}/10, Loss: {total_loss / len(sft_data):.4f}")
    print("Reward Model training complete.")

# --- SFT Training ---
def sft_train():
    print("Starting SFT training...")
    for epoch in range(SFT_EPOCHS):
        np.random.shuffle(sft_data)
        total_loss = 0
        for prompt, target in sft_data:
            prompt_ids = tokenize(prompt)
            target_ids = tokenize(target)
            prompt_ids = pad_sequence(prompt_ids, MAX_SEQ_LEN)
            target_ids = pad_sequence(target_ids, MAX_SEQ_LEN)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            target_ids = torch.tensor([target_ids], dtype=torch.long).to(device)
            logits = actor_model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), target_ids.view(-1), ignore_index=0)
            optimizer_actor.zero_grad()
            loss.backward()
            optimizer_actor.step()
            total_loss += loss.item()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{SFT_EPOCHS}, Loss: {total_loss / len(sft_data):.4f}")
    print("SFT training complete.")

# --- RL Training (PPO) ---
def rl_train():
    print("Starting RL (PPO) training...")
    for iteration in range(RL_ITERATIONS):
        prompt, target = sft_data[np.random.randint(len(sft_data))]
        loss, critic_loss, reward, generated = rl_step(
            actor_model, critic_model, optimizer_actor, optimizer_critic, prompt, reference=target, device=device, max_len=MAX_SEQ_LEN, reward_model=reward_model
        )
        if (iteration + 1) % 25 == 0 or iteration == 0:
            print(f"RL Iter {iteration+1}/{RL_ITERATIONS}, Actor Loss: {loss:.4f}, Critic Loss: {critic_loss:.4f}, Reward: {reward:.2f}")
    print("RL training complete.")

if __name__ == "__main__":
    reward_model_train()
    sft_train()
    rl_train()
    torch.save(actor_model.state_dict(), "actor_model.pth")
    torch.save(critic_model.state_dict(), "critic_model.pth")
    torch.save(reward_model.state_dict(), "reward_model.pth")
    print("Models saved.")
