# SimpleLM: Lightweight Transformer Language Model with RLHF (PPO)

## Overview

**SimpleLM** is a minimal, CPU-friendly language model based on a Transformer decoder architecture. It supports both supervised fine-tuning (SFT) with prompt/response pairs and PPO-based RL training with a neural reward model (RLHF-style) for human/AI feedback (RLHF) using PPO. The project is designed for educational purposes and easy experimentation on small hardware.

## Features
- Pure PyTorch implementation
- Lightweight: Trains on CPU in minutes
- Modular codebase: Model, tokenizer, RL logic separated
- Includes both SFT and RLHF (PPO) training scripts
- Inference/demo script included

## Project Structure

```
SimpleLM/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py           # TransformerLM, Critic, etc.
│   ├── tokenizer.py       # Tokenizer, vocab
│   ├── rl.py              # PPO/RLHF logic
│   └── utils.py           # Padding, helpers
├── scripts/
│   ├── train.py           # Training script (SFT + RL)
│   └── infer_demo.py      # Inference/demo script
```

## Installation

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training (SFT + RLHF)

```bash
python scripts/train.py
```
- This will run supervised fine-tuning (SFT) followed by RLHF (PPO) on toy data.
- Model checkpoints (`actor_model.pth`, `critic_model.pth`) will be saved after training.

### 2. Inference / Demo

```bash
python scripts/infer_demo.py
```
- Loads the trained model and generates text for sample prompts.

## Model Architecture
- Transformer decoder (no encoder)
- Multi-head self-attention, positionwise feedforward, positional encoding
- Critic network for PPO

## RLHF (PPO) Details
- Uses Proximal Policy Optimization (PPO) for RLHF
- Reward function is simple and can be replaced with human/AI feedback
- See `src/rl.py` for details

## Customization
- Modify vocab, model size, or training data in `src/tokenizer.py` and `scripts/train.py`
- Change RL reward logic in `src/rl.py`

## Requirements
- Python 3.7+
- PyTorch
- numpy

## License
MIT
