# 🃏 FreecellRL – Reinforcement Learning Environment for Freecell (with Action Masking)

## Overview

This project extends the original **Freecell** text-based game implemented in Python by [yintellect](https://github.com/yintellect). While preserving the object-oriented design of the card engine and game logic, this fork **repurposes the game into a reinforcement learning (RL) environment**, following the OpenAI Gym interface.

The goal is to provide a sandbox for experimenting with **masked reinforcement learning agents** (e.g. PPO with invalid action masking) on a complex, partially structured task like Freecell.

---

## 🚀 Features

- ♠️ Object-oriented card and deck system (fully reusable across games)
- ♦️ Modular Freecell logic with cell, pile, and foundation state tracking
- ♣️ OpenAI Gym-style environment: `reset()`, `step()`, and `.render()` implemented
- ❤️ **Action masking support** (valid action indices only)
- 🧠 Compatible with **Stable-Baselines3** (SB3) algorithms like `PPO`
- 🎲 Random agent with masking included
- 📈 PPO agent with custom masked policy and state extractor
- 🔬 Useful for RL benchmarking on long-horizon, logic-based games

---

## 🧱 Project Structure

```
.
├── environment/
│   ├── freecell_env.py            # OpenAI Gym-compatible environment
│   ├── card.py                    # Card class (face, suit, color, etc.)
│   ├── deck.py                    # Deck creation and manipulation
│   ├── game.py                    # Freecell logic (moves, rules, rendering)
│   ├── masked_policy.py           # Custom MaskedActorCriticPolicy for PPO
│   └── utils.py                   # Encoding states, masking logic
├── agents/
│   ├── random_agent.py            # Random agent using valid action mask
│   ├── train_ppo.py               # PPO training loop
│   └── evaluate.py                # Testing trained agents
├── assets/
│   └── sample_board.txt           # Text render example
├── README.md
```

---

## 🎯 Example: Training PPO Agent with Action Masking

```bash
python agents/train_ppo.py
```

This will:

1. Initialize `FreecellEnv`, which returns observations as dicts:
   ```python
   {
     "obs": <game_state_vector>,
     "action_mask": <binary mask of valid actions>
   }
   ```
2. Use `MaskedActorCriticPolicy` to dynamically mask logits of invalid actions.
3. Train a PPO agent with custom feature extractor `CustomDictFeaturesExtractor`.

---

## 🧪 Example: Run Random Agent with Masking

```bash
python agents/random_agent.py
```

This script simulates random gameplay, but only chooses from legal actions using the `action_mask`.

Sample output:

```
Step 1: action=52, reward=+0.20
  > moved_to_foundation: +0.50
  > moved_with_correct_color: +0.10
  > invalid_action: -0.40
...
Episode finished after 23 steps, total reward: 1.60
```

---

## 🛠️ Notes on Masked RL

Action masking is implemented by modifying the logits of the policy network to `-inf` where invalid actions exist. In this project:

- The `MaskedActorCriticPolicy` modifies the logits using:
  ```python
  logits += (mask + 1e-8).log()
  ```
- This allows the agent to **never select invalid moves**, improving sample efficiency and convergence.

---

## 🧹 Integration with Stable-Baselines3

To use SB3 with this environment:

1. Wrap `FreecellEnv` in `DummyVecEnv`:

   ```python
   from stable_baselines3.common.vec_env import DummyVecEnv
   env = DummyVecEnv([lambda: FreecellEnv()])
   ```

2. Use custom policy:

   ```python
   model = PPO(MaskedActorCriticPolicy, env, policy_kwargs={...})
   ```

3. Observe training metrics (reward, episode length, KL divergence, etc.)

---

## 📇 Roadmap

-

---

## 📜 Original Project (Legacy)

The original version focused on simulating a **console-based Freecell game** in Python with:

- Reusable `Card` and `Deck` classes
- Fully working game logic with moves like `p2f`, `c2p`, etc.
- Command-line play loop
- Rule-checking and win condition logic

Much of that core logic is retained in this project but abstracted into a reinforcement learning framework.

---

## 📋 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym Environment API](https://www.gymlibrary.dev/)
- [Original Project by yintellect](https://github.com/yintellect/freecell-game-python)

---

## 👤 Attribution

This project is adapted from the original work of [@yintellect](https://github.com/yintellect) and restructured by **ノヴォチンスキーフィリップ** for RL research and experimentation.

---

## 🧑‍💻 License

MIT License. Use it, fork it, break it, improve it — just don't forget to mask your logits 😄.

