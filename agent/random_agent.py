import sys
import os
import random
import time
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.freecell_env import FreecellEnv

def select_action(env):
    """
    Wybiera losowo indeks akcji z przestrzeni 0..max_actions-1,
    spośród tych, których odpowiadająca akcja jest legalna.
    Zwraca indeks (int) lub None, jeśli brak akcji.
    """
    legal_actions = set(env.get_legal_actions())
    candidate_indices = [i for i, action_str in enumerate(env.all_actions) if action_str in legal_actions]

    if not candidate_indices:
        return None

    return random.choice(candidate_indices)

def run_random_agent(episodes=1, render=False):
    env = FreecellEnv()

    MAX_STEPS = 50
    NO_PROGRESS_LIMIT = 30

    for episode in range(episodes):
        print(f"\n===== Episode {episode + 1} =====")
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        no_progress_counter = 0
        last_state_snapshot = str(env.game.get_state())

        reward_components = defaultdict(float)
        reward_counts = defaultdict(int)

        while not done:

            legal = env.get_legal_actions()
            if not legal:
                print("No legal moves.")
                break

            action_idx = select_action(env)
            if action_idx is None:
                print("No legal moves available — terminating episode.")
                break

            state, reward, done, breakdown = env.step(action_idx)
            total_reward += reward
            step_count += 1

            for k, v in breakdown.items():
                reward_components[k] += v
                reward_counts[k] += 1

            if render:
                print(f"\nStep {step_count}:")
                print(f"Action index: {action_idx}, Action: {env.all_actions[action_idx]}, Reward: {reward:.2f}")
                for k, v in breakdown.items():
                    print(f"  > {k}: {v:+.2f}")
                env.render()

            current_snapshot = str(env.game.get_state())
            if current_snapshot == last_state_snapshot:
                no_progress_counter += 1
            else:
                no_progress_counter = 0

            last_state_snapshot = current_snapshot

            if step_count >= MAX_STEPS:
                print("Reached step limit — terminating episode.")
                done = True
            elif no_progress_counter >= NO_PROGRESS_LIMIT:
                print("No progress for too long — terminating episode.")
                done = True

        print(f"\nEpisode finished after {step_count} steps, total reward: {total_reward:.2f}")
        print("\nReward Breakdown:")
        for k in sorted(reward_components):
            total = reward_components[k]
            count = reward_counts[k]
            print(f"{k:30s} → {total:+.2f}   ({count}x)")

if __name__ == "__main__":
    run_random_agent(episodes=1, render=True)
