import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.freecell_env import FreecellEnv

def heuristic_policy(legal_actions):
    # 1. Foundation moves – zawsze na początku
    for act in legal_actions:
        if '2f' in act:
            return act

    # 2. Ruchy w kolumnach (p2p) – może porządkuje coś
    p2p = [a for a in legal_actions if a.startswith("p2p")]
    if p2p:
        return random.choice(p2p)

    # 3. Unikanie zapełniania freecelli – raczej unikaj p2c
    non_p2c = [a for a in legal_actions if not a.startswith("p2c")]
    if non_p2c:
        return random.choice(non_p2c)

    # 4. Ostateczność: jak nic lepszego nie ma
    return random.choice(legal_actions)

def select_action(env):
    legal = env.get_legal_actions()
    if not legal:
        return None

    chosen_action_str = heuristic_policy(legal)

    try:
        action_idx = env.all_actions.index(chosen_action_str)
    except ValueError:
        return None

    return action_idx

def run_heuristic_agent(episodes=10, render=False):
    env = FreecellEnv()

    MAX_STEPS = 50
    NO_PROGRESS_LIMIT = 30

    for ep in range(episodes):
        print(f"\n===== Episode {ep+1} =====")
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        no_progress_counter = 0
        last_state_snapshot = str(env.game.get_state())

        reward_components = {}
        reward_counts = {}

        while not done:
            legal = env.get_legal_actions()
            if not legal:
                print("No legal moves.")
                break

            action_idx = select_action(env)
            if action_idx is None:
                print("No legal moves available — terminating episode.")
                break

            obs, reward, done, breakdown = env.step(action_idx)
            total_reward += reward
            step_count += 1

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

        print(f"Episode finished after {step_count} steps. Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    run_heuristic_agent(episodes=1, render=True)
