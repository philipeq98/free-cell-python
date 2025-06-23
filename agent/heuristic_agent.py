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
    return heuristic_policy(legal)


def run_heuristic_agent(episodes=10):
    env = FreecellEnv()

    for ep in range(episodes):
        print(f"\n===== Episode {ep+1} =====")
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            legal = env.get_legal_actions()
            if not legal:
                print("No legal moves.")
                break

            action = select_action(env)
            obs, reward, done, breakdown = env.step(action)

            total_reward += reward
            step_count += 1

            print(f"Step {step_count} | Action: {action} | Reward: {reward:.2f}")
            for k, v in breakdown.items():
                print(f"  > {k}: {v:+.2f}")

        print(f"Episode finished after {step_count} steps. Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    run_heuristic_agent()
