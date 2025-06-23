import sys
import os
from collections import defaultdict, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.freecell_env import FreecellEnv

# Dynamiczny wybór agenta
def get_agent_by_name(name):
    if name == "random":
        from agent import random_agent
        return random_agent.select_action
    elif name == "heuristic":
        from agent import heuristic_agent
        return heuristic_agent.select_action
    else:
        raise ValueError(f"Unknown agent: {name}")

def run_benchmark(select_action_fn, episodes=1000):
    env = FreecellEnv()
    MAX_STEPS = 50
    NO_PROGRESS_LIMIT = 30

    episode_stats = []

    for ep in range(episodes):
        if ep % (episodes // 10) == 0:
            print(f"[{ep}/{episodes}] Running episode...")

        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        no_progress_counter = 0
        last_snapshot = str(env.game.get_state())

        reward_components = defaultdict(float)
        reward_counts = defaultdict(int)
        end_reason = "unknown"

        while not done:
            # Agent zwraca indeks akcji (int) lub None
            action_idx = select_action_fn(env)
            if action_idx is None:
                end_reason = "no_moves"
                break

            state, reward, done, breakdown = env.step(action_idx)
            total_reward += reward
            step_count += 1

            for k, v in breakdown.items():
                reward_components[k] += v
                reward_counts[k] += 1

            snapshot = str(env.game.get_state())
            if snapshot == last_snapshot:
                no_progress_counter += 1
            else:
                no_progress_counter = 0
            last_snapshot = snapshot

            if "loop_detected" in breakdown:
                end_reason = "loop_detected"
                done = True
            elif step_count >= MAX_STEPS:
                end_reason = "max_steps"
                done = True
            elif no_progress_counter >= NO_PROGRESS_LIMIT:
                end_reason = "no_progress"
                done = True
            elif all(len(stack) == 13 for stack in env.get_state()["foundation"]):
                end_reason = "game_won"
                done = True

        episode_stats.append({
            "reward": total_reward,
            "steps": step_count,
            "end_reason": end_reason,
            "rewards": dict(reward_components),
            "reward_counts": dict(reward_counts),
        })

    return episode_stats


def summarize_stats(stats):
    total_rewards = [ep["reward"] for ep in stats]
    total_steps = [ep["steps"] for ep in stats]
    reasons = Counter(ep["end_reason"] for ep in stats)

    print("\n--- BENCHMARK SUMMARY ---")
    print(f"Episodes:         {len(stats)}")
    print(f"Avg Reward:       {sum(total_rewards) / len(stats):.2f}")
    print(f"Avg Steps:        {sum(total_steps) / len(stats):.2f}")
    print("End Reasons:")
    for reason, count in reasons.items():
        print(f"  {reason:15s} → {count} ({count / len(stats) * 100:.1f}%)")

    reward_totals = defaultdict(float)
    reward_counts = defaultdict(int)
    for ep in stats:
        for k, v in ep["rewards"].items():
            reward_totals[k] += v
        for k, c in ep["reward_counts"].items():
            reward_counts[k] += c

    print("\nAverage Reward Breakdown per Episode:")
    for k in sorted(reward_totals):
        avg = reward_totals[k] / len(stats)
        count = reward_counts[k]
        print(f"{k:30s} → {avg:+.2f} avg, {count} total occurrences")


if __name__ == "__main__":
    print("Which agent do you want to use? [random / heuristic]")
    agent_name = input("> ").strip().lower()

    if agent_name not in {"random", "heuristic"}:
        print("❌ Invalid agent name. Use 'random' or 'heuristic'.")
        sys.exit(1)

    print("How many episodes?")
    try:
        episodes = int(input("> "))
    except ValueError:
        print("❌ Invalid number. Using default: 1000")
        episodes = 1000

    print(f"\n▶️ Running benchmark with agent: {agent_name}, episodes: {episodes}")
    select_action_fn = get_agent_by_name(agent_name)
    results = run_benchmark(select_action_fn, episodes=episodes)
    summarize_stats(results)
