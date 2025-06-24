import sys
import os
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.freecell_env import FreecellEnv
from environment.masked_policy import MaskedActorCriticPolicy, CustomDictFeaturesExtractor

def run_trained_agent(model_path: str, max_steps=100):
    env = FreecellEnv()

    policy_kwargs = dict(
        features_extractor_class=CustomDictFeaturesExtractor
    )

    # Załaduj model z pliku (ważne: podać env i policy_kwargs)
    model = PPO.load(model_path, env=env, policy_kwargs=policy_kwargs)

    print("=== RUNNING TRAINED AGENT ===")
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done and step < max_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info, action_str = env.step(action, return_action_str=True)
        total_reward += reward
        env.render()
        print(f"Step {step + 1}: action={action}, reward={reward:.2f}, done={done}, info={info}, action_str={action_str}")
        step += 1

    print(f"Test episode finished. Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    # Podaj ścieżkę do wytrenowanego modelu (dostosuj według lokalizacji)
    model_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'ppo_freecell_100000_steps.zip'))
    run_trained_agent(model_file)
