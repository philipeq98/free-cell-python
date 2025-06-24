import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.freecell_env import FreecellEnv
from environment.masked_policy import MaskedActorCriticPolicy, CustomDictFeaturesExtractor
from utils.checkpoint_callback import CheckpointCallback

def train_ppo(total_timesteps=500_000, log_interval=10_000):
    env = FreecellEnv()
    print(">>> ENV OBS SPACE:", env.observation_space['obs'].shape)

    policy_kwargs = dict(
        features_extractor_class=CustomDictFeaturesExtractor
    )

    model = PPO(
        MaskedActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        verbose=1
    )

    class ProgressCallback(BaseCallback):
        def __init__(self, log_interval):
            super().__init__()
            self.log_interval = log_interval
            self.last_log = 0
        def _on_step(self):
            step = self.num_timesteps
            if step - self.last_log >= self.log_interval:
                print(f"[Training] Timesteps: {step}/{total_timesteps}")
                self.last_log = step
            return True

    progress_callback = ProgressCallback(log_interval)
    checkpoint_callback = CheckpointCallback(
        save_freq=log_interval,
        save_path="/free-cell-python/checkpoints",
        name_prefix="ppo_freecell"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, checkpoint_callback]
    )

    print("\n=== TESTING TRAINED AGENT ===")
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done and step < 100:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info, action_str = env.step(action, return_action_str=True)
        total_reward += reward
        env.render()
        print(f"Step {step + 1}: action={action}, reward={reward:.2f}, done={done}, info={info}, action_str={action_str}")
        step += 1
    print(f"Test episode finished. Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    train_ppo()
