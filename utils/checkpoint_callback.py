import os
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = 'model', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = os.path.abspath(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_file = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"[CheckpointCallback] Saved checkpoint: {save_file}")
        return True
