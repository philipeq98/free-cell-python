import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomDictFeaturesExtractor(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.extractors = nn.ModuleDict()
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            self.extractors[key] = FlattenExtractor(subspace)
            total_concat_size += self.extractors[key].features_dim

        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

    @property
    def features_dim(self):
        return self._features_dim


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)

        # Action mask
        mask = obs["action_mask"]
        if not isinstance(mask, th.Tensor):
            mask = th.as_tensor(mask).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        # Policy/value network
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Apply mask to logits
        masked_logits = distribution.distribution.logits + (mask + 1e-8).log()
        distribution.distribution.logits = masked_logits

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions
