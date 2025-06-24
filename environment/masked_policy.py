import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical

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
        mask = obs["action_mask"]
        if not isinstance(mask, th.Tensor):
            mask = th.as_tensor(mask).to(self.device)
        else:
            mask = mask.to(self.device)

        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)

        # mask to binary tensor (0 or 1), akcje niedozwolone => logit = -1e9
        neg_inf = -1e9
        masked_logits = logits.clone()
        masked_logits[mask == 0] = neg_inf

        # Utwórz rozkład kategorii z zmodyfikowanymi logitami
        distribution = Categorical(logits=masked_logits)

        actions = distribution.sample() if not deterministic else distribution.probs.argmax(dim=1)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions
