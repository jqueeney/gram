# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 ETH Zurich, NVIDIA CORPORATION
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRAM adversary."""


from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import get_activation
from torch.distributions import Normal

from source.utils.torch_utils import weight_init


class GRAMAdversary(nn.Module):

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        adversary_hidden_dims=[512, 256, 128],
        use_log_std=True,
        activation="elu",
        adversary_init_noise_std=1.0,
        init_nn_weights=True,
        adversary_gain=1e-4,
        adversary_prob=0.0,
        adversary_clamp=5.0,
        adversary_random_scale=True,
        adversary_random_angle=False,
        adversary_max_scale_start=0.0,
        **kwargs,
    ):
        super().__init__()
        activation = get_activation(activation)

        # Adversary
        adversary_layers = []
        adversary_layers.append(nn.Linear(num_actor_obs, adversary_hidden_dims[0]))
        adversary_layers.append(activation)
        for layer_index in range(len(adversary_hidden_dims)):
            if layer_index == len(adversary_hidden_dims) - 1:
                adversary_layers.append(nn.Linear(adversary_hidden_dims[layer_index], num_actions))
            else:
                adversary_layers.append(
                    nn.Linear(adversary_hidden_dims[layer_index], adversary_hidden_dims[layer_index + 1])
                )
                adversary_layers.append(activation)
        self.adversary = nn.Sequential(*adversary_layers)

        print(f"Adversary MLP: {self.adversary}")

        # Action noise
        self.use_log_std = use_log_std
        if self.use_log_std:
            self.log_std = nn.Parameter(torch.log(adversary_init_noise_std * torch.ones(num_actions)))
        else:
            self.std = nn.Parameter(adversary_init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.adversary_prob = adversary_prob
        self.adversary_clamp = adversary_clamp

        self.adversary_random_scale = adversary_random_scale
        self.adversary_random_angle = adversary_random_angle

        self.adversary_max_scale_start = adversary_max_scale_start
        self.adversary_max_scale_cur = adversary_max_scale_start

        if (self.adversary_prob == 0.0) or self.adversary_random_angle:
            self.train_adversary = False
        else:
            self.train_adversary = True

        # Initialize NN weights
        if init_nn_weights:
            self.apply(weight_init)
            nn.init.xavier_normal_(self.adversary[-1].weight, gain=adversary_gain)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    def get_std(self):
        if self.use_log_std:
            return torch.exp(self.log_std)
        else:
            return self.std

    def update_distribution(self, observations):
        mean = self.adversary(observations)
        mean_clamp = mean.clamp(min=-self.adversary_clamp, max=self.adversary_clamp)
        if self.use_log_std:
            std = self.log_std.exp()
        else:
            std = self.std
        self.distribution = Normal(mean_clamp, mean_clamp * 0.0 + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        actions_mean_clamp = actions_mean.clamp(min=-self.adversary_clamp, max=self.adversary_clamp)
        return actions_mean_clamp

    def enforce_minimum_std(self):
        if self.use_log_std:
            self.log_std.data.clamp_(min=math.log(1e-2), max=math.log(10.0))
        else:
            self.std.data.clamp_(min=1e-2, max=10.0)

    def generate_adversary_masks(self, observations):
        mask_shape = observations.shape[:-1] + (1,)
        adversary_mask_probs = torch.ones(mask_shape, device=observations.device) * self.adversary_prob
        return torch.bernoulli(adversary_mask_probs)

    def angle_to_vector(self, theta):
        if self.adversary_random_angle:
            theta = torch.rand_like(theta) * 2 * torch.pi

        vector = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        if self.adversary_random_scale:
            scale = torch.rand_like(theta) * self.adversary_max_scale_cur
            vector = scale * vector

        return vector

    def update_magnitude(self, train_progress):
        self.adversary_max_scale_cur = (
            self.adversary_max_scale_start + (1.0 - self.adversary_max_scale_start) * train_progress
        )
