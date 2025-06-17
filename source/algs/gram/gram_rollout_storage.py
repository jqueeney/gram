# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 ETH Zurich, NVIDIA CORPORATION
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRAM rollout storage."""


from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage


class GRAMRolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.observations_history = None

            self.adversary_masks = None
            self.adversary_actions = None
            self.adversary_log_prob = None
            self.adversary_mean = None
            self.adversary_sigma = None

            self.robust_masks = None

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_history_length,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        adversary_action_shape,
        device="cpu",
    ):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)

        self.obs_history_length = obs_history_length

        self.observations_history = torch.zeros(
            num_transitions_per_env, num_envs, obs_history_length, *obs_shape, device=self.device
        )

        self.adversary_masks = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.adversary_actions = torch.zeros(
            num_transitions_per_env, num_envs, *adversary_action_shape, device=self.device
        )
        self.adversary_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.adversary_mu = torch.zeros(num_transitions_per_env, num_envs, *adversary_action_shape, device=self.device)
        self.adversary_sigma = torch.zeros(
            num_transitions_per_env, num_envs, *adversary_action_shape, device=self.device
        )

        self.robust_masks = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def init_extras(self, context):
        self.context = context.unsqueeze(dim=0).repeat(self.num_transitions_per_env, 1, 1)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.observations_history[self.step].copy_(transition.observations_history)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.adversary_masks[self.step].copy_(transition.adversary_masks)
        self.adversary_actions[self.step].copy_(transition.adversary_actions)
        self.adversary_log_prob[self.step].copy_(transition.adversary_log_prob.view(-1, 1))
        self.adversary_mu[self.step].copy_(transition.adversary_mean)
        self.adversary_sigma[self.step].copy_(transition.adversary_sigma)

        self.robust_masks[self.step].copy_(transition.robust_masks)

        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values

        if torch.sum(self.robust_masks == 1) > 0:
            robust_mean = self.advantages[self.robust_masks == 1].mean()
            robust_std = self.advantages[self.robust_masks == 1].std()
            self.advantages[self.robust_masks == 1] = (self.advantages[self.robust_masks == 1] - robust_mean) / (
                robust_std + 1e-8
            )

        if torch.sum(self.robust_masks == 0) > 0:
            adapt_mean = self.advantages[self.robust_masks == 0].mean()
            adapt_std = self.advantages[self.robust_masks == 0].std()
            self.advantages[self.robust_masks == 0] = (self.advantages[self.robust_masks == 0] - adapt_mean) / (
                adapt_std + 1e-8
            )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        observations_history = self.observations_history.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        rewards = self.rewards.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        adversary_masks = self.adversary_masks.flatten(0, 1)
        adversary_actions = self.adversary_actions.flatten(0, 1)
        old_adversary_log_prob = self.adversary_log_prob.flatten(0, 1)
        old_adversary_mu = self.adversary_mu.flatten(0, 1)
        old_adversary_sigma = self.adversary_sigma.flatten(0, 1)

        context = self.context.flatten(0, 1)

        robust_masks = self.robust_masks.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                obs_history_batch = observations_history[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                rewards_batch = rewards[batch_idx]
                dones_batch = dones[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                adversary_masks_batch = adversary_masks[batch_idx]
                adversary_actions_batch = adversary_actions[batch_idx]
                old_adversary_log_prob_batch = old_adversary_log_prob[batch_idx]
                old_adversary_mu_batch = old_adversary_mu[batch_idx]
                old_adversary_sigma_batch = old_adversary_sigma[batch_idx]

                context_batch = context[batch_idx]
                robust_masks_batch = robust_masks[batch_idx]

                yield (
                    obs_batch,
                    obs_history_batch,
                    critic_observations_batch,
                    rewards_batch,
                    dones_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    adversary_masks_batch,
                    adversary_actions_batch,
                    old_adversary_log_prob_batch,
                    old_adversary_mu_batch,
                    old_adversary_sigma_batch,
                    context_batch,
                    robust_masks_batch,
                )
