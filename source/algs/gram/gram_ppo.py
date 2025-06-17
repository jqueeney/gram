# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 ETH Zurich, NVIDIA CORPORATION
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRAM PPO."""


from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO

from source.algs.gram.gram_rollout_storage import GRAMRolloutStorage


class GRAMPPO(PPO):

    def init_extras(
        self,
        context,
        adversary_policy,
        env,
        train_and_finetune,
        num_learning_epochs_finetune,
        history_loss_actions,
        history_loss_robust_as_zeros,
        history_loss_adapt_only,
        history_loss_epinet_separate,
        history_lr,
        adversary_update_every,
    ):

        self.context = context
        self.storage.init_extras(context)

        self.train_and_finetune = train_and_finetune
        if num_learning_epochs_finetune is not None:
            self.num_learning_epochs_finetune = num_learning_epochs_finetune
        else:
            self.num_learning_epochs_finetune = self.num_learning_epochs

        self.history_loss_actions = history_loss_actions
        self.history_loss_robust_as_zeros = history_loss_robust_as_zeros
        self.history_loss_adapt_only = history_loss_adapt_only
        self.history_loss_epinet_separate = history_loss_epinet_separate
        self.history_lr = history_lr

        self.adversary_policy = adversary_policy

        self.env = env

        self.since_adversary_update = 0
        self.adversary_update_every = adversary_update_every
        self.adversary_lr = self.learning_rate

        self.optimizer = optim.Adam(self.actor_critic.opt_params, lr=self.learning_rate)
        self.adversary_optimizer = optim.Adam(self.adversary_policy.parameters(), lr=self.adversary_lr)
        self.history_optimizer = optim.Adam(self.actor_critic.history_params, lr=self.history_lr)

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        obs_history_length,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        adversary_action_shape,
    ):
        self.storage = GRAMRolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_history_length,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            adversary_action_shape,
            self.device,
        )
        self.transition = GRAMRolloutStorage.Transition()

    def act(self, obs, critic_obs, obs_history, robust_masks):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, self.context, obs_history, robust_masks).detach()
        self.transition.values = self.actor_critic.evaluate(
            critic_obs, self.context, obs_history, robust_masks
        ).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.observations_history = obs_history
        # Compute adversary stats
        self.transition.adversary_actions = self.adversary_policy.act(obs).detach()
        self.transition.adversary_log_prob = self.adversary_policy.get_actions_log_prob(
            self.transition.adversary_actions
        ).detach()
        self.transition.adversary_mean = self.adversary_policy.action_mean.detach()
        self.transition.adversary_sigma = self.adversary_policy.action_std.detach()

        adversary_masks = self.adversary_policy.generate_adversary_masks(obs).detach()
        self.transition.adversary_masks = adversary_masks * robust_masks
        self.transition.robust_masks = robust_masks

        self.adversary_update_sim(self.transition.adversary_actions, self.transition.adversary_masks)

        return self.transition.actions

    def adversary_update_sim(self, adversary_actions, adversary_masks):
        adversary_cfg = self.env.unwrapped.event_manager.get_term_cfg("adversary_push")
        adversary_cfg.params["adversary_actions"] = self.adversary_policy.angle_to_vector(adversary_actions)
        adversary_cfg.params["adversary_masks"] = adversary_masks

    def compute_returns(self, last_critic_obs, obs_history, robust_masks):
        last_values = self.actor_critic.evaluate(last_critic_obs, self.context, obs_history, robust_masks).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_history_loss = 0

        mean_latent_error_l2 = 0
        mean_latent_error_kl = 0

        mean_latent_l2 = 0
        mean_latent_abs = 0
        mean_latent_max = 0
        mean_latent_std = 0

        mean_adversary_l2 = 0
        mean_adversary_abs = 0
        mean_adversary_max = 0

        mean_uncertainty_metric = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            obs_history_batch,
            critic_obs_batch,
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
        ) in generator:
            self.actor_critic.act(obs_batch, context_batch, obs_history_batch, robust_masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, context_batch, obs_history_batch, robust_masks_batch
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.ac_adapt_params, self.max_grad_norm)
            if not self.actor_critic.robust_as_zeros:
                nn.utils.clip_grad_norm_(self.actor_critic.ac_robust_params, self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic.context_params, self.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                self.actor_critic.enforce_minimum_std()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # History encoder loss
            if self.train_and_finetune:
                with torch.no_grad():
                    latent_context = self.actor_critic.encode_context(context_batch)
                    if self.history_loss_robust_as_zeros and self.actor_critic.robust_as_zeros:
                        robust_latent_context = torch.zeros_like(latent_context.detach())
                        latent_context = (
                            robust_masks_batch * robust_latent_context + (1 - robust_masks_batch) * latent_context
                        )

                    self.actor_critic.update_distribution(obs_batch, latent_context, robust_masks_batch)
                    mu_context = self.actor_critic.action_mean

                if self.actor_critic.history_epinet:
                    latent_history_all, latent_base, epinet_all = self.actor_critic.encode_history_epinet(
                        obs_history_batch
                    )
                    if self.history_loss_epinet_separate:
                        latent_squared_error_base = (latent_context - latent_base).pow(2).sum(dim=-1)
                        latent_squared_error_epinet = epinet_all.pow(2).sum(dim=-1).mean(dim=0)
                        latent_squared_error = latent_squared_error_base + latent_squared_error_epinet
                    else:
                        latent_squared_error = (latent_context - latent_history_all).pow(2).sum(dim=-1).mean(dim=0)

                    if self.history_loss_adapt_only:
                        history_loss = (latent_squared_error * (1 - robust_masks_batch.squeeze(dim=-1))).sum() / (
                            1 - robust_masks_batch
                        ).sum().clamp(min=1)
                    else:
                        history_loss = latent_squared_error.mean()

                    if self.history_loss_epinet_separate:
                        if self.history_loss_actions:
                            self.actor_critic.update_distribution(obs_batch, latent_base, robust_masks_batch)
                            mu_history = self.actor_critic.action_mean
                            history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)

                            if self.history_loss_adapt_only:
                                history_action_loss_ave = (
                                    history_action_loss * (1 - robust_masks_batch.squeeze(dim=-1))
                                ).sum() / (1 - robust_masks_batch).sum().clamp(min=1)
                            else:
                                history_action_loss_ave = history_action_loss.mean()

                                history_loss += history_action_loss_ave
                        else:
                            with torch.no_grad():
                                self.actor_critic.update_distribution(obs_batch, latent_base, robust_masks_batch)
                                mu_history = self.actor_critic.action_mean
                                history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)
                    else:
                        obs_batch_repeat = obs_batch.unsqueeze(dim=0).repeat(
                            (self.actor_critic.history_epinet_num_samples, 1, 1)
                        )
                        robust_masks_batch_repeat = robust_masks_batch.unsqueeze(dim=0).repeat(
                            (self.actor_critic.history_epinet_num_samples, 1, 1)
                        )
                        if self.history_loss_actions:
                            self.actor_critic.update_distribution(
                                obs_batch_repeat, latent_history_all, robust_masks_batch_repeat
                            )
                            mu_history = self.actor_critic.action_mean
                            history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1).mean(dim=0)

                            if self.history_loss_adapt_only:
                                history_action_loss_ave = (
                                    history_action_loss * (1 - robust_masks_batch.squeeze(dim=-1))
                                ).sum() / (1 - robust_masks_batch).sum().clamp(min=1)
                            else:
                                history_action_loss_ave = history_action_loss.mean()

                            history_loss += history_action_loss_ave
                        else:
                            with torch.no_grad():
                                self.actor_critic.update_distribution(
                                    obs_batch_repeat, latent_history_all, robust_masks_batch_repeat
                                )
                                mu_history = self.actor_critic.action_mean
                                history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1).mean(dim=0)

                else:
                    latent_history = self.actor_critic.encode_history(obs_history_batch)
                    latent_squared_error = (latent_context - latent_history).pow(2).sum(dim=-1)

                    if self.history_loss_adapt_only:
                        history_loss = (latent_squared_error * (1 - robust_masks_batch.squeeze(dim=-1))).sum() / (
                            1 - robust_masks_batch
                        ).sum().clamp(min=1)
                    else:
                        history_loss = latent_squared_error.mean()

                    if self.history_loss_actions:
                        self.actor_critic.update_distribution(obs_batch, latent_history, robust_masks_batch)
                        mu_history = self.actor_critic.action_mean
                        history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)
                        history_loss += history_action_loss.mean()
                    else:
                        with torch.no_grad():
                            self.actor_critic.update_distribution(obs_batch, latent_history, robust_masks_batch)
                            mu_history = self.actor_critic.action_mean
                            history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)

                # History gradient step
                self.history_optimizer.zero_grad()
                history_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.history_params, self.max_grad_norm)
                self.history_optimizer.step()

            else:
                with torch.no_grad():
                    latent_context = self.actor_critic.encode_context(context_batch)

                history_loss = torch.zeros_like(value_loss)

            mean_history_loss += history_loss.item()

            if self.train_and_finetune:
                mean_latent_error_l2 += latent_squared_error.mean().item()
                mean_latent_error_kl += history_action_loss.mean().item()

            mean_latent_l2 += torch.norm(latent_context, dim=-1).mean().item()
            mean_latent_abs += torch.abs(latent_context).mean().item()
            mean_latent_max += torch.abs(latent_context).amax(dim=-1).mean().item()
            mean_latent_std += torch.std(latent_context, dim=0).mean().item()

            mean_adversary_l2 += torch.norm(adversary_actions_batch, dim=-1).mean().item()
            mean_adversary_abs += torch.abs(adversary_actions_batch).mean().item()
            mean_adversary_max += torch.abs(adversary_actions_batch).amax().item()

            uncertainty_metric = self.actor_critic.calculate_uncertainty_metric(obs_history_batch)
            mean_uncertainty_metric += uncertainty_metric.mean().item()

            # Adversary update
            if (
                self.since_adversary_update % self.adversary_update_every == 0
            ) and self.adversary_policy.train_adversary:
                self.adversary_policy.act(obs_batch)
                adversary_log_prob_batch = self.adversary_policy.get_actions_log_prob(adversary_actions_batch)
                adversary_mu_batch = self.adversary_policy.action_mean
                adversary_sigma_batch = self.adversary_policy.action_std

                # KL
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(adversary_sigma_batch / old_adversary_sigma_batch + 1.0e-5)
                            + (
                                torch.square(old_adversary_sigma_batch)
                                + torch.square(old_adversary_mu_batch - adversary_mu_batch)
                            )
                            / (2.0 * torch.square(adversary_sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.adversary_lr = max(1e-5, self.adversary_lr / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.adversary_lr = min(1e-2, self.adversary_lr * 1.5)

                        for param_group in self.adversary_optimizer.param_groups:
                            param_group["lr"] = self.adversary_lr

                # Surrogate loss
                adversary_batch_size = max(adversary_masks_batch.sum().item(), 1.0)

                # center adversary advantages
                adversary_advantages_batch_mean = (
                    advantages_batch * adversary_masks_batch
                ).sum() / adversary_batch_size
                adversary_advantages_batch = (
                    (advantages_batch - adversary_advantages_batch_mean) * adversary_masks_batch * -1
                )

                adversary_ratio = torch.exp(adversary_log_prob_batch - torch.squeeze(old_adversary_log_prob_batch))
                adversary_surrogate = -torch.squeeze(adversary_advantages_batch) * adversary_ratio
                adversary_surrogate_clipped = -torch.squeeze(adversary_advantages_batch) * torch.clamp(
                    adversary_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                adversary_loss = (
                    torch.max(adversary_surrogate, adversary_surrogate_clipped).sum() / adversary_batch_size
                )

                # Gradient step
                self.adversary_optimizer.zero_grad()
                adversary_loss.backward()
                nn.utils.clip_grad_norm_(self.adversary_policy.parameters(), self.max_grad_norm)
                self.adversary_optimizer.step()
                with torch.no_grad():
                    self.adversary_policy.enforce_minimum_std()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        mean_history_loss /= num_updates

        mean_latent_error_l2 /= num_updates
        mean_latent_error_kl /= num_updates

        mean_latent_l2 /= num_updates
        mean_latent_abs /= num_updates
        mean_latent_max /= num_updates
        mean_latent_std /= num_updates

        mean_adversary_l2 /= num_updates
        mean_adversary_abs /= num_updates
        mean_adversary_max /= num_updates

        mean_uncertainty_metric /= num_updates

        self.storage.clear()

        self.since_adversary_update += 1
        self.since_adversary_update = self.since_adversary_update % self.adversary_update_every

        loss_stats = {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_history_loss": mean_history_loss,
        }

        update_stats = {
            "mean_adapt_std": self.actor_critic.get_std().mean().item(),
            "mean_robust_std": self.actor_critic.get_robust_std().mean().item(),
            "mean_latent_error_l2": mean_latent_error_l2,
            "mean_latent_error_kl": mean_latent_error_kl,
            "mean_latent_l2": mean_latent_l2,
            "mean_latent_abs": mean_latent_abs,
            "mean_latent_max": mean_latent_max,
            "mean_latent_std": mean_latent_std,
            "mean_adversary_std": self.adversary_policy.get_std().mean().item(),
            "mean_adversary_l2": mean_adversary_l2,
            "mean_adversary_abs": mean_adversary_abs,
            "mean_adversary_max": mean_adversary_max,
            "uncertainty_metric_ave": mean_uncertainty_metric,
        }

        return loss_stats, update_stats

    def finetune(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_history_loss = 0

        mean_latent_error_l2 = 0
        mean_latent_error_kl = 0
        mean_epi_error_l2 = 0

        mean_latent_l2 = 0
        mean_latent_abs = 0
        mean_latent_max = 0
        mean_latent_std = 0

        mean_uncertainty_metric = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs_finetune)
        for (
            obs_batch,
            obs_history_batch,
            critic_obs_batch,
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
        ) in generator:

            # History encoder loss
            with torch.no_grad():
                latent_context = self.actor_critic.encode_context(context_batch)
                if self.history_loss_robust_as_zeros and self.actor_critic.robust_as_zeros:
                    robust_latent_context = torch.zeros_like(latent_context.detach())
                    latent_context = (
                        robust_masks_batch * robust_latent_context + (1 - robust_masks_batch) * latent_context
                    )

                self.actor_critic.update_distribution(obs_batch, latent_context, robust_masks_batch)
                mu_context = self.actor_critic.action_mean

            if self.actor_critic.history_epinet:
                latent_history_all, latent_base, epinet_all = self.actor_critic.encode_history_epinet(obs_history_batch)
                if self.history_loss_epinet_separate:
                    epi_squared_error = epinet_all.pow(2).sum(dim=-1).mean(dim=0)
                    latent_squared_error = (latent_context - latent_base).pow(2).sum(dim=-1)
                    history_squared_error = latent_squared_error + epi_squared_error
                else:
                    latent_squared_error = (latent_context - latent_history_all).pow(2).sum(dim=-1).mean(dim=0)
                    epi_squared_error = torch.zeros_like(latent_squared_error)
                    history_squared_error = latent_squared_error

                history_loss = history_squared_error.mean()

                if self.history_loss_epinet_separate:
                    if self.history_loss_actions:
                        self.actor_critic.update_distribution(obs_batch, latent_base, robust_masks_batch)
                        mu_history = self.actor_critic.action_mean
                        history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)

                        history_action_loss_ave = history_action_loss.mean()
                        history_loss += history_action_loss_ave
                    else:
                        with torch.no_grad():
                            self.actor_critic.update_distribution(obs_batch, latent_base, robust_masks_batch)
                            mu_history = self.actor_critic.action_mean
                            history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)
                else:
                    obs_batch_repeat = obs_batch.unsqueeze(dim=0).repeat(
                        (self.actor_critic.history_epinet_num_samples, 1, 1)
                    )
                    robust_masks_batch_repeat = robust_masks_batch.unsqueeze(dim=0).repeat(
                        (self.actor_critic.history_epinet_num_samples, 1, 1)
                    )
                    if self.history_loss_actions:
                        self.actor_critic.update_distribution(
                            obs_batch_repeat, latent_history_all, robust_masks_batch_repeat
                        )
                        mu_history = self.actor_critic.action_mean
                        history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1).mean(dim=0)

                        history_action_loss_ave = history_action_loss.mean()

                        history_loss += history_action_loss_ave
                    else:
                        with torch.no_grad():
                            self.actor_critic.update_distribution(
                                obs_batch_repeat, latent_history_all, robust_masks_batch_repeat
                            )
                            mu_history = self.actor_critic.action_mean
                            history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1).mean(dim=0)

            else:
                latent_history = self.actor_critic.encode_history(obs_history_batch)
                latent_squared_error = (latent_context - latent_history).pow(2).sum(dim=-1)

                history_loss = latent_squared_error.mean()

                if self.history_loss_actions:
                    self.actor_critic.update_distribution(obs_batch, latent_history, robust_masks_batch)
                    mu_history = self.actor_critic.action_mean
                    history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)
                    history_loss += history_action_loss.mean()
                else:
                    with torch.no_grad():
                        self.actor_critic.update_distribution(obs_batch, latent_history, robust_masks_batch)
                        mu_history = self.actor_critic.action_mean
                        history_action_loss = (mu_context - mu_history).pow(2).sum(dim=-1)

                epi_squared_error = torch.zeros_like(history_action_loss)

            # History gradient step
            self.history_optimizer.zero_grad()
            history_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.history_params, self.max_grad_norm)
            self.history_optimizer.step()

            mean_history_loss += history_loss.item()

            mean_latent_error_l2 += latent_squared_error.mean().item()
            mean_latent_error_kl += history_action_loss.mean().item()
            mean_epi_error_l2 += epi_squared_error.mean().item()

            mean_latent_l2 += torch.norm(latent_context, dim=-1).mean().item()
            mean_latent_abs += torch.abs(latent_context).mean().item()
            mean_latent_max += torch.abs(latent_context).amax(dim=-1).mean().item()
            mean_latent_std += torch.std(latent_context, dim=0).mean().item()

            uncertainty_metric = self.actor_critic.calculate_uncertainty_metric(obs_history_batch)
            mean_uncertainty_metric += uncertainty_metric.mean().item()

        num_updates = self.num_learning_epochs_finetune * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        mean_history_loss /= num_updates

        mean_latent_error_l2 /= num_updates
        mean_latent_error_kl /= num_updates
        mean_epi_error_l2 /= num_updates

        mean_latent_l2 /= num_updates
        mean_latent_abs /= num_updates
        mean_latent_max /= num_updates
        mean_latent_std /= num_updates

        mean_uncertainty_metric /= num_updates

        self.storage.clear()

        loss_stats = {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_history_loss": mean_history_loss,
        }

        update_stats = {
            "mean_latent_error_l2": mean_latent_error_l2,
            "mean_latent_error_kl": mean_latent_error_kl,
            "mean_epi_error_l2": mean_epi_error_l2,
            "mean_latent_l2": mean_latent_l2,
            "mean_latent_abs": mean_latent_abs,
            "mean_latent_max": mean_latent_max,
            "mean_latent_std": mean_latent_std,
            "uncertainty_metric_ave": mean_uncertainty_metric,
        }

        return loss_stats, update_stats
