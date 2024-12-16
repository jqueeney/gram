# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 ETH Zurich, NVIDIA CORPORATION
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRAM actor-critic."""


from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import get_activation
from torch.distributions import Normal

from source.utils.torch_utils import weight_init


class GRAMActorCritic(nn.Module):

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_context,
        num_latent,
        obs_history_length,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        context_hidden_dims=[64, 64],
        use_log_std=True,
        history_epinet=True,
        history_epinet_apply_bias=False,
        history_epinet_output_type="base",
        history_epinet_finetune_scale=True,
        history_epinet_finetune_shift=True,
        history_epinet_finetune_min_quantile=0.90,
        history_epinet_finetune_max_quantile=0.99,
        history_epinet_finetune_target=0.01,
        history_epinet_type="mlp_dot",
        history_epinet_input_type="concat",
        history_epinet_gain=1e-4,
        history_epinet_prior_gain=1.0,
        history_epinet_hidden_dims=[16, 16],
        history_epinet_dim=8,
        history_epinet_num_samples=8,
        history_epinet_coef=1.0,
        context_encoder_clamp=False,
        context_encoder_clamp_value=1.0,
        context_encoder_stochastic=False,
        context_encoder_noise_std=0.25,
        history_encoder_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        init_nn_weights=True,
        actor_gain=1e-4,
        critic_gain=1e-4,
        context_gain=1e-4,
        history_gain=1e-4,
        no_context_critic=False,
        latent_detach_actor=False,
        latent_detach_critic=False,
        robust_as_zeros=True,
        act_from_context=True,
        act_inference_adapt=False,
        act_inference_robust=False,
        robust_use_latent_actor=False,
        robust_use_latent_critic=False,
        **kwargs,
    ):
        super().__init__()
        activation = get_activation(activation)

        if num_context == 0:
            raise ValueError("Algorithm requires context definition.")

        self.no_context_critic = no_context_critic
        if self.no_context_critic:
            num_latent_critic = 0
        else:
            num_latent_critic = num_latent

        # -------------------------------------------------------------------- #
        # Adaptive components
        # -------------------------------------------------------------------- #

        mlp_input_dim_a = num_actor_obs + num_latent
        mlp_input_dim_c = num_critic_obs + num_latent_critic

        # Adaptive policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Adaptive value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Context Actor MLP: {self.actor}")
        print(f"Context Critic MLP: {self.critic}")

        # Action noise
        self.use_log_std = use_log_std
        if self.use_log_std:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # Context encoder
        context_layers = []
        context_layers.append(nn.Linear(num_context, context_hidden_dims[0]))
        context_layers.append(activation)
        for layer_index in range(len(context_hidden_dims)):
            if layer_index == len(context_hidden_dims) - 1:
                context_layers.append(nn.Linear(context_hidden_dims[layer_index], num_latent))
            else:
                context_layers.append(nn.Linear(context_hidden_dims[layer_index], context_hidden_dims[layer_index + 1]))
                context_layers.append(activation)
        self.context_encoder = nn.Sequential(*context_layers)

        print(f"Context Encoder MLP: {self.context_encoder}")

        self.context_encoder_stochastic = context_encoder_stochastic
        self.context_encoder_noise_std = context_encoder_noise_std

        # History encoder
        self.obs_history_length = obs_history_length

        self.history_epinet = history_epinet
        self.history_epinet_apply_bias = history_epinet_apply_bias
        self.history_epinet_output_type = history_epinet_output_type
        self.history_epinet_finetune_scale = history_epinet_finetune_scale
        self.history_epinet_finetune_shift = history_epinet_finetune_shift
        self.history_epinet_finetune_min_quantile = history_epinet_finetune_min_quantile
        self.history_epinet_finetune_max_quantile = history_epinet_finetune_max_quantile
        self.history_epinet_finetune_target = history_epinet_finetune_target

        self.validation_quantiles_q = nn.Parameter(torch.linspace(0, 1, 101), requires_grad=False)
        self.validation_quantiles_value = nn.Parameter(torch.zeros(101), requires_grad=False)
        self.validation_quantiles_saved = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.set_uncertainty_params()

        self.history_epinet_type = history_epinet_type
        self.history_epinet_input_type = history_epinet_input_type
        self.history_epinet_dim = history_epinet_dim
        self.history_epinet_num_samples = history_epinet_num_samples
        self.history_epinet_coef = history_epinet_coef
        self.history_epinet_coef_active = history_epinet_coef

        encode_layers = []
        encode_layers.append(nn.Flatten())
        encode_layers.append(nn.Linear(num_actor_obs * self.obs_history_length, history_encoder_hidden_dims[0]))
        encode_layers.append(activation)
        for layer_index in range(len(history_encoder_hidden_dims)):
            if layer_index == len(history_encoder_hidden_dims) - 1:
                encode_layers.append(nn.Linear(history_encoder_hidden_dims[layer_index], num_latent))
            else:
                encode_layers.append(
                    nn.Linear(history_encoder_hidden_dims[layer_index], history_encoder_hidden_dims[layer_index + 1])
                )
                encode_layers.append(activation)
        self.history_encoder = nn.Sequential(*encode_layers)

        print(f"History Encoder: {self.history_encoder}")

        if self.history_epinet:
            self.input_flattener = nn.Flatten()

            if self.history_epinet_input_type == "full":
                epinet_in_dim = num_actor_obs * self.obs_history_length + self.history_epinet_dim
            elif self.history_epinet_input_type == "concat":
                epinet_in_dim = (
                    num_actor_obs * self.obs_history_length + history_encoder_hidden_dims[-1] + self.history_epinet_dim
                )
            else:
                epinet_in_dim = history_encoder_hidden_dims[-1] + self.history_epinet_dim

            if self.history_epinet_type == "mlp":
                epinet_layers = []
                epinet_layers.append(nn.Linear(epinet_in_dim, history_epinet_hidden_dims[0]))
                epinet_layers.append(activation)
                for layer_index in range(len(history_epinet_hidden_dims)):
                    if layer_index == len(history_epinet_hidden_dims) - 1:
                        epinet_layers.append(nn.Linear(history_epinet_hidden_dims[layer_index], num_latent))
                    else:
                        epinet_layers.append(
                            nn.Linear(
                                history_epinet_hidden_dims[layer_index], history_epinet_hidden_dims[layer_index + 1]
                            )
                        )
                        epinet_layers.append(activation)
                self.history_epinet_trainable = nn.Sequential(*epinet_layers)

                epinet_layers = []
                epinet_layers.append(nn.Linear(epinet_in_dim, history_epinet_hidden_dims[0]))
                epinet_layers.append(activation)
                for layer_index in range(len(history_epinet_hidden_dims)):
                    if layer_index == len(history_epinet_hidden_dims) - 1:
                        epinet_layers.append(nn.Linear(history_epinet_hidden_dims[layer_index], num_latent))
                    else:
                        epinet_layers.append(
                            nn.Linear(
                                history_epinet_hidden_dims[layer_index], history_epinet_hidden_dims[layer_index + 1]
                            )
                        )
                        epinet_layers.append(activation)
                self.history_epinet_prior = nn.Sequential(*epinet_layers)
                self.history_epinet_prior.requires_grad_(requires_grad=False)

            elif self.history_epinet_type == "mlp_dot":
                epinet_layers = []
                epinet_layers.append(nn.Linear(epinet_in_dim, history_epinet_hidden_dims[0]))
                epinet_layers.append(activation)
                for layer_index in range(len(history_epinet_hidden_dims)):
                    if layer_index == len(history_epinet_hidden_dims) - 1:
                        epinet_layers.append(
                            nn.Linear(history_epinet_hidden_dims[layer_index], num_latent * self.history_epinet_dim)
                        )
                    else:
                        epinet_layers.append(
                            nn.Linear(
                                history_epinet_hidden_dims[layer_index], history_epinet_hidden_dims[layer_index + 1]
                            )
                        )
                        epinet_layers.append(activation)
                self.history_epinet_trainable = nn.Sequential(*epinet_layers)

                epinet_layers = []
                epinet_layers.append(nn.Linear(epinet_in_dim, history_epinet_hidden_dims[0]))
                epinet_layers.append(activation)
                for layer_index in range(len(history_epinet_hidden_dims)):
                    if layer_index == len(history_epinet_hidden_dims) - 1:
                        epinet_layers.append(
                            nn.Linear(history_epinet_hidden_dims[layer_index], num_latent * self.history_epinet_dim)
                        )
                    else:
                        epinet_layers.append(
                            nn.Linear(
                                history_epinet_hidden_dims[layer_index], history_epinet_hidden_dims[layer_index + 1]
                            )
                        )
                        epinet_layers.append(activation)
                self.history_epinet_prior = nn.Sequential(*epinet_layers)
                self.history_epinet_prior.requires_grad_(requires_grad=False)

        # -------------------------------------------------------------------- #
        # Robust components
        # -------------------------------------------------------------------- #

        self.robust_as_zeros = robust_as_zeros
        self.robust_use_latent_actor = robust_use_latent_actor
        self.robust_use_latent_critic = robust_use_latent_critic

        if self.robust_as_zeros:
            self.robust_actor = self.actor
            self.robust_critic = self.critic
            if self.use_log_std:
                self.robust_log_std = self.log_std
            else:
                self.robust_std = self.std
        else:
            if self.robust_use_latent_actor:
                mlp_input_dim_a_robust = num_actor_obs + num_latent
            else:
                mlp_input_dim_a_robust = num_actor_obs
            if self.robust_use_latent_critic:
                mlp_input_dim_c_robust = num_critic_obs + num_latent_critic
            else:
                mlp_input_dim_c_robust = num_critic_obs

            # Robust policy
            robust_actor_layers = []
            robust_actor_layers.append(nn.Linear(mlp_input_dim_a_robust, actor_hidden_dims[0]))
            robust_actor_layers.append(activation)
            for layer_index in range(len(actor_hidden_dims)):
                if layer_index == len(actor_hidden_dims) - 1:
                    robust_actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                else:
                    robust_actor_layers.append(
                        nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1])
                    )
                    robust_actor_layers.append(activation)
            self.robust_actor = nn.Sequential(*robust_actor_layers)

            # Robust value function
            robust_critic_layers = []
            robust_critic_layers.append(nn.Linear(mlp_input_dim_c_robust, critic_hidden_dims[0]))
            robust_critic_layers.append(activation)
            for layer_index in range(len(critic_hidden_dims)):
                if layer_index == len(critic_hidden_dims) - 1:
                    robust_critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
                else:
                    robust_critic_layers.append(
                        nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1])
                    )
                    robust_critic_layers.append(activation)
            self.robust_critic = nn.Sequential(*robust_critic_layers)

            print(f"Robust Actor MLP: {self.robust_actor}")
            print(f"Robust Critic MLP: {self.robust_critic}")

            # Action noise
            if self.use_log_std:
                self.robust_log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                self.robust_std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # -------------------------------------------------------------------- #

        # Distribution
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Action options
        self.act_from_context = act_from_context
        self.act_inference_adapt = act_inference_adapt
        self.act_inference_robust = act_inference_robust

        # Context training arguments
        self.latent_detach_actor = latent_detach_actor
        self.latent_detach_critic = latent_detach_critic
        self.context_encoder_clamp = context_encoder_clamp
        self.context_encoder_clamp_value = context_encoder_clamp_value

        # Initialize NN weights
        if init_nn_weights:
            self.apply(weight_init)
            nn.init.xavier_normal_(self.actor[-1].weight, gain=actor_gain)
            nn.init.xavier_normal_(self.critic[-1].weight, gain=critic_gain)
            nn.init.xavier_normal_(self.context_encoder[-1].weight, gain=context_gain)
            nn.init.xavier_normal_(self.history_encoder[-1].weight, gain=history_gain)

            if not self.robust_as_zeros:
                nn.init.xavier_normal_(self.robust_actor[-1].weight, gain=actor_gain)
                nn.init.xavier_normal_(self.robust_critic[-1].weight, gain=critic_gain)

        if self.history_epinet:
            nn.init.xavier_normal_(self.history_epinet_trainable[-1].weight, gain=history_epinet_gain)
            nn.init.xavier_normal_(self.history_epinet_prior[-1].weight, gain=history_epinet_prior_gain)

        # Actor / critic parameters
        if self.use_log_std:
            std_params = [self.log_std]
            robust_std_params = [self.robust_log_std]
        else:
            std_params = [self.std]
            robust_std_params = [self.robust_std]

        if self.robust_as_zeros:
            self.ac_params = itertools.chain(
                self.actor.parameters(),
                std_params,
                self.critic.parameters(),
                self.context_encoder.parameters(),
            )

            self.ac_adapt_params = itertools.chain(
                self.actor.parameters(),
                std_params,
                self.critic.parameters(),
            )
        else:
            self.ac_params = itertools.chain(
                self.actor.parameters(),
                std_params,
                self.critic.parameters(),
                self.context_encoder.parameters(),
                self.robust_actor.parameters(),
                robust_std_params,
                self.robust_critic.parameters(),
            )

            self.ac_robust_params = itertools.chain(
                self.robust_actor.parameters(),
                robust_std_params,
                self.robust_critic.parameters(),
            )

            self.ac_adapt_params = itertools.chain(
                self.actor.parameters(),
                std_params,
                self.critic.parameters(),
            )

        self.context_params = self.context_encoder.parameters()
        self.opt_params = self.ac_params

        if self.history_epinet:
            self.history_params = itertools.chain(
                self.history_encoder.parameters(),
                self.history_epinet_trainable.parameters(),
            )
        else:
            self.history_params = self.history_encoder.parameters()

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

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_std(self):
        if self.use_log_std:
            return torch.exp(self.log_std)
        else:
            return self.std

    def get_robust_std(self):
        if self.use_log_std:
            return torch.exp(self.robust_log_std)
        else:
            return self.robust_std

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_distribution(self, observations, latent, robust_masks):
        actor_input = torch.cat([observations, latent], dim=-1)
        if self.robust_as_zeros:
            robust_latent = torch.zeros_like(latent.detach())
            robust_actor_input = torch.cat([observations, robust_latent], dim=-1)
        else:
            if self.robust_use_latent_actor:
                robust_actor_input = actor_input
            else:
                robust_actor_input = observations

        robust_mean = self.robust_actor(robust_actor_input)
        adapt_mean = self.actor(actor_input)

        mean = robust_masks * robust_mean + (1 - robust_masks) * adapt_mean
        mean_clamp = mean.clamp(min=-20.0, max=20.0)

        if self.use_log_std:
            log_std = robust_masks * self.robust_log_std + (1 - robust_masks) * self.log_std
            std = torch.exp(log_std)
        else:
            std = robust_masks * self.robust_std + (1 - robust_masks) * self.std

        self.distribution = Normal(mean_clamp, mean_clamp * 0.0 + std)

    def act(self, observations, context, observations_history, robust_masks, **kwargs):
        latent = self.calculate_latent(context, observations_history, from_context=self.act_from_context)

        if self.latent_detach_actor:
            self.update_distribution(observations, latent.detach(), robust_masks)
        else:
            self.update_distribution(observations, latent, robust_masks)

        return self.distribution.sample()

    def act_inference_from_mask(
        self, observations, context, observations_history, robust_masks=None, deterministic=False
    ):
        latent = self.calculate_latent(context, observations_history, from_context=self.act_from_context)

        if self.act_inference_robust:
            robust_masks = torch.ones((observations.shape[0], 1), device=observations.device)
        elif self.act_inference_adapt or (robust_masks is None):
            robust_masks = torch.zeros((observations.shape[0], 1), device=observations.device)

        self.update_distribution(observations, latent, robust_masks)

        if deterministic:
            actions = self.action_mean
        else:
            actions = self.distribution.sample()

        return actions

    def act_inference(self, observations, context, observations_history, uncertainty_metric, deterministic=False):
        latent = self.calculate_latent(context, observations_history, from_context=self.act_from_context)

        robust_masks = self.calculate_uncertainty_mask(uncertainty_metric)

        if self.act_inference_robust:
            robust_masks = torch.ones_like(robust_masks)
        elif self.act_inference_adapt:
            robust_masks = torch.zeros_like(robust_masks)

        self.update_distribution(observations, latent, robust_masks)

        if deterministic:
            actions = self.action_mean
        else:
            actions = self.distribution.sample()

        return actions

    def evaluate(self, critic_observations, context, observations_history, robust_masks, **kwargs):
        if self.no_context_critic:
            critic_input = critic_observations
            robust_critic_input = critic_input
        else:
            latent = self.calculate_latent(context, observations_history, from_context=self.act_from_context)
            if self.latent_detach_critic:
                critic_input = torch.cat([critic_observations, latent.detach()], dim=-1)
            else:
                critic_input = torch.cat([critic_observations, latent], dim=-1)

            if self.robust_as_zeros:
                robust_latent = torch.zeros_like(latent.detach())
                robust_critic_input = torch.cat([critic_observations, robust_latent], dim=-1)
            else:
                if self.robust_use_latent_critic:
                    robust_critic_input = critic_input
                else:
                    robust_critic_input = critic_observations

        robust_value = self.robust_critic(robust_critic_input)
        adapt_value = self.critic(critic_input)
        value = robust_masks * robust_value + (1 - robust_masks) * adapt_value

        return value

    def calculate_latent(self, context, observations_history, from_context=True):
        if from_context:
            latent = self.encode_context(context)
        else:
            latent = self.encode_history(observations_history)

        return latent

    def encode_context(self, context):
        latent = self.context_encoder(context)
        if self.context_encoder_clamp:
            latent = latent.clamp(min=self.context_encoder_clamp_value * -1, max=self.context_encoder_clamp_value)

        if self.context_encoder_stochastic:
            latent_noise = torch.randn_like(latent.detach()) * self.context_encoder_noise_std
            latent = latent + latent_noise

        return latent

    def encode_history(self, observations_history):
        latent = self.history_encoder(observations_history)
        if self.history_epinet:
            if self.history_epinet_input_type == "full":
                encoder_feature = self.input_flattener(observations_history).detach()
            elif self.history_epinet_input_type == "concat":
                full_feature = self.input_flattener(observations_history).detach()
                encoder_feature = self.history_encoder[:-1](observations_history).detach()
                encoder_feature = torch.cat([full_feature, encoder_feature], dim=-1)
            else:
                encoder_feature = self.history_encoder[:-1](observations_history).detach()

            encoder_feature_repeat = encoder_feature.unsqueeze(dim=0).repeat((self.history_epinet_num_samples, 1, 1))

            epinet_samples = torch.randn(
                (self.history_epinet_num_samples, self.history_epinet_dim), device=encoder_feature.device
            )
            epinet_samples_repeat = epinet_samples.unsqueeze(dim=1).repeat((1, encoder_feature.shape[0], 1))

            # shape: (epinet_samples, batch_size, total_epinet_input_size)  # noqa: E800
            epinet_input = torch.cat([encoder_feature_repeat, epinet_samples_repeat], dim=-1)

            # shape: (epinet_samples, batch_size, total_epinet_output_size)  # noqa: E800
            epinet_output_trainable = self.history_epinet_trainable(epinet_input)
            epinet_output_prior = self.history_epinet_prior(epinet_input)
            epinet_output_difference = epinet_output_trainable - epinet_output_prior

            # shape: (epinet_samples, batch_size, latent_size)  # noqa: E800
            if self.history_epinet_type == "mlp":
                epinet_term_all = epinet_output_difference
            elif self.history_epinet_type == "mlp_dot":
                epinet_output_difference = epinet_output_difference.reshape(
                    (epinet_output_difference.shape[0], epinet_output_difference.shape[1], -1, self.history_epinet_dim)
                )
                epinet_term_all = (epinet_output_difference * epinet_samples_repeat.unsqueeze(dim=-2)).sum(dim=-1)

            latent_all = latent + self.history_epinet_coef_active * epinet_term_all
            latent_mean = latent_all.mean(dim=0)

            if self.history_epinet_apply_bias:
                latent_std_l2 = latent_all.var(dim=0).sum(dim=-1)
                uncertainty = (latent_std_l2 - self.history_epinet_shift_param).clamp(min=0)
                alpha = torch.exp(self.history_epinet_scale_param * uncertainty * -1)

                if self.history_epinet_output_type == "mean":
                    latent = alpha.unsqueeze(dim=-1) * latent_mean
                else:
                    latent = alpha.unsqueeze(dim=-1) * latent

            else:
                if self.history_epinet_output_type == "mean":
                    latent = latent_mean

        return latent

    def encode_history_epinet(self, observations_history):
        latent = self.history_encoder(observations_history)
        if self.history_epinet:
            if self.history_epinet_input_type == "full":
                encoder_feature = self.input_flattener(observations_history).detach()
            elif self.history_epinet_input_type == "concat":
                full_feature = self.input_flattener(observations_history).detach()
                encoder_feature = self.history_encoder[:-1](observations_history).detach()
                encoder_feature = torch.cat([full_feature, encoder_feature], dim=-1)
            else:
                encoder_feature = self.history_encoder[:-1](observations_history).detach()

            encoder_feature_repeat = encoder_feature.unsqueeze(dim=0).repeat((self.history_epinet_num_samples, 1, 1))

            epinet_samples = torch.randn(
                (self.history_epinet_num_samples, self.history_epinet_dim), device=encoder_feature.device
            )
            epinet_samples_repeat = epinet_samples.unsqueeze(dim=1).repeat((1, encoder_feature.shape[0], 1))

            # shape: (epinet_samples, batch_size, total_epinet_input_size)  # noqa: E800
            epinet_input = torch.cat([encoder_feature_repeat, epinet_samples_repeat], dim=-1)

            # shape: (epinet_samples, batch_size, total_epinet_output_size)  # noqa: E800
            epinet_output_trainable = self.history_epinet_trainable(epinet_input)
            epinet_output_prior = self.history_epinet_prior(epinet_input)
            epinet_output_difference = epinet_output_trainable - epinet_output_prior

            # shape: (epinet_samples, batch_size, latent_size)  # noqa: E800
            if self.history_epinet_type == "mlp":
                epinet_term_all = epinet_output_difference
            elif self.history_epinet_type == "mlp_dot":
                epinet_output_difference = epinet_output_difference.reshape(
                    (epinet_output_difference.shape[0], epinet_output_difference.shape[1], -1, self.history_epinet_dim)
                )
                epinet_term_all = (epinet_output_difference * epinet_samples_repeat.unsqueeze(dim=-2)).sum(dim=-1)

            latent_all = latent + self.history_epinet_coef_active * epinet_term_all
        else:
            latent_all = latent

        return latent_all, latent, epinet_term_all

    def enforce_minimum_std(self):
        if self.use_log_std:
            self.log_std.data.clamp_(min=math.log(1e-2), max=math.log(10.0))
            if not self.robust_as_zeros:
                self.robust_log_std.data.clamp_(min=math.log(1e-2), max=math.log(10.0))
        else:
            self.std.data.clamp_(min=1e-2, max=10.0)
            if not self.robust_as_zeros:
                self.robust_std.data.clamp_(min=1e-2, max=10.0)

    def calculate_uncertainty_metric(self, obs_history):
        if self.history_epinet:
            latent_all, _, _ = self.encode_history_epinet(obs_history)
            metric = latent_all.var(dim=0).sum(dim=-1)

        else:
            metric = torch.zeros((obs_history.shape[0]), dtype=torch.float32, device=obs_history.device)

        return metric

    def calculate_epinet_metrics(self, obs_history):
        if self.history_epinet:
            latent_all, latent_base, _ = self.encode_history_epinet(obs_history)
            latent_mean = latent_all.mean(dim=0)

            latent_base_l2 = latent_base.pow(2).sum(dim=-1)
            latent_mean_l2 = latent_mean.pow(2).sum(dim=-1)
            latent_std_l2 = latent_all.var(dim=0).sum(dim=-1)

            latent_std_l2 = latent_all.var(dim=0).sum(dim=-1)
            uncertainty = (latent_std_l2 - self.history_epinet_shift_param).clamp(min=0)
            alpha = torch.exp(self.history_epinet_scale_param * uncertainty * -1)

        else:
            raise ValueError("can only be called when using history epinet")

        return latent_base_l2, latent_mean_l2, latent_std_l2, alpha

    def store_validation(self, validation_quantiles_value):
        self.validation_quantiles_value.data = validation_quantiles_value
        self.validation_quantiles_saved.data = torch.tensor(True, device=self.validation_quantiles_saved.device)

    def set_uncertainty_params(self):
        # shift / scale parameters for GRAM alpha calculation
        if self.history_epinet_finetune_shift and self.validation_quantiles_saved:
            self.history_epinet_shift_param = self.validation_quantiles_value.quantile(
                q=self.history_epinet_finetune_min_quantile
            )
        else:
            self.history_epinet_shift_param = torch.tensor(0.0, device=self.validation_quantiles_value.device)

        if self.history_epinet_finetune_scale and self.validation_quantiles_saved:
            latent_std_l2_quantile = self.validation_quantiles_value.quantile(
                q=self.history_epinet_finetune_max_quantile
            )
            self.history_epinet_scale_param = (
                math.log(self.history_epinet_finetune_target)
                * -1
                / (latent_std_l2_quantile - self.history_epinet_shift_param)
            )
        else:
            self.history_epinet_scale_param = torch.tensor(1.0, device=self.validation_quantiles_value.device)

        # Threshold for GRAM modular ablation
        self.uncertainty_max_threshold = self.validation_quantiles_value.quantile(
            q=self.history_epinet_finetune_max_quantile
        )

    def get_uncertainty_params(self):
        return self.history_epinet_shift_param, self.history_epinet_scale_param, self.uncertainty_max_threshold

    def calculate_uncertainty_mask(self, uncertainty_metric):
        uncertainty_mask_bool = uncertainty_metric > self.uncertainty_max_threshold
        uncertainty_mask_bool[torch.isnan(uncertainty_metric)] = False

        uncertainty_mask = uncertainty_mask_bool.to(dtype=torch.float32).unsqueeze(dim=-1)

        return uncertainty_mask
