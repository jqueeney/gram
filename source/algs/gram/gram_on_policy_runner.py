# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 ETH Zurich, NVIDIA CORPORATION
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRAM on-policy runner."""


from __future__ import annotations

import copy
import os
import statistics
import time
from collections import deque

import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from source.algs.gram.gram_actor_critic import GRAMActorCritic
from source.algs.gram.gram_adversary import GRAMAdversary
from source.algs.gram.gram_ppo import GRAMPPO


class GRAMOnPolicyRunner(OnPolicyRunner):
    """GRAM on-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # context
        context = self.env.get_context()
        num_context = context.shape[1]

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        num_adversary_actions = 1
        self.obs_history_length = self.policy_cfg["obs_history_length"]
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # GRAMActorCritic
        actor_critic: GRAMActorCritic = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, num_context, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # GRAMPPO
        self.alg: GRAMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=None).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=None).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization

        # settings
        self.robust_mask_update_every = self.cfg["robust_mask_update_every"]
        self.robust_mask_reset_update = self.cfg["robust_mask_reset_update"]

        self.collect_data_from_context = self.cfg["collect_data_from_context"]
        self.collect_data_deterministic = self.cfg["collect_data_deterministic"]
        self.collect_data_mix = self.cfg["collect_data_mix"]
        self.uncertainty_metric_full_only = self.cfg["uncertainty_metric_full_only"]
        self.uncertainty_metric_adapt_only = self.cfg["uncertainty_metric_adapt_only"]

        self.robust_train_only = self.cfg["robust_train_only"]
        self.adapt_train_only = self.cfg["adapt_train_only"]

        self.finetune_iterations = self.cfg["finetune_iterations"]
        self.finetune_fresh_start = self.cfg["finetune_fresh_start"]
        self.finetune_w_adversary = self.cfg["finetune_w_adversary"]

        self.print_stats = self.cfg["print_stats"]
        self.save_eval_detailed = self.cfg["save_eval_detailed"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            self.obs_history_length,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
            [num_adversary_actions],
        )

        # load extras into alg
        adversary_policy = GRAMAdversary(num_obs, num_adversary_actions, **self.policy_cfg).to(self.device)
        self.alg.init_extras(
            context,
            adversary_policy,
            env,
            self.cfg["train_and_finetune"],
            self.cfg["num_learning_epochs_finetune"],
            self.cfg["history_loss_actions"],
            self.cfg["history_loss_robust_as_zeros"],
            self.cfg["history_loss_adapt_only"],
            self.cfg["history_loss_epinet_separate"],
            self.cfg["history_lr"],
            self.cfg["adversary_update_every"],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = list()

        # Actor critic config for saving
        self.actor_critic_cfg = copy.deepcopy(self.policy_cfg)
        self.actor_critic_cfg["num_actor_obs"] = num_obs
        self.actor_critic_cfg["num_critic_obs"] = num_critic_obs
        self.actor_critic_cfg["num_actions"] = self.env.num_actions
        self.actor_critic_cfg["num_context"] = num_context

        # Hardware config for saving
        self.hardware_cfg = dict()
        self.hardware_cfg["joint_pos_clamp"] = self.cfg["joint_pos_clamp"]
        self.hardware_cfg["hip_scale_mult"] = self.cfg["hip_scale_mult"]

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)

        if self.finetune_fresh_start:
            pop_keys = []
            for key in loaded_dict["model_state_dict"].keys():
                if key.startswith("history_encoder") or key.startswith("history_epinet"):
                    pop_keys.append(key)

            for key in pop_keys:
                loaded_dict["model_state_dict"].pop(key)

        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        self.alg.actor_critic.set_uncertainty_params()

        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "alg_name": self.cfg["alg_name"],
            "hardware_cfg": self.hardware_cfg,
            "actor_critic_cfg": self.actor_critic_cfg,
            "empirical_normalization": self.empirical_normalization,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()

        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def _update_observation_history(self, obs_history, obs, reset_env_ids):
        if len(reset_env_ids) > 0:
            obs_history[reset_env_ids] = 0.0

        return torch.cat((obs_history[:, 1:], obs.unsqueeze(dim=1)), dim=1)

    def _update_robust_masks(self, robust_masks):
        robust_masks += 1.0
        robust_masks %= 2

        if self.robust_train_only:
            robust_masks[:] = 1.0
        elif self.adapt_train_only:
            robust_masks[:] = 0.0

        return robust_masks

    def _update_robust_masks_on_reset(self, robust_masks, reset_env_ids):
        if len(reset_env_ids) > 0:
            robust_masks[reset_env_ids] += 1.0
            robust_masks %= 2

        if self.robust_train_only:
            robust_masks[:] = 1.0
        elif self.adapt_train_only:
            robust_masks[:] = 0.0

        return robust_masks

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        obs = self.obs_normalizer(obs)
        if "critic" in extras["observations"]:
            critic_obs = self.critic_obs_normalizer(extras["observations"]["critic"])
        else:
            critic_obs = obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        obs_history = torch.zeros(
            (self.env.num_envs, self.obs_history_length, obs.shape[1]), dtype=torch.float32, device=self.device
        )
        reset_env_ids = torch.arange(self.env.num_envs, dtype=torch.int64, device=self.device)
        obs_history = self._update_observation_history(obs_history, obs, reset_env_ids)
        obs_history = obs_history.to(self.device)

        robust_mask_update_counter = 0
        robust_masks = torch.zeros((self.env.num_envs, 1), dtype=torch.float32, device=self.device)
        robust_masks[int(self.env.num_envs / 2) :] = 1

        # ----------------------------------------------------------------------
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations + self.finetune_iterations
        finetune_start_iter = tot_iter - self.finetune_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                if it == finetune_start_iter:
                    title_string = "BEGIN FINETUNE PHASE"
                    log_string = f"""{'#' * 80}\n""" f"""{title_string.center(80, ' ')}\n\n""" f"""{'#' * 80}\n"""
                    print(log_string)

                    self.alg.actor_critic.act_from_context = False

                    # turn off adversary
                    if not self.finetune_w_adversary:
                        self.alg.adversary_policy.adversary_prob = 0.0
                        adversary_actions = self.alg.adversary_policy.act(obs).detach()
                        adversary_masks = self.alg.adversary_policy.generate_adversary_masks(obs).detach()
                        self.alg.adversary_update_sim(adversary_actions, torch.zeros_like(adversary_masks))

                    # switch normalizers to eval mode
                    if self.empirical_normalization:
                        self.obs_normalizer.eval()
                        self.critic_obs_normalizer.eval()

                    # save pre-finetune checkpoint
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

                if (robust_mask_update_counter % self.robust_mask_update_every == 0) and (
                    not self.robust_mask_reset_update
                ):
                    robust_masks = self._update_robust_masks(robust_masks)
                    robust_masks = robust_masks.to(self.device)
                robust_mask_update_counter += 1

                for i in range(self.num_steps_per_env):
                    if self.robust_mask_reset_update:
                        robust_masks = self._update_robust_masks_on_reset(robust_masks, reset_env_ids)
                        robust_masks = robust_masks.to(self.device)

                    actions = self.alg.act(obs, critic_obs, obs_history, robust_masks)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs

                    reset_env_ids = self.env.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    obs_history = self._update_observation_history(obs_history, obs, reset_env_ids)
                    obs_history = obs_history.to(self.device)

                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, obs_history, robust_masks)

            if it >= finetune_start_iter:
                loss_stats, update_stats = self.alg.finetune()
            else:
                loss_stats, update_stats = self.alg.update()
                self.alg.adversary_policy.update_magnitude((it - start_iter + 1) / (finetune_start_iter - start_iter))
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Calculate and save validation metrics
        self.collect_validation_data()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.get_std().mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["loss_stats"]["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["loss_stats"]["mean_surrogate_loss"], locs["it"])
        if "mean_history_loss" in locs["loss_stats"]:
            self.writer.add_scalar("Loss/history", locs["loss_stats"]["mean_history_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        for key, value in locs["update_stats"].items():
            self.writer.add_scalar("Update/%s" % key, value, locs["it"])

        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs["loss_stats"]['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["loss_stats"]['mean_surrogate_loss']:.4f}\n"""
                f"""{'History loss:':>{pad}} {locs["loss_stats"].get('mean_history_loss',0.0):.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs["loss_stats"]['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["loss_stats"]['mean_surrogate_loss']:.4f}\n"""
                f"""{'History loss:':>{pad}} {locs["loss_stats"].get('mean_history_loss',0.0):.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def calculate_and_save_validation_metrics(self):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        # Calculate and save validation metrics
        self.collect_validation_data()
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def collect_validation_data(self):
        title_string = "Collecting validation data"
        log_string = f"""{'#' * 80}\n""" f"""{title_string.center(80, ' ')}\n\n"""
        print(log_string)

        self.eval_mode()

        # turn off adversary
        obs, _ = self.env.get_observations()
        obs = self.obs_normalizer(obs)
        self.alg.adversary_policy.adversary_prob = 0.0
        adversary_actions = self.alg.adversary_policy.act(obs).detach()
        adversary_masks = self.alg.adversary_policy.generate_adversary_masks(obs).detach()
        self.alg.adversary_update_sim(adversary_actions, torch.zeros_like(adversary_masks))

        if self.collect_data_from_context:
            self.alg.actor_critic.act_from_context = True
        else:
            self.alg.actor_critic.act_from_context = False

        with torch.inference_mode():

            uncertainty_metrics = torch.zeros(
                (2, self.env.num_envs, self.env.unwrapped.max_episode_length), device=self.device
            )
            alive_mask = torch.zeros((2, self.env.num_envs, self.env.unwrapped.max_episode_length), device=self.device)

            for robust_mask_value in [0, 1]:

                alive = torch.ones(self.env.num_envs, device=self.device)

                obs, extras = self.env.reset()
                obs = self.obs_normalizer(obs)
                if "critic" in extras["observations"]:
                    critic_obs = self.critic_obs_normalizer(extras["observations"]["critic"])
                else:
                    critic_obs = obs
                obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

                obs_history = torch.zeros(
                    (self.env.num_envs, self.obs_history_length, obs.shape[1]), dtype=torch.float32, device=self.device
                )
                reset_env_ids = torch.arange(self.env.num_envs, dtype=torch.int64, device=self.device)
                obs_history = self._update_observation_history(obs_history, obs, reset_env_ids)
                obs_history = obs_history.to(self.device)

                if self.collect_data_mix:
                    robust_masks = torch.zeros((self.env.num_envs, 1), dtype=torch.float32, device=self.device)
                    robust_masks[int(self.env.num_envs / 2) :] = 1
                else:
                    robust_masks = (
                        torch.ones((self.env.num_envs, 1), dtype=torch.float32, device=self.device) * robust_mask_value
                    )
                robust_masks = robust_masks.to(self.device)

                # ---------------------------------------------------------------- #
                for t in range(self.env.unwrapped.max_episode_length):
                    if self.collect_data_mix and (t % self.num_steps_per_env == 0):
                        robust_masks = self._update_robust_masks(robust_masks)
                        robust_masks = robust_masks.to(self.device)

                    actions = self.alg.actor_critic.act_inference_from_mask(
                        obs, self.alg.context, obs_history, robust_masks, deterministic=self.collect_data_deterministic
                    )
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)

                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs

                    reset_env_ids = self.env.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    obs_history = self._update_observation_history(obs_history, obs, reset_env_ids)
                    obs_history = obs_history.to(self.device)
                    # ----------------------------------------------------------
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    uncertainty_metric_step = self.alg.actor_critic.calculate_uncertainty_metric(obs_history)
                    uncertainty_metrics[robust_mask_value, :, t] = uncertainty_metric_step

                    alive *= 1 - dones
                    alive_mask[robust_mask_value, :, t] = alive

            alive_traj_mask = alive_mask.sum(dim=-1) == (self.env.unwrapped.max_episode_length - 1)

        # Calculate quantile values
        if self.uncertainty_metric_adapt_only:
            alive_mask = alive_mask[0]
            alive_mask_traj = alive_traj_mask[0]
            uncertainty_metrics = uncertainty_metrics[0]
        else:
            alive_mask = alive_mask.flatten(start_dim=0, end_dim=1)
            alive_mask_traj = alive_traj_mask.flatten(start_dim=0, end_dim=1)
            uncertainty_metrics = uncertainty_metrics.flatten(start_dim=0, end_dim=1)

        if self.uncertainty_metric_full_only:
            uncertainty_metrics = uncertainty_metrics[alive_mask_traj]
            alive_mask = alive_mask[alive_mask_traj]

        uncertainty_metrics = uncertainty_metrics[alive_mask == 1]
        validation_quantiles_value = uncertainty_metrics.quantile(q=self.alg.actor_critic.validation_quantiles_q)

        # Save quantile values
        self.alg.actor_critic.store_validation(validation_quantiles_value)

        if self.print_stats:
            print("# ------------------------------------------------------------- #")
            print("  Validation: Uncertainty Metric")
            print("# ------------------------------------------------------------- #")
            for idx in range(self.alg.actor_critic.validation_quantiles_q.shape[0]):
                q = self.alg.actor_critic.validation_quantiles_q[idx]
                v = validation_quantiles_value[idx]
                print("  Q %3.0f:  %8.4f" % (q * 100, v))
            print("# ------------------------------------------------------------- #")

    def evaluate(self, eval_inputs_dict, deterministic=False):
        self.eval_mode()  # switch to evaluation mode (dropout for example)

        # statistics
        alive = torch.ones(self.env.num_envs, device=self.device)
        total_rewards = torch.zeros(self.env.num_envs, device=self.device)
        total_steps = torch.zeros(self.env.num_envs, device=self.device)

        alive_mask = torch.zeros((self.env.num_envs, self.env.unwrapped.max_episode_length), device=self.device)

        if self.alg.actor_critic.history_epinet:
            epinet_base_l2 = torch.zeros((self.env.num_envs, self.env.unwrapped.max_episode_length), device=self.device)
            epinet_mean_l2 = torch.zeros((self.env.num_envs, self.env.unwrapped.max_episode_length), device=self.device)
            epinet_std_l2 = torch.zeros((self.env.num_envs, self.env.unwrapped.max_episode_length), device=self.device)
            alpha = torch.zeros((self.env.num_envs, self.env.unwrapped.max_episode_length), device=self.device)

        # simulate environment
        obs, extras = self.env.reset()
        obs = self.obs_normalizer(obs)
        obs = obs.to(self.device)

        obs_history = torch.zeros(
            (self.env.num_envs, self.obs_history_length, obs.shape[1]), dtype=torch.float32, device=self.device
        )
        reset_env_ids = torch.arange(self.env.num_envs, dtype=torch.int64, device=self.device)
        obs_history = self._update_observation_history(obs_history, obs, reset_env_ids)

        uncertainty_metric_step = self.alg.actor_critic.calculate_uncertainty_metric(obs_history)

        for t in range(self.env.unwrapped.max_episode_length):
            # run everything in inference mode
            with torch.inference_mode():
                alive_mask[:, t] = alive
                if self.alg.actor_critic.history_epinet:
                    epinet_base_l2_step, epinet_mean_l2_step, epinet_std_l2_step, alpha_step = (
                        self.alg.actor_critic.calculate_epinet_metrics(obs_history)
                    )
                    epinet_base_l2[:, t] = epinet_base_l2_step
                    epinet_mean_l2[:, t] = epinet_mean_l2_step
                    epinet_std_l2[:, t] = epinet_std_l2_step
                    alpha[:, t] = alpha_step

                # agent stepping
                actions = self.alg.actor_critic.act_inference(
                    obs, self.alg.context, obs_history, uncertainty_metric_step, deterministic=deterministic
                )
                # env stepping
                obs, rewards, dones, infos = self.env.step(actions)
                obs = self.obs_normalizer(obs)
                obs = obs.to(self.device)

                reset_env_ids = self.env.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)

                obs_history = self._update_observation_history(obs_history, obs, reset_env_ids)

                uncertainty_metric_step = self.alg.actor_critic.calculate_uncertainty_metric(obs_history)

                # accumulate statistics
                total_rewards += rewards * alive
                total_steps += alive
                alive *= 1 - dones

        # organize evaluation information
        eval_log = {
            "params": eval_inputs_dict,
            "returns": total_rewards.numpy(force=True),
            "episode_length": total_steps.numpy(force=True),
        }

        if self.alg.actor_critic.history_epinet:
            alpha_ave = torch.sum(alpha * alive_mask, dim=-1) / torch.sum(alive_mask, dim=-1)
            eval_log["alpha"] = alpha_ave.numpy(force=True)

        if self.save_eval_detailed:
            eval_log["alive/alive_mask"] = alive_mask.numpy(force=True)
            if self.alg.actor_critic.history_epinet:
                epinet_base_l2[alive_mask == 0] = torch.nan
                epinet_mean_l2[alive_mask == 0] = torch.nan
                epinet_std_l2[alive_mask == 0] = torch.nan
                alpha[alive_mask == 0] = torch.nan

                eval_log["epinet/base_l2"] = epinet_base_l2.numpy(force=True)
                eval_log["epinet/mean_l2"] = epinet_mean_l2.numpy(force=True)
                eval_log["epinet/std_l2"] = epinet_std_l2.numpy(force=True)
                eval_log["epinet/alpha"] = alpha.numpy(force=True)

                alpha_shift, alpha_scale, alpha_max_threshold = self.alg.actor_critic.get_uncertainty_params()
                eval_log["epinet/alpha_shift"] = alpha_shift.numpy(force=True)
                eval_log["epinet/alpha_scale"] = alpha_scale.numpy(force=True)
                eval_log["epinet/alpha_max_threshold"] = alpha_max_threshold.numpy(force=True)

        if self.print_stats:
            print("*****************************************")
            print("Parameters:")
            print("-----------")
            for key, value in eval_inputs_dict.items():
                print("%s:  %s" % (key, value))
            print("*****************************************")
            print("Summary:")
            print("--------")
            print(
                "Total rewards: %.1f (%.1f) [%.1f, %.1f]"
                % (
                    torch.mean(total_rewards),
                    torch.std(total_rewards),
                    torch.amin(total_rewards),
                    torch.amax(total_rewards),
                )
            )
            print(
                "Total steps:   %.0f (%.0f) [%.0f, %.0f]"
                % (torch.mean(total_steps), torch.std(total_steps), torch.amin(total_steps), torch.amax(total_steps))
            )

            if self.alg.actor_critic.history_epinet:
                print("*****************************************")
                print("Alpha:")
                print("------")
                print("Mean: %8.4f" % alpha[alive_mask == 1].mean())
                print("Min:  %8.4f" % alpha[alive_mask == 1].amin())
                print("Q01:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.01))
                print("Q05:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.05))
                print("Q10:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.10))
                print("Q20:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.20))
                print("Q30:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.30))
                print("Q40:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.40))
                print("Q50:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.50))
                print("Q60:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.60))
                print("Q70:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.70))
                print("Q80:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.80))
                print("Q90:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.90))
                print("Q95:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.95))
                print("Q99:  %8.4f" % alpha[alive_mask == 1].quantile(q=0.99))
                print("Max:  %8.4f" % alpha[alive_mask == 1].amax())
                print("*****************************************")
                print("Uncertainty Metric:")
                print("-------------------")
                print("Mean: %8.4f" % epinet_std_l2[alive_mask == 1].mean())
                print("Min:  %8.4f" % epinet_std_l2[alive_mask == 1].amin())
                print("Q01:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.01))
                print("Q05:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.05))
                print("Q10:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.10))
                print("Q20:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.20))
                print("Q30:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.30))
                print("Q40:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.40))
                print("Q50:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.50))
                print("Q60:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.60))
                print("Q70:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.70))
                print("Q80:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.80))
                print("Q90:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.90))
                print("Q95:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.95))
                print("Q99:  %8.4f" % epinet_std_l2[alive_mask == 1].quantile(q=0.99))
                print("Max:  %8.4f" % epinet_std_l2[alive_mask == 1].amax())

        print("*****************************************")
        print("Evaluation complete.")
        print("*****************************************")

        return eval_log
