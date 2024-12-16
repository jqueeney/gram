# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022-2024, The Isaac Lab Project Developers
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command line utilities."""


from __future__ import annotations

import argparse


def create_parser():
    """Create parser for training / evaluation."""
    # create parser
    parser = argparse.ArgumentParser(description="Command line parser.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=10000, help="RL training iterations.")

    # -- experiment arguments
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    parser.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    parser.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )

    # ---------------------- #
    # -- custom arguments -- #
    # ---------------------- #

    # -- task arguments
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Custom-Unitree-Go2-v0", help="Name of the task.")
    parser.add_argument(
        "--id_context", help="ID context set", type=str, default="base_id", choices=["base_id", "base_id_frozen_joints"]
    )

    # -- evaluation arguments
    parser.add_argument(
        "--stochastic_eval", action="store_true", default=False, help="use stochastic policy for evaluation"
    )
    parser.add_argument("--save_eval_detailed", help="save detailed evaluation statistics", action="store_true")
    parser.add_argument("--print_stats", help="print statistics to console", action="store_true")

    # -- finetune arguments (supervised learning phase)
    parser.add_argument(
        "--calculate_validation_metrics_only", help="only calculate validation metrics", action="store_true"
    )

    parser.add_argument(
        "--finetune_iterations", type=int, default=5000, help="number of finetune iterations for GRAM alg"
    )
    parser.add_argument("--train_and_finetune", help="train and finetune encoder", action="store_true")
    parser.add_argument("--finetune_fresh_start", help="ignore saved history encoder", action="store_true")
    parser.add_argument("--finetune_w_adversary", help="include adversary during finetuning", action="store_true")
    parser.add_argument(
        "--num_learning_epochs_finetune",
        type=int,
        help="number of learning epochs per update for finetuning (if different from train)",
    )

    # -- NN sizes
    parser.add_argument("--actor_hidden_dims", nargs="+", help="list of hidden layer sizes for actor", type=int)
    parser.add_argument("--critic_hidden_dims", nargs="+", help="list of hidden layer sizes for critic", type=int)
    parser.add_argument(
        "--context_hidden_dims", nargs="+", help="list of hidden layer sizes for context encoder", type=int
    )
    parser.add_argument("--adversary_hidden_dims", nargs="+", help="list of hidden layer sizes for adversary", type=int)
    parser.add_argument(
        "--history_encoder_hidden_dims", nargs="+", help="list of hidden layer sizes for history encoder", type=int
    )
    parser.add_argument(
        "--history_epinet_hidden_dims", nargs="+", help="list of hidden layer sizes for history epinet", type=int
    )

    # -- NN initializations
    parser.add_argument("--no_init_nn_weights", help="do not initialize NN weights", action="store_true")
    parser.add_argument("--use_std_direct", help="use std parameter directly instead of log_std", action="store_true")
    parser.add_argument("--actor_gain", help="gain for actor initialization", type=float, default=1e-4)
    parser.add_argument("--critic_gain", help="gain for critic initialization", type=float, default=1e-4)
    parser.add_argument("--context_gain", help="gain for context initialization", type=float, default=1e-4)
    parser.add_argument("--adversary_gain", help="gain for adversary initialization", type=float, default=1e-4)
    parser.add_argument("--history_gain", help="gain for history encoder initialization", type=float, default=1e-4)
    parser.add_argument(
        "--history_epinet_gain", help="gain for history epinet initialization", type=float, default=1e-4
    )
    parser.add_argument(
        "--history_epinet_prior_gain", help="gain for history epinet prior initialization", type=float, default=1.0
    )

    # -- training arguments
    parser.add_argument(
        "--alg_name",
        help="algorithm name",
        type=str,
        default="gram",
        choices=[
            "gram",
            "gram_separate",
            "gram_modular",
            "robust_rl",
            "contextual_rl",
            "contextual_rl_noise",
            "domain_rand",
            "domain_rand_priv",
        ],
    )

    parser.add_argument("--no_empirical_normalization", help="do not use empirical normalization", action="store_true")
    parser.add_argument("--num_learning_epochs", type=int, help="number of learning epochs per update")
    parser.add_argument("--num_mini_batches", type=int, help="number of minibatches per update")

    parser.add_argument(
        "--robust_mask_update_every", type=int, default=1, help="how often to update robust masks during training"
    )

    # -- adversary
    parser.add_argument("--adversary_magnitude", help="max adversary magnitude", type=float, default=1.0)
    parser.add_argument("--adversary_clamp", help="adversary action clamping", type=float, default=5.0)
    parser.add_argument(
        "--no_adversary_random_scale",
        help="do not apply random scale in [0,M_k] to adversary actions",
        action="store_true",
    )
    parser.add_argument(
        "--adversary_random_angle", help="apply random angle for adversary actions", action="store_true"
    )
    parser.add_argument(
        "--adversary_max_scale_start", help="starting max scale for adversary random scale", type=float, default=0.0
    )
    parser.add_argument("--adversary_prob", help="adversary probability", type=float, default=0.05)
    parser.add_argument("--adversary_update_every", type=int, default=10, help="how often to update adversary")
    parser.add_argument("--adversary_init_noise_std", help="adversary initial noise std", type=float, default=1.0)

    # -- context / history encoder
    parser.add_argument("--robust_use_latent_actor", help="use latent input for robust policy", action="store_true")
    parser.add_argument("--robust_use_latent_critic", help="use latent input for robust critic", action="store_true")

    parser.add_argument("--num_latent", type=int, default=8, help="number of latent dimensions for context encoding")
    parser.add_argument("--latent_detach_actor", help="detach latent in actor", action="store_true")
    parser.add_argument("--latent_detach_critic", help="detach latent in critic", action="store_true")
    parser.add_argument("--no_context_critic", help="no context in critic", action="store_true")

    parser.add_argument("--obs_history_length", type=int, default=16, help="observation history length")
    parser.add_argument("--context_encoder_clamp", help="clamp output of context encoder", action="store_true")
    parser.add_argument(
        "--context_encoder_clamp_value", help="value for context encoder clamping", type=float, default=1.0
    )
    parser.add_argument("--context_encoder_stochastic", help="use stochastic context encoder", action="store_true")
    parser.add_argument(
        "--context_encoder_noise_std", help="stochastic context encoder noise std", type=float, default=0.25
    )

    parser.add_argument(
        "--history_epinet_output_type",
        help="output of epinet used for ID contexts",
        type=str,
        default="base",
        choices=["base", "mean"],
    )
    parser.add_argument(
        "--no_history_epinet_finetune_scale", help="do not finetune epinet scale parameter", action="store_true"
    )
    parser.add_argument(
        "--no_history_epinet_finetune_shift", help="do not finetune epinet shift parameter", action="store_true"
    )
    parser.add_argument(
        "--history_epinet_finetune_min_quantile", help="min quantile for epinet finetuning", type=float, default=0.90
    )
    parser.add_argument(
        "--history_epinet_finetune_max_quantile", help="max quantile for epinet finetuning", type=float, default=0.99
    )
    parser.add_argument(
        "--history_epinet_finetune_target",
        help="target coefficient value at selected quantile",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--history_epinet_type", help="history epinet type", type=str, default="mlp_dot", choices=["mlp", "mlp_dot"]
    )
    parser.add_argument(
        "--history_epinet_input_type",
        help="history epinet input type",
        type=str,
        default="concat",
        choices=["full", "concat", "intermediate"],
    )

    parser.add_argument(
        "--history_epinet_dim", type=int, default=8, help="number of dimensions for epinet random input variable"
    )
    parser.add_argument(
        "--history_epinet_num_samples", type=int, default=8, help="number of samples of epinet variable"
    )
    parser.add_argument("--history_epinet_coef", help="history epinet coefficient", type=float, default=1.0)
    parser.add_argument("--history_loss_actions", help="include actions in history loss", action="store_true")
    parser.add_argument(
        "--history_loss_robust_as_zeros",
        help="latent target as zeros in history loss for robust data",
        action="store_true",
    )
    parser.add_argument("--history_loss_adapt_only", help="exclude robust data from history loss", action="store_true")
    parser.add_argument(
        "--history_loss_include_epinet", help="include epinet in history loss calculation", action="store_true"
    )
    parser.add_argument("--history_lr", help="history learning rate", type=float, default=1e-3)

    # -- validation
    parser.add_argument(
        "--collect_data_from_context", help="use context encoder for validation data collection", action="store_true"
    )
    parser.add_argument(
        "--collect_data_stochastic", help="use stochastic policy for validation data collection", action="store_true"
    )

    parser.add_argument(
        "--uncertainty_metric_full_only",
        help="only use full trajectories to calculate uncertainty metrics",
        action="store_true",
    )
    parser.add_argument(
        "--uncertainty_metric_adapt_only",
        help="only use adaptive trajectories to calculate uncertainty metrics",
        action="store_true",
    )

    # -- env context

    # terrain
    parser.add_argument("--terrain_roughness_cm", help="terrain roughness (cm)", type=float)
    parser.add_argument("--enable_height_scan", help="enable height scan", action="store_true")

    # frozen joints
    parser.add_argument(
        "--frozen_joint_type",
        help="frozen joint type for evaluation",
        type=str,
        default="none",
        choices=["none", "hip", "thigh", "calf"],
    )

    # base mass
    parser.add_argument("--base_mass_min", help="min additive base mass", type=float)
    parser.add_argument("--base_mass_max", help="max additive base mass", type=float)

    parser.add_argument("--no_base_mass_context_scale", help="do not scale base mass in context", action="store_true")
    parser.add_argument("--base_mass_context_scale_min", help="min additive base mass for context scaling", type=float)
    parser.add_argument("--base_mass_context_scale_max", help="max additive base mass for context scaling", type=float)

    # friction
    parser.add_argument("--friction_mult_min", help="min multiplicative friction", type=float)
    parser.add_argument("--friction_mult_max", help="max multiplicative friction", type=float)

    parser.add_argument(
        "--no_friction_mult_context_scale", help="do not scale friction in context", action="store_true"
    )
    parser.add_argument("--friction_mult_context_scale_min", help="min friction mult for context scaling", type=float)
    parser.add_argument("--friction_mult_context_scale_max", help="max friction mult for context scaling", type=float)

    # motor strength
    parser.add_argument("--motor_strength_mult_min", help="motor strength multiple min", type=float)
    parser.add_argument("--motor_strength_mult_max", help="motor strength multiple max", type=float)
    parser.add_argument("--motor_fault_indices", nargs="+", action="append", help="motor fault indices", type=int)

    parser.add_argument(
        "--no_motor_strength_mult_context_scale",
        help="do not scale motor strength multiple in context",
        action="store_true",
    )
    parser.add_argument(
        "--motor_strength_mult_context_scale_min", help="min motor strength multiple for context scaling", type=float
    )
    parser.add_argument(
        "--motor_strength_mult_context_scale_max", help="max motor strength multiple for context scaling", type=float
    )

    # action scale
    parser.add_argument("--action_mult_indices", nargs="+", action="append", help="action mult indices", type=int)
    parser.add_argument("--action_mult_min", help="action mult min for active indices", type=float)
    parser.add_argument("--action_mult_max", help="action mult max for active indices", type=float)
    parser.add_argument("--action_mult_count", help="action mult count (discrete choices)", type=int)

    parser.add_argument(
        "--no_action_mult_context_scale", help="do not scale action mult in context", action="store_true"
    )
    parser.add_argument("--action_mult_context_scale_min", help="min action mult for context scaling", type=float)
    parser.add_argument("--action_mult_context_scale_max", help="max action mult for context scaling", type=float)

    # joint angle bias
    parser.add_argument("--joint_bias_min", help="joint angle bias min", type=float)
    parser.add_argument("--joint_bias_max", help="joint angle bias max", type=float)

    parser.add_argument("--no_joint_bias_context_scale", help="do not scale joint bias in context", action="store_true")
    parser.add_argument("--joint_bias_context_scale_min", help="min joint bias for context scaling", type=float)
    parser.add_argument("--joint_bias_context_scale_max", help="max joint bias for context scaling", type=float)

    # task
    parser.add_argument("--target_x_vel_min", help="min target x velocity", type=float)
    parser.add_argument("--target_x_vel_max", help="max target x velocity", type=float)

    parser.add_argument("--target_y_vel_min", help="min target y velocity", type=float)
    parser.add_argument("--target_y_vel_max", help="max target y velocity", type=float)

    parser.add_argument("--disable_self_collisions", help="disable self collisions", action="store_true")
    parser.add_argument("--disable_obs_noise", help="disable observation noise", action="store_true")
    parser.add_argument(
        "--no_terminate_base_height", help="exclude termination for low base height", action="store_true"
    )
    parser.add_argument("--base_height_terminate_thresh", help="base height termination threshold", type=float)

    return parser


def parse_custom_inputs_cfg(task_name: str, id_context: str, args_cli: argparse.Namespace, eval_mode: bool):
    """Parse configuration based on inputs."""
    from source.envs import get_custom_inputs_cfg

    # load the default configuration
    inputs_cfg = get_custom_inputs_cfg(task_name, id_context)

    # eval_mode
    inputs_cfg.eval_mode = eval_mode

    # terrain roughness
    if args_cli.terrain_roughness_cm is not None:
        inputs_cfg.terrain_roughness_cm = args_cli.terrain_roughness_cm
    if args_cli.enable_height_scan:
        inputs_cfg.height_scan = True

    # frozen joints
    if eval_mode:
        if args_cli.frozen_joint_type == "none":
            inputs_cfg.action_mult_indices = None
        elif args_cli.frozen_joint_type == "hip":
            inputs_cfg.action_mult_indices = [[0], [1], [2], [3]]
        elif args_cli.frozen_joint_type == "thigh":
            inputs_cfg.action_mult_indices = [[4], [5], [6], [7]]
        elif args_cli.frozen_joint_type == "calf":
            inputs_cfg.action_mult_indices = [[8], [9], [10], [11]]

    # base mass
    if args_cli.base_mass_min is not None:
        inputs_cfg.base_mass_min = args_cli.base_mass_min
    if args_cli.base_mass_max is not None:
        inputs_cfg.base_mass_max = args_cli.base_mass_max

    if args_cli.no_base_mass_context_scale:
        inputs_cfg.base_mass_context_scale = False
    if args_cli.base_mass_context_scale_min is not None:
        inputs_cfg.base_mass_context_scale_min = args_cli.base_mass_context_scale_min
    if args_cli.base_mass_context_scale_max is not None:
        inputs_cfg.base_mass_context_scale_max = args_cli.base_mass_context_scale_max

    # friction
    if args_cli.friction_mult_min is not None:
        inputs_cfg.friction_mult_min = args_cli.friction_mult_min
    if args_cli.friction_mult_max is not None:
        inputs_cfg.friction_mult_max = args_cli.friction_mult_max

    if args_cli.no_friction_mult_context_scale:
        inputs_cfg.friction_mult_context_scale = False
    if args_cli.friction_mult_context_scale_min is not None:
        inputs_cfg.friction_mult_context_scale_min = args_cli.friction_mult_context_scale_min
    if args_cli.friction_mult_context_scale_max is not None:
        inputs_cfg.friction_mult_context_scale_max = args_cli.friction_mult_context_scale_max

    # motor strength
    if args_cli.motor_strength_mult_min is not None:
        inputs_cfg.motor_strength_mult_min = args_cli.motor_strength_mult_min
    if args_cli.motor_strength_mult_max is not None:
        inputs_cfg.motor_strength_mult_max = args_cli.motor_strength_mult_max
    if args_cli.motor_fault_indices is not None:
        inputs_cfg.motor_fault_indices = args_cli.motor_fault_indices

    if args_cli.no_motor_strength_mult_context_scale:
        inputs_cfg.motor_strength_mult_context_scale = False
    if args_cli.motor_strength_mult_context_scale_min is not None:
        inputs_cfg.motor_strength_mult_context_scale_min = args_cli.motor_strength_mult_context_scale_min
    if args_cli.motor_strength_mult_context_scale_max is not None:
        inputs_cfg.motor_strength_mult_context_scale_max = args_cli.motor_strength_mult_context_scale_max

    # action mult
    if args_cli.action_mult_indices is not None:
        inputs_cfg.action_mult_indices = args_cli.action_mult_indices
    if args_cli.action_mult_min is not None:
        inputs_cfg.action_mult_min = args_cli.action_mult_min
    if args_cli.action_mult_max is not None:
        inputs_cfg.action_mult_max = args_cli.action_mult_max
    if args_cli.action_mult_count is not None:
        inputs_cfg.action_mult_count = args_cli.action_mult_count

    if args_cli.no_action_mult_context_scale:
        inputs_cfg.action_mult_context_scale = False
    if args_cli.action_mult_context_scale_min is not None:
        inputs_cfg.action_mult_context_scale_min = args_cli.action_mult_context_scale_min
    if args_cli.action_mult_context_scale_max is not None:
        inputs_cfg.action_mult_context_scale_max = args_cli.action_mult_context_scale_max

    # joint bias
    if args_cli.joint_bias_min is not None:
        inputs_cfg.joint_bias_min = args_cli.joint_bias_min
    if args_cli.joint_bias_max is not None:
        inputs_cfg.joint_bias_max = args_cli.joint_bias_max

    if args_cli.no_joint_bias_context_scale:
        inputs_cfg.joint_bias_context_scale = False
    if args_cli.joint_bias_context_scale_min is not None:
        inputs_cfg.joint_bias_context_scale_min = args_cli.joint_bias_context_scale_min
    if args_cli.joint_bias_context_scale_max is not None:
        inputs_cfg.joint_bias_context_scale_max = args_cli.joint_bias_context_scale_max

    # adversary
    inputs_cfg = set_adversary_magnitude_by_alg(inputs_cfg, args_cli, eval_mode)

    # task
    if args_cli.target_x_vel_min is not None:
        inputs_cfg.target_x_vel_min = args_cli.target_x_vel_min
    if args_cli.target_x_vel_max is not None:
        inputs_cfg.target_x_vel_max = args_cli.target_x_vel_max

    if args_cli.target_y_vel_min is not None:
        inputs_cfg.target_y_vel_min = args_cli.target_y_vel_min
    if args_cli.target_y_vel_max is not None:
        inputs_cfg.target_y_vel_max = args_cli.target_y_vel_max

    if args_cli.disable_self_collisions:
        inputs_cfg.activate_self_collisions = False
    if args_cli.disable_obs_noise:
        inputs_cfg.disable_obs_noise = True
    if args_cli.no_terminate_base_height:
        inputs_cfg.terminate_base_height = False
    if args_cli.base_height_terminate_thresh is not None:
        inputs_cfg.base_height_terminate_thresh = args_cli.base_height_terminate_thresh

    return inputs_cfg


def set_adversary_magnitude_by_alg(inputs_cfg, args_cli, eval_mode):
    """Update adversary magnitude based on algorithm."""
    if eval_mode:
        inputs_cfg.adversary_magnitude = 0.0
    else:
        if args_cli.alg_name.startswith("contextual_rl") or args_cli.alg_name.startswith("domain_rand"):
            inputs_cfg.adversary_magnitude = 0.0
        else:
            inputs_cfg.adversary_magnitude = args_cli.adversary_magnitude

    return inputs_cfg


def update_agent_cfg_by_alg(agent_cfg, args_cli, eval_mode):
    """Update agent config based on algorithm."""
    agent_cfg.adapt_train_only = False
    agent_cfg.robust_train_only = False
    agent_cfg.policy.act_inference_robust = False
    agent_cfg.policy.act_inference_adapt = False
    agent_cfg.policy.history_epinet_apply_bias = False
    agent_cfg.policy.robust_as_zeros = False
    agent_cfg.policy.history_epinet = False
    agent_cfg.robust_mask_reset_update = False
    agent_cfg.collect_data_mix = True
    agent_cfg.finetune_iterations = args_cli.finetune_iterations
    agent_cfg.policy.adversary_prob = args_cli.adversary_prob
    agent_cfg.policy.robust_use_latent_actor = args_cli.robust_use_latent_actor
    agent_cfg.policy.robust_use_latent_critic = args_cli.robust_use_latent_critic
    agent_cfg.policy.context_encoder_stochastic = args_cli.context_encoder_stochastic

    if eval_mode:
        agent_cfg.policy.act_from_context = False
        agent_cfg.policy.adversary_prob = 0.0
    else:
        agent_cfg.policy.act_from_context = True
        agent_cfg.policy.adversary_prob = args_cli.adversary_prob

    if args_cli.alg_name == "gram":
        agent_cfg.policy.robust_as_zeros = True
        agent_cfg.policy.history_epinet = True

        if eval_mode:
            agent_cfg.policy.act_inference_adapt = True
            agent_cfg.policy.history_epinet_apply_bias = True

    elif args_cli.alg_name == "gram_separate":
        agent_cfg.policy.robust_as_zeros = True
        agent_cfg.policy.history_epinet = True
        agent_cfg.robust_mask_reset_update = True
        agent_cfg.collect_data_mix = False

        if eval_mode:
            agent_cfg.policy.act_inference_adapt = True
            agent_cfg.policy.history_epinet_apply_bias = True

    elif args_cli.alg_name == "gram_modular":
        agent_cfg.policy.history_epinet = True
        agent_cfg.robust_mask_reset_update = True
        agent_cfg.collect_data_mix = False

    elif args_cli.alg_name == "robust_rl":
        if not agent_cfg.policy.robust_use_latent_actor:
            agent_cfg.finetune_iterations = 0

        if eval_mode:
            agent_cfg.policy.act_inference_robust = True
        else:
            agent_cfg.robust_train_only = True

    elif args_cli.alg_name.startswith("contextual_rl"):
        agent_cfg.policy.adversary_prob = 0.0

        if eval_mode:
            agent_cfg.policy.act_inference_adapt = True
        else:
            agent_cfg.adapt_train_only = True

        if args_cli.alg_name == "contextual_rl_noise":
            agent_cfg.policy.context_encoder_stochastic = True

    elif args_cli.alg_name.startswith("domain_rand"):
        agent_cfg.policy.adversary_prob = 0.0
        if not agent_cfg.policy.robust_use_latent_actor:
            agent_cfg.finetune_iterations = 0

        if eval_mode:
            agent_cfg.policy.act_inference_robust = True
        else:
            agent_cfg.robust_train_only = True

        if args_cli.alg_name == "domain_rand_priv":
            agent_cfg.policy.robust_use_latent_critic = True

    else:
        raise ValueError("invalid alg_name")

    return agent_cfg


def parse_agent_cfg(task_name: str, args_cli: argparse.Namespace, eval_mode: bool):
    """Parse configuration for agent based on inputs."""
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    # update based on algorithm
    agent_cfg = update_agent_cfg_by_alg(agent_cfg, args_cli, eval_mode)

    # override the default configuration with CLI arguments
    agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
        agent_cfg.resume = True
    else:
        agent_cfg.resume = False
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger

    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    # ---------------------- #
    # -- custom arguments -- #
    # ---------------------- #

    # -- evaluation arguments
    agent_cfg.save_eval_detailed = args_cli.save_eval_detailed
    agent_cfg.print_stats = args_cli.print_stats

    # -- finetune arguments (supervised learning phase)
    agent_cfg.calculate_validation_metrics_only = args_cli.calculate_validation_metrics_only

    agent_cfg.train_and_finetune = args_cli.train_and_finetune
    agent_cfg.finetune_fresh_start = args_cli.finetune_fresh_start
    agent_cfg.finetune_w_adversary = args_cli.finetune_w_adversary
    agent_cfg.num_learning_epochs_finetune = args_cli.num_learning_epochs_finetune

    # -- NN setup
    if args_cli.actor_hidden_dims is not None:
        agent_cfg.policy.actor_hidden_dims = args_cli.actor_hidden_dims
    if args_cli.critic_hidden_dims is not None:
        agent_cfg.policy.critic_hidden_dims = args_cli.critic_hidden_dims
    if args_cli.context_hidden_dims is not None:
        agent_cfg.policy.context_hidden_dims = args_cli.context_hidden_dims
    if args_cli.adversary_hidden_dims is not None:
        agent_cfg.policy.adversary_hidden_dims = args_cli.adversary_hidden_dims
    if args_cli.history_encoder_hidden_dims is not None:
        agent_cfg.policy.history_encoder_hidden_dims = args_cli.history_encoder_hidden_dims
    if args_cli.history_epinet_hidden_dims is not None:
        agent_cfg.policy.history_epinet_hidden_dims = args_cli.history_epinet_hidden_dims

    agent_cfg.policy.init_nn_weights = not args_cli.no_init_nn_weights
    agent_cfg.policy.use_log_std = not args_cli.use_std_direct
    agent_cfg.policy.actor_gain = args_cli.actor_gain
    agent_cfg.policy.critic_gain = args_cli.critic_gain
    agent_cfg.policy.context_gain = args_cli.context_gain
    agent_cfg.policy.adversary_gain = args_cli.adversary_gain
    agent_cfg.policy.history_gain = args_cli.history_gain
    agent_cfg.policy.history_epinet_gain = args_cli.history_epinet_gain
    agent_cfg.policy.history_epinet_prior_gain = args_cli.history_epinet_prior_gain

    agent_cfg.empirical_normalization = not args_cli.no_empirical_normalization

    # -- training arguments
    if args_cli.num_learning_epochs is not None:
        agent_cfg.algorithm.num_learning_epochs = args_cli.num_learning_epochs
    if args_cli.num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches

    agent_cfg.robust_mask_update_every = args_cli.robust_mask_update_every

    # -- adversary
    agent_cfg.policy.adversary_clamp = args_cli.adversary_clamp
    agent_cfg.policy.adversary_random_scale = not args_cli.no_adversary_random_scale
    agent_cfg.policy.adversary_random_angle = args_cli.adversary_random_angle
    agent_cfg.policy.adversary_max_scale_start = args_cli.adversary_max_scale_start
    agent_cfg.adversary_update_every = args_cli.adversary_update_every
    agent_cfg.policy.adversary_init_noise_std = args_cli.adversary_init_noise_std

    # -- context / history encoder
    agent_cfg.policy.num_latent = args_cli.num_latent
    agent_cfg.policy.latent_detach_actor = args_cli.latent_detach_actor
    agent_cfg.policy.latent_detach_critic = args_cli.latent_detach_critic
    agent_cfg.policy.no_context_critic = args_cli.no_context_critic

    agent_cfg.policy.obs_history_length = args_cli.obs_history_length
    agent_cfg.policy.context_encoder_clamp = args_cli.context_encoder_clamp
    agent_cfg.policy.context_encoder_clamp_value = args_cli.context_encoder_clamp_value
    agent_cfg.policy.context_encoder_noise_std = args_cli.context_encoder_noise_std

    agent_cfg.policy.history_epinet_output_type = args_cli.history_epinet_output_type
    agent_cfg.policy.history_epinet_finetune_scale = not args_cli.no_history_epinet_finetune_scale
    agent_cfg.policy.history_epinet_finetune_shift = not args_cli.no_history_epinet_finetune_shift
    agent_cfg.policy.history_epinet_finetune_min_quantile = args_cli.history_epinet_finetune_min_quantile
    agent_cfg.policy.history_epinet_finetune_max_quantile = args_cli.history_epinet_finetune_max_quantile
    agent_cfg.policy.history_epinet_finetune_target = args_cli.history_epinet_finetune_target

    agent_cfg.policy.history_epinet_type = args_cli.history_epinet_type
    agent_cfg.policy.history_epinet_input_type = args_cli.history_epinet_input_type
    agent_cfg.policy.history_epinet_dim = args_cli.history_epinet_dim
    agent_cfg.policy.history_epinet_num_samples = args_cli.history_epinet_num_samples
    agent_cfg.policy.history_epinet_coef = args_cli.history_epinet_coef

    agent_cfg.history_loss_actions = args_cli.history_loss_actions
    agent_cfg.history_loss_robust_as_zeros = args_cli.history_loss_robust_as_zeros
    agent_cfg.history_loss_adapt_only = args_cli.history_loss_adapt_only
    agent_cfg.history_loss_epinet_separate = not args_cli.history_loss_include_epinet
    agent_cfg.history_lr = args_cli.history_lr

    # -- validation
    agent_cfg.collect_data_from_context = args_cli.collect_data_from_context
    agent_cfg.collect_data_deterministic = not args_cli.collect_data_stochastic

    agent_cfg.uncertainty_metric_full_only = args_cli.uncertainty_metric_full_only
    agent_cfg.uncertainty_metric_adapt_only = args_cli.uncertainty_metric_adapt_only

    # class names
    agent_cfg.policy.class_name = "GRAMActorCritic"
    agent_cfg.algorithm.class_name = "GRAMPPO"

    return agent_cfg
