# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022-2024, The Isaac Lab Project Developers
#
# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa

"""Entry point for evaluation."""


def evaluate(launcher=None):
    """Evaluate agent."""

    # Load
    if launcher is None:
        from source.launcher import args_cli, simulation_app
    else:
        simulation_app, args_cli = launcher

    import os
    import random
    from datetime import datetime

    import gymnasium as gym
    import numpy as np
    import omni.isaac.lab_tasks
    import torch
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg

    from source.algs import GRAMOnPolicyRunner
    from source.envs import CUSTOM_ENVS, create_custom_env
    from source.envs.context_env_wrapper import ContextRslRlVecEnvWrapper
    from source.utils.cli_utils import parse_agent_cfg, parse_custom_inputs_cfg

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # create custom task
    if args_cli.task in CUSTOM_ENVS:
        eval_inputs_cfg = parse_custom_inputs_cfg(args_cli.task, args_cli.id_context, args_cli, eval_mode=True)
        eval_inputs_dict = eval_inputs_cfg.to_dict()
        create_custom_env(args_cli.task, **eval_inputs_dict)
    else:
        eval_inputs_dict = dict()

    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = parse_agent_cfg(args_cli.task, args_cli, eval_mode=True)

    # set seeds
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)
    torch.cuda.manual_seed(agent_cfg.seed)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if agent_cfg.resume:
        log_dir = os.path.join(log_root_path, agent_cfg.load_run)
        log_dir = os.path.join(log_dir, "eval")
    else:
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
        log_dir = os.path.join(log_dir, "eval")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": "%s-eval-video" % agent_cfg.run_name,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # environment wrapper
    env = ContextRslRlVecEnvWrapper(env, eval_inputs_cfg)

    # create runner
    runner = GRAMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        # load previously trained model
        runner.load(resume_path)
    else:
        print("[WARNING] No folder provided to load_run. Evaluating untrained model.")

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # evaluate
    eval_log = runner.evaluate(eval_inputs_dict, deterministic=(not args_cli.stochastic_eval))

    # save evaluation information
    save_name = "eval"
    if agent_cfg.run_name:
        save_name += f"_{agent_cfg.run_name}"
    dump_pickle(os.path.join(log_dir, save_name), eval_log)

    # close the simulator
    env.close()

    return simulation_app


if __name__ == "__main__":
    # run the evaluate function
    simulation_app = evaluate()
    # close sim app
    simulation_app.close()
