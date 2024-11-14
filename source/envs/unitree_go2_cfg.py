# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config classes for custom Unitree Go2 environments."""

from omni.isaac.lab.utils import configclass


def get_unitree_go2_custom_env_config(id_context):
    """Returns custom env config based on specified ID context."""
    if id_context == "base_id":  # Base ID context set
        cfg = UnitreeGo2CustomBaseIDCfg()
    elif id_context == "base_id_frozen_joints":  # Base ID + Frozen Joints context set
        cfg = UnitreeGo2CustomBaseIDFrozenJointsCfg()
    else:
        raise ValueError("invalid id_context")

    return cfg


################################################################################
# Base ID config
################################################################################


@configclass
class UnitreeGo2CustomBaseIDCfg:
    """Config for Unitree Go2 custom task w/ Base ID context set."""

    """Terrain."""
    terrain_roughness_cm: float = 0.0
    height_scan: bool = False

    """Collisions."""
    activate_self_collisions: bool = True

    """Additive base mass."""
    base_mass_min: float = -1.00
    base_mass_max: float = 3.00
    base_mass_context_include: bool = True
    base_mass_context_scale: bool = True
    base_mass_context_scale_min: float = -1.00
    base_mass_context_scale_max: float = 3.00

    """Multiplicative friction."""
    friction_mult_min: float = 0.25
    friction_mult_max: float = 2.00
    friction_mult_context_include: bool = True
    friction_mult_context_scale: bool = True
    friction_mult_context_scale_min: float = 0.25
    friction_mult_context_scale_max: float = 2.00

    """Motor scale."""
    motor_strength_mult_min: float = 0.80
    motor_strength_mult_max: float = 1.20
    motor_strength_mult_context_include: bool = True
    motor_strength_mult_context_scale: bool = True
    motor_strength_mult_context_scale_min: float = 0.80
    motor_strength_mult_context_scale_max: float = 1.20
    motor_fault_indices: list = None

    """Action shift / scale / clamp."""
    action_mult_indices: list = None
    action_mult_min: float = 0.0
    action_mult_max: float = 0.0
    action_mult_count: int = None
    action_mult_context_include: bool = False
    action_mult_context_scale: bool = True
    action_mult_context_scale_min: float = 0.0
    action_mult_context_scale_max: float = 1.0

    joint_bias_min: float = -0.10
    joint_bias_max: float = 0.10
    joint_bias_context_include: bool = True
    joint_bias_context_scale: bool = True
    joint_bias_context_scale_min: float = -0.10
    joint_bias_context_scale_max: float = 0.10

    joint_pos_clamp: bool = True

    """Adversary."""
    adversary_magnitude: float = 0.0

    """Task."""
    target_x_vel_min: float = 0.5
    target_x_vel_max: float = 1.0

    target_y_vel_min: float = 0.0
    target_y_vel_max: float = 0.0

    disable_obs_noise: bool = False

    """Evaluation mode options."""
    eval_mode: bool = False
    terminate_base_height: bool = True
    base_height_terminate_thresh: float = 0.15


################################################################################
# Base ID + Frozen Joints config
################################################################################


@configclass
class UnitreeGo2CustomBaseIDFrozenJointsCfg(UnitreeGo2CustomBaseIDCfg):
    """Config for Unitree Go2 custom task w/ Base ID + Frozen Joints context set."""

    def __post_init__(self):
        super().__post_init__()

        self.action_mult_context_include = True
        self.action_mult_indices = [[-1], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
