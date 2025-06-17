# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config classes for custom Unitree Go2 environments."""

import math

from omni.isaac.lab.utils import configclass


def get_unitree_go2_custom_env_config(id_context):
    """Returns custom env config based on specified ID context."""
    if id_context == "default":  # Default ID context set
        cfg = UnitreeGo2CustomDefaultIDCfg()
    elif id_context == "wide":  # Wide ID context set
        cfg = UnitreeGo2CustomWideIDCfg()
    else:
        raise ValueError("invalid id_context")

    return cfg


################################################################################
# Default ID training config
################################################################################


@configclass
class UnitreeGo2CustomDefaultIDCfg:
    """Config for Unitree Go2 custom task w/ default ID context set."""

    """Terrain."""
    terrain_roughness_cm: float = 0.0
    terrain_slope_degrees: float = 0.0
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
    friction_mult_min: float = 0.05
    friction_mult_max: float = 4.50
    friction_mult_context_include: bool = True
    friction_mult_context_scale: bool = True
    friction_mult_context_scale_min: float = 0.05
    friction_mult_context_scale_max: float = 4.50

    """Motor scale."""
    motor_strength_mult_min: float = 0.80
    motor_strength_mult_max: float = 1.20
    motor_strength_mult_context_include: bool = True
    motor_strength_mult_context_scale: bool = True
    motor_strength_mult_context_scale_min: float = 0.80
    motor_strength_mult_context_scale_max: float = 1.20
    motor_fault_indices: list = None

    motor_Kp: float = 25.0
    motor_Kd: float = 0.5

    """Action shift / scale / clamp."""
    hip_scale_mult: float = 0.5

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
    target_x_vel_min: float = -1.0
    target_x_vel_max: float = 1.0

    target_y_vel_min: float = -1.0
    target_y_vel_max: float = 1.0

    target_heading_min: float = -math.pi
    target_heading_max: float = math.pi
    rel_standing_envs: float = 0.02
    resample_time_min: float = 10.0
    resample_time_max: float = 10.0
    start_yaw_min: float = -math.pi
    start_yaw_max: float = math.pi

    disable_obs_noise: bool = False

    """Evaluation mode options."""
    eval_mode: bool = False
    terminate_contacts: bool = True

    """Display mode options."""
    display_mode: bool = False
    display_type: str = "world"
    display_resolution: int = 720


################################################################################
# Wide ID training config
################################################################################


@configclass
class UnitreeGo2CustomWideIDCfg(UnitreeGo2CustomDefaultIDCfg):
    """Config for Unitree Go2 custom task w/ wide ID context set."""

    def __post_init__(self):
        super().__post_init__()

        self.base_mass_max = 9.00
        self.base_mass_context_scale_max = 9.00
