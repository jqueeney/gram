# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022-2024, The Isaac Lab Project Developers
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities used in custom Unitree Go2 environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
from omni.isaac.lab.actuators import DCMotor, DCMotorCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

################################################################################
# Action terms
################################################################################


class ScaledJointPositionAction(mdp.JointPositionAction):
    """Joint action term that scales magnitudes of raw actions, and applies the
    processed actions to the articulation's joints as position commands.
    """

    cfg: ScaledJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ScaledJointPositionActionCfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # default action scale
        self._action_scale = torch.ones((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)

        self._set_action_scale()

    def _set_action_scale(self):
        action_scale_default = np.ones((self.num_envs, self.action_dim))

        if self.cfg.action_mult_indices is not None:
            action_scale_mask_options = np.zeros((len(self.cfg.action_mult_indices), self.action_dim))
            for row_idx, action_idx in enumerate(self.cfg.action_mult_indices):
                if not (action_idx == [-1]):
                    action_scale_mask_options[row_idx, action_idx] = 1.0

            mask_indices = np.random.choice(len(self.cfg.action_mult_indices), self.num_envs)
            action_scale_idx_mask = action_scale_mask_options[mask_indices]
            action_scale_default_mask = 1.0 - action_scale_idx_mask

            if self.cfg.action_mult_count is not None:
                idx_scale_choices = np.linspace(
                    self.cfg.action_mult_min, self.cfg.action_mult_max, self.cfg.action_mult_count
                )

                if idx_scale_choices.shape[0] > 1:
                    action_scale_idx_mults = np.random.choice(idx_scale_choices, self.num_envs)
                else:
                    action_scale_idx_mults = np.ones(self.num_envs) * idx_scale_choices[0]
            else:
                if self.cfg.action_mult_min < self.cfg.action_mult_max:
                    action_scale_idx_mults = self.cfg.action_mult_min + (
                        np.random.random(self.num_envs) * (self.cfg.action_mult_max - self.cfg.action_mult_min)
                    )
                else:
                    action_scale_idx_mults = np.ones(self.num_envs) * self.cfg.action_mult_min

            action_scale_idx_mults = np.expand_dims(action_scale_idx_mults, axis=-1)
            action_scale_idx_active = action_scale_idx_mask * action_scale_idx_mults
            action_scale_default_active = action_scale_default_mask * action_scale_default

            action_scale = action_scale_default_active + action_scale_idx_active

        else:
            action_scale = action_scale_default

        self._action_scale[:] = torch.tensor(action_scale, dtype=torch.float32, device=self.device)

    @property
    def action_scale(self) -> torch.Tensor:
        return self._action_scale

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = (self._raw_actions * self._action_scale) * self._scale + self._offset


# ------------------------------------------------------------------------------#


@configclass
class ScaledJointPositionActionCfg(mdp.JointPositionActionCfg):
    """Configuration for the scaled joint position action term.

    See :class:`ScaledJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = ScaledJointPositionAction

    action_mult_indices: list = None
    action_mult_min: float = 1.0
    action_mult_max: float = 1.0
    action_mult_count: int = None


################################################################################


class ShiftScaleClampJointPositionAction(ScaledJointPositionAction):
    """Joint action term that scales magnitudes of raw actions, shifts processed action,
    and applies the clamped processed actions to the articulation's joints as position commands.
    """

    cfg: ShiftScaleClampJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ShiftScaleClampJointPositionActionCfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # default action shift
        self._action_shift = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)

        self._set_action_shift()

        # action limits
        self.joint_pos_min = self._asset.data.soft_joint_pos_limits[..., 0]
        self.joint_pos_max = self._asset.data.soft_joint_pos_limits[..., 1]

    def _set_action_shift(self):
        if self.cfg.joint_bias_min < self.cfg.joint_bias_max:
            action_shift = self.cfg.joint_bias_min + (
                np.random.random((self.num_envs, self.action_dim)) * (self.cfg.joint_bias_max - self.cfg.joint_bias_min)
            )
        else:
            action_shift = np.ones((self.num_envs, self.action_dim)) * self.cfg.joint_bias_min

        self._action_shift[:] = torch.tensor(action_shift, dtype=torch.float32, device=self.device)

    @property
    def action_shift(self) -> torch.Tensor:
        return self._action_shift

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        processed_actions = (self._raw_actions * self._action_scale) * self._scale + self._offset + self._action_shift
        if self.cfg.joint_pos_clamp:
            processed_actions.clamp_(min=self.joint_pos_min, max=self.joint_pos_max)
        self._processed_actions[:] = processed_actions


# ------------------------------------------------------------------------------#


@configclass
class ShiftScaleClampJointPositionActionCfg(ScaledJointPositionActionCfg):
    """Configuration for the shift, scale, clamp joint position action term.

    See :class:`ShiftScaleClampScaledJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = ShiftScaleClampJointPositionAction

    joint_bias_min: float = 0.0
    joint_bias_max: float = 0.0
    joint_pos_clamp: bool = False


################################################################################


################################################################################
# Event functions
################################################################################


def randomize_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    friction_mult_range: tuple[float, float],
    static_friction_default: float,
    dynamic_friction_default: float,
    restitution_default: float,
    num_buckets: int,
    asset_cfg: SceneEntityCfg,
):
    """Randomize the physics materials on all geometries of the asset.

    For each environment, multiplies static and dynamic friction of all geometries by same value.

    Modifies randomize_rigid_body_material in Isaac Lab.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if not isinstance(asset, (RigidObject, Articulation)):
        raise ValueError(
            f"Randomization term 'randomize_friction' not supported for asset: '{asset_cfg.name}'"
            f" with type: '{type(asset)}'."
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # retrieve material buffer and set to defaults
    materials = asset.root_physx_view.get_material_properties()

    materials[..., 0] = static_friction_default
    materials[..., 1] = dynamic_friction_default
    materials[..., 2] = restitution_default

    # sample update factors
    friction_mult_buckets = torch.linspace(friction_mult_range[0], friction_mult_range[1], num_buckets, device="cpu")
    friction_mult_idx = torch.randint(num_buckets, (env.scene.num_envs,), device="cpu")
    friction_mult = friction_mult_buckets[friction_mult_idx]

    # update materials
    if isinstance(asset, Articulation) and asset_cfg.body_ids != slice(None):
        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        num_shapes_per_body = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)

        # set materials for each body_id
        for body_id in asset_cfg.body_ids:
            # start index of shape
            start_idx = sum(num_shapes_per_body[:body_id])
            # end index of shape
            end_idx = start_idx + num_shapes_per_body[body_id]

            # update materials
            materials[:, start_idx:end_idx, 0] = static_friction_default * friction_mult.unsqueeze(dim=-1)
            materials[:, start_idx:end_idx, 1] = dynamic_friction_default * friction_mult.unsqueeze(dim=-1)
    else:
        # update materials
        materials[..., 0] = static_friction_default * friction_mult.unsqueeze(dim=-1)
        materials[..., 1] = dynamic_friction_default * friction_mult.unsqueeze(dim=-1)

    # apply to simulation
    asset.root_physx_view.set_material_properties(materials, env_ids)


# ------------------------------------------------------------------------------#


def adversarial_push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    magnitude: float = 0.0,
    adversary_actions: torch.Tensor = None,
    adversary_masks: torch.Tensor = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity based on an adversary."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if (adversary_actions is not None) and (adversary_masks is not None) and (magnitude > 0):

        # velocities
        vel_w = asset.data.root_vel_w[env_ids]
        # push x,y velocity
        processed_adversary_actions = adversary_actions.clamp(min=-1, max=1)
        vel_xy_push_all = magnitude * processed_adversary_actions
        vel_xy_push = vel_xy_push_all[env_ids]

        # additive
        vel_w[:, :2] += adversary_masks[env_ids] * vel_xy_push

        # set the velocities into the physics simulation
        asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


################################################################################
# Actuator
################################################################################


class ScaledDCMotor(DCMotor):
    """Scaled version of DCMotor."""

    cfg: ScaledDCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: ScaledDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._set_effort_scale()

    def _set_effort_scale(self):
        self._effort_scale = torch.ones_like(self.computed_effort)
        self._effort_scale.uniform_(self.cfg.motor_strength_mult_min, self.cfg.motor_strength_mult_max)

        if self.cfg.motor_fault_indices is not None:
            motor_fault_mask_options = np.ones((len(self.cfg.motor_fault_indices), self._effort_scale.shape[-1]))
            for row_idx, action_idx in enumerate(self.cfg.motor_fault_indices):
                if not (action_idx == [-1]):
                    motor_fault_mask_options[row_idx, action_idx] = 0.0

            mask_indices = np.random.choice(len(self.cfg.motor_fault_indices), self._effort_scale.shape[0])
            motor_fault_mask = motor_fault_mask_options[mask_indices]

            motor_fault_mask = torch.tensor(motor_fault_mask, dtype=torch.float32, device=self._effort_scale.device)

            self._effort_scale[:] = self._effort_scale * motor_fault_mask

    def get_effort_scale(self):
        return self._effort_scale

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        scaled_effort = self._effort_scale * effort
        return super()._clip_effort(scaled_effort)


@configclass
class ScaledDCMotorCfg(DCMotorCfg):
    """Configuration for scaled DC motor actuator model."""

    class_type: type = ScaledDCMotor

    motor_strength_mult_min: float = 1.0
    motor_strength_mult_max: float = 1.0
    motor_fault_indices: list = None


# ------------------------------------------------------------------------------#
