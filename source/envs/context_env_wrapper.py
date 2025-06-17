# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that contains context information."""

import torch
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


class ContextRslRlVecEnvWrapper(RslRlVecEnvWrapper):

    def __init__(self, env, inputs_cfg):
        super().__init__(env)

        self._setup_context_args(inputs_cfg)
        self._set_context()

    def _setup_context_args(self, inputs_cfg):

        # Body mass
        self.base_mass_context_include = inputs_cfg.base_mass_context_include
        self.base_mass_context_scale = inputs_cfg.base_mass_context_scale
        if self.base_mass_context_include and self.base_mass_context_scale:
            if inputs_cfg.base_mass_context_scale_min is not None:
                self.base_mass_context_scale_min = inputs_cfg.base_mass_context_scale_min
            else:
                self.base_mass_context_scale_min = inputs_cfg.base_mass_min

            if inputs_cfg.base_mass_context_scale_max is not None:
                self.base_mass_context_scale_max = inputs_cfg.base_mass_context_scale_max
            else:
                self.base_mass_context_scale_max = inputs_cfg.base_mass_max

            self.base_mass_context_scale_range = self.base_mass_context_scale_max - self.base_mass_context_scale_min

        # Friction
        self.friction_mult_context_include = inputs_cfg.friction_mult_context_include
        self.friction_mult_context_scale = inputs_cfg.friction_mult_context_scale
        if self.friction_mult_context_include and self.friction_mult_context_scale:
            if inputs_cfg.friction_mult_context_scale_min is not None:
                self.friction_mult_context_scale_min = inputs_cfg.friction_mult_context_scale_min
            else:
                self.friction_mult_context_scale_min = inputs_cfg.friction_mult_min

            if inputs_cfg.friction_mult_context_scale_max is not None:
                self.friction_mult_context_scale_max = inputs_cfg.friction_mult_context_scale_max
            else:
                self.friction_mult_context_scale_max = inputs_cfg.friction_mult_max

            self.default_friction = 0.8
            self.friction_mult_context_scale_range = (
                self.friction_mult_context_scale_max - self.friction_mult_context_scale_min
            )

        # Action multiple
        self.action_mult_context_include = inputs_cfg.action_mult_context_include
        self.action_mult_context_scale = inputs_cfg.action_mult_context_scale
        if self.action_mult_context_include and self.action_mult_context_scale:
            if inputs_cfg.action_mult_context_scale_min is not None:
                self.action_mult_context_scale_min = inputs_cfg.action_mult_context_scale_min
            else:
                self.action_mult_context_scale_min = inputs_cfg.action_mult_min

            if inputs_cfg.action_mult_context_scale_max is not None:
                self.action_mult_context_scale_max = inputs_cfg.action_mult_context_scale_max
            else:
                self.action_mult_context_scale_max = inputs_cfg.action_mult_max

            self.action_mult_context_scale_range = (
                self.action_mult_context_scale_max - self.action_mult_context_scale_min
            )

        # Joint angle bias
        self.joint_bias_context_include = inputs_cfg.joint_bias_context_include
        self.joint_bias_context_scale = inputs_cfg.joint_bias_context_scale
        if self.joint_bias_context_include and self.joint_bias_context_scale:
            if inputs_cfg.joint_bias_context_scale_min is not None:
                self.joint_bias_context_scale_min = inputs_cfg.joint_bias_context_scale_min
            else:
                self.joint_bias_context_scale_min = inputs_cfg.joint_bias_min

            if inputs_cfg.joint_bias_context_scale_max is not None:
                self.joint_bias_context_scale_max = inputs_cfg.joint_bias_context_scale_max
            else:
                self.joint_bias_context_scale_max = inputs_cfg.joint_bias_max

            self.joint_bias_context_scale_range = self.joint_bias_context_scale_max - self.joint_bias_context_scale_min

        # Motor strength multiple
        self.motor_strength_mult_context_include = inputs_cfg.motor_strength_mult_context_include
        self.motor_strength_mult_context_scale = inputs_cfg.motor_strength_mult_context_scale
        if self.motor_strength_mult_context_include and self.motor_strength_mult_context_scale:
            if inputs_cfg.motor_strength_mult_context_scale_min is not None:
                self.motor_strength_mult_context_scale_min = inputs_cfg.motor_strength_mult_context_scale_min
            else:
                self.motor_strength_mult_context_scale_min = inputs_cfg.motor_strength_mult_min

            if inputs_cfg.motor_strength_mult_context_scale_max is not None:
                self.motor_strength_mult_context_scale_max = inputs_cfg.motor_strength_mult_context_scale_max
            else:
                self.motor_strength_mult_context_scale_max = inputs_cfg.motor_strength_mult_max

            self.motor_strength_mult_context_scale_range = (
                self.motor_strength_mult_context_scale_max - self.motor_strength_mult_context_scale_min
            )

    def _set_context(self):
        """Sets context."""
        self.context = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)

        # Body mass
        if self.base_mass_context_include:
            term_cfg = self.unwrapped.event_manager.get_term_cfg("add_base_mass")
            asset_cfg = term_cfg.params["asset_cfg"]

            asset = self.unwrapped.scene[asset_cfg.name]
            body_ids = asset_cfg.body_ids
            mass = asset.root_physx_view.get_masses()
            default_mass = asset.data.default_mass
            delta_mass = mass[:, body_ids] - default_mass[:, body_ids]
            delta_mass = delta_mass.to(device=self.device)

            if self.base_mass_context_scale:
                delta_shift_scale = (delta_mass - self.base_mass_context_scale_min) / self.base_mass_context_scale_range
                delta_mass_context = -1 + 2 * delta_shift_scale
            else:
                delta_mass_context = delta_mass

            self.context = torch.cat([self.context, delta_mass_context], dim=-1)

        # Friction
        if self.friction_mult_context_include:
            term_cfg = self.unwrapped.event_manager.get_term_cfg("physics_material")
            asset_cfg = term_cfg.params["asset_cfg"]

            asset = self.unwrapped.scene[asset_cfg.name]
            materials = asset.root_physx_view.get_material_properties()

            # -1 index captures material of a foot
            self.materials = materials[:, -1, :].to(device=self.device)
            # friction: only include single value, since static and dynamic friction are scaled by same factor
            friction = self.materials[:, :1]

            if self.friction_mult_context_include and self.friction_mult_context_scale:
                friction_mult = friction / self.default_friction
                friction_shift_scale = (
                    friction_mult - self.friction_mult_context_scale_min
                ) / self.friction_mult_context_scale_range
                friction_context = -1 + 2 * friction_shift_scale
            else:
                friction_context = friction

            if self.friction_mult_context_include:
                self.context = torch.cat([self.context, friction_context], dim=-1)

        # Action multiple
        if self.action_mult_context_include:
            action_scale = self.unwrapped.action_manager.get_term("joint_pos").action_scale.detach()

            if self.action_mult_context_scale:
                action_context_normalized = (
                    action_scale - self.action_mult_context_scale_min
                ) / self.action_mult_context_scale_range
                action_scale_context = -1 + 2 * action_context_normalized
            else:
                action_scale_context = action_scale

            self.context = torch.cat([self.context, action_scale_context], dim=-1)

        # Joint angle bias
        if self.joint_bias_context_include:
            action_shift = self.unwrapped.action_manager.get_term("joint_pos").action_shift.detach()

            if self.joint_bias_context_scale:
                action_shift_context_normalized = (
                    action_shift - self.joint_bias_context_scale_min
                ) / self.joint_bias_context_scale_range
                action_shift_context = -1 + 2 * action_shift_context_normalized
            else:
                action_shift_context = action_shift

            self.context = torch.cat([self.context, action_shift_context], dim=-1)

        # Motor strength multiple
        if self.motor_strength_mult_context_include:
            motor_scale = self.unwrapped.scene["robot"].actuators["base_legs"].get_effort_scale()

            if self.motor_strength_mult_context_scale:
                motor_context_shift_scale = (
                    motor_scale - self.motor_strength_mult_context_scale_min
                ) / self.motor_strength_mult_context_scale_range
                motor_scale_context = -1 + 2 * motor_context_shift_scale
            else:
                motor_scale_context = motor_scale

            self.context = torch.cat([self.context, motor_scale_context], dim=-1)

        self.context_dim = self.context.shape[1]

    def get_context(self) -> torch.Tensor:
        """Returns context."""
        return self.context
