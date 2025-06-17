# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

from source.envs.unitree_go2_cfg import get_unitree_go2_custom_env_config
from source.envs.unitree_go2_custom_env import create_unitree_go2_custom_env

################################################################################
# Custom task loader
################################################################################

CUSTOM_ENVS = [
    "Isaac-Velocity-Custom-Unitree-Go2-v0",
]


def get_custom_inputs_cfg(task, id_context):
    if task in CUSTOM_ENVS:
        cfg = get_unitree_go2_custom_env_config(id_context)
    else:
        raise ValueError("custom task %s not defined" % task)

    return cfg


def create_custom_env(task, **kwargs):
    if task in CUSTOM_ENVS:
        create_unitree_go2_custom_env(task, **kwargs)
    else:
        raise ValueError("custom task %s not defined" % task)
