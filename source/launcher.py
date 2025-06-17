# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022-2024, The Isaac Lab Project Developers
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create parser and launch Isaac Sim Simulator."""


from omni.isaac.lab.app import AppLauncher

from source.utils.cli_utils import create_parser

# create parser
parser = create_parser()
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always run in headless mode
args_cli.headless = True
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
