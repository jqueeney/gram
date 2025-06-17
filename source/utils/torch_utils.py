# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch utilities."""


import torch.nn as nn


def weight_init(module):
    """Initializes PyTorch module weights based on Xavier normal."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        try:
            module.bias.data.zero_()
        except:  # noqa: E722
            pass

    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_normal_(module.weight)
        try:
            module.bias.data.zero_()
        except:  # noqa: E722
            pass

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_normal_(module.weight_ih_l0)
        nn.init.xavier_normal_(module.weight_hh_l0)
        try:
            module.bias_ih_l0.data.zero_()
            module.bias_hh_l0.data.zero_()
        except:  # noqa: E722
            pass
