"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
@File: pointnet2_wypr.py
@Author:Huitong Jin
@Date:2023/2/01
"""

# =============================
# imports and global variables
# =============================
import sys

sys.path.append('/data/jinhuitong/Code/pointnet2_pcam')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
