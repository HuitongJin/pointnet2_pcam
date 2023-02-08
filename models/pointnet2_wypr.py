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


class Pointnet2Backbone(nn.Module):
    """
    Backbone network for point cloud feature learning.
    Based on Pointnet++ single-scale grouping network.
    Parameters
    ----------
    input_feature_dim: int
        Number of input channels in the feature descriptor for each point.
    e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        """
        Forward pass of the network
        Parameters
        ----------
        pointcloud: torch.cuda.FloatTensor (B, N, 3 + input_feature_dim) tensor
        Point cloud to run predicts on Each point in the point-cloud MUST
        be formated as (x, y, z, features...)

        Returns
        ----------
        end_points: {XXX_xyz, XXX_features, XXX_inds}
        XXX_xyz: float32 Tensor of shape (B,K,3)
        XXX_features: float32 Tensor of shape (B,K,D)
        XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz
        end_points['input_features'] = features
        # print("xyz shape:", xyz.shape)
        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features  # N x 256 x 256

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)

        # use fp2 output as the backbone output
        end_points['backbone_feat'] = features
        end_points['backbone_feat_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['sa2_xyz'].shape[1]
        end_points['backbone_feat_inds'] = end_points['sa1_inds'][:, :num_seed]  # indices among the entire input point clouds
        return end_points


class Pointnet2SegHead(nn.Module):
    """ Segmentation head for pointnet++ """

    def __init__(self, input_feature_dim, num_class, suffix=""):
        super().__init__()
        self.suffix = suffix
        self.seg_fp1 = PointnetFPModule(mlp=[256 + 128, 256, 256])
        self.seg_fp2 = PointnetFPModule(mlp=[input_feature_dim + 256, 256, 256])
        self.norm = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(True))

        # self.classifier = torch.nn.Sequential(
        #                     torch.nn.BatchNorm1d(256),
        #                     torch.nn.ReLU(True),
        #                     torch.nn.Conv1d(256, num_class, kernel_size=1))
        # self.conv1d = torch.nn.Conv1d(256, num_class, kernel_size=1)

    def forward(self, end_points=None, classification=True):
        """Forward pass of the network """
        features_1 = self.seg_fp1(
            end_points['sa1_xyz'], end_points['sa2_xyz'],
            end_points['sa1_features'], end_points['backbone_feat'])
        # print("feature_1 shape:", features_1.shape)
        features_2 = self.seg_fp2(
            end_points['input_xyz'], end_points['sa1_xyz'],
            end_points['input_features'], features_1)
        # print("feature_2 shape:", features_2.shape)
        end_points['sem_seg_feat' + self.suffix] = self.norm(features_2)
        # if classification:
        #     end_points['sem_seg_pred'+self.suffix] = self.classifier(features_2)
        #     end_points['sem_seg_pred_avg'] = end_points['sem_seg_pred'+self.suffix].mean(dim=-1)
        return end_points


class get_model(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.point2backbone = Pointnet2Backbone(input_feature_dim=3)
        self.point2seghead = Pointnet2SegHead(input_feature_dim=3, num_class=num_class)
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv1d(256, num_class, kernel_size=1))
        self.conv1d = torch.nn.Conv1d(256, num_class, kernel_size=1)

    def forward(self, pointcloud):
        end_points = self.point2backbone(pointcloud)
        end_points = self.point2seghead(end_points)

        # 1.5

        end_points['sem_seg_pred'] = self.conv1d(end_points['sem_seg_feat'])
        end_points['sem_seg_pred_avg'] = end_points['sem_seg_pred'].mean(dim=-1)
        return end_points['sem_seg_pred_avg'], end_points['sem_seg_pred']

    def forward_cam(self, pointcloud):
        end_points = self.point2backbone(pointcloud)
        end_points = self.point2seghead(end_points)

        # 1.5
        cam = F.conv1d(end_points['sem_seg_feat'], self.conv1d.weight)

        end_points['sem_seg_pred'] = F.relu(cam)
        return end_points['sem_seg_pred']


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, pred, target):
        total_loss = self.criterion(pred, target)
        return total_loss


class get_accuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, target):
        predicted = torch.round(torch.sigmoid(pred))

        correct = (predicted == target)
        correct = torch.mean(correct.float())
        return correct


if __name__ == '__main__':
    # backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    # print(backbone_net)
    # backbone_net.eval()
    # out = backbone_net(torch.rand(16,20000,6).cuda())
    # for key in sorted(out.keys()):
    #     print(key, '\t', out[key].shape)
    backbone_net = get_model(20).cuda()
    # print(backbone_net)
    backbone_net.eval()
    out1, out2 = backbone_net(torch.rand(16, 20000, 6).cuda())
    print(out1.shape, out2.shape)
    # for key in sorted(out.keys()):
    #     print(key, '\t', out[key].shape)
