"""
@File: train_pointnet2_pcam.py
@Author:Huitong Jin
@Date:2023/1/11
"""

# =============================
# imports and global variables
# =============================
import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.Scannetv2DataLoader import ScannetDataset
from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # return to the upper directory
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # reference models


# ===========================
# create argparse parameters
# ===========================
def parse_args():
    parser = argparse.ArgumentParser("training")
    parser.add_argument("--use_cpu", action="store_true", default=False, help="use cpu mode")
    parser.add_argument("--gpu", type=str, default=2, help="specify gpu device")
    parser.add_argument("--batch_size", type=int, default=30, help="batch size in training")
    parser.add_argument("--model", type=str, default="pointnet2_wypr", help="model name [default:pointnet2_wypr]")
    parser.add_argument("--num_category", type=int, default=20, help="number of category on dataset")
    parser.add_argument("--epoch", type=int, default=200, help="number of epoch in training")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="learning rate in training")
    parser.add_argument("--num_point", type=int, default=1024, help="number of points")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer for training")
    parser.add_argument("--log_dir", type=str, default="pointnet2_wypr", help="log file directory")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="decay rate")
    parser.add_argument("--use_normals", action="store_true", default=False, help="use normals")
    parser.add_argument("--process_data", action="store_true", default=False, help="save data offline")
    parser.add_argument("--use_uniform_sample", action="store_true", default=False, help="use uniform sampling")
    return parser.parse_args()


# ==========
#  Utilities
# ==========
def inplace_relu(m):
    """
    Determine whether the activation function is relu？if not， replace it with relu.
    """
    class_name = m.__class__.__name__
    if class_name.find('Relu') != -1:
        m.inplace = True


def exists(path):
    """
    Test whether a path exists, Returns False for broken symbolic links.
    """
    try:
        os.stat(path)
    except OSError:
        return False
    return True


def fast_confusion(y_true, y_pred, label_values=None):
    """
    Fast confusion matrix (100x faster than Scikit learn). But only works if labels are la.
    :param y_true: real label
    :param y_pred: predicted label
    :param label_values: label list
    :return: confusion matrix (n * n)
    """
    # Ensure data is in the right format
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if len(y_true.shape) != 1:
        raise ValueError('Truth values are stored in a {:d}D array instead of 1D array'.format(len(y_true.shape)))
    if len(y_pred.shape) != 1:
        raise ValueError('Prediction values are stored in a {:d}D array instead of 1D array'.format(len(y_pred.shape)))
    if y_true.dtype not in [np.int32, np.int64]:
        raise ValueError('Truth values are {:s} instead of int32 or int64'.format(y_true.dtype))
    if y_pred.dtype not in [np.int32, np.int64]:
        raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(y_pred.dtype))
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    # Get the label values
    if label_values is None:
        # From data if they are not given
        label_values = np.unique(np.hstack((y_true, y_pred)))
    else:
        # Ensure they are good if given
        if label_values.dtype not in [np.int32, np.int64]:
            raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
        if len(np.unique(label_values) < len(label_values)):
            raise ValueError('Given labels are not unique')

    # Start labels
    label_values = np.sort(label_values)
    # Get the number of classes
    num_classes = len(label_values)
    # Start confusion computations
    if label_values[0] == 0 and label_values[-1] == num_classes - 1:
        # Vectorized confusion
        vec_conf = np.bincount(y_true * num_classes + y_pred)
        # Add possible missing value due to classes not being in y_pred or y_true
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')
        # Reshape confusion in a matrix
        return vec_conf.reshape(num_classes, num_classes)

    else:
        # Ensure no negative classes
        if label_values[0] < 0:
            raise ValueError('Unsupported negative classes')
        # Get the data in [0, num_classed]
        label_map = np.zeros((label_values[-1] + 1,), dtype=np.int32)
        for k, v in enumerate(label_values):
            label_map[v] = k
        y_pred = label_map[y_pred]
        y_true = label_map[y_true]

        # Vectorized confusion
        vec_conf = np.bincount(y_true * num_classes + y_pred)
        # Add possible missing values due to classes not being in y_pred or y_true
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')
        # Reshape confusion in a matrix
        return vec_conf.reshape((num_classes, num_classes))


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :return: ([..., n_c] np.float32) IoU score
    """
    # Compute TP, FP, FN. This assume that the second to last axit counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU


# =============
# Main Function
# =============
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """CREATE DIR"""
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('pointnet2_pcam')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(time_str)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    writer = SummaryWriter(log_dir='tensorboard_wypr_conv_3', filename_suffix='test')

    """DATA LOADING"""
    log_string('Load dataset ...')
    data_path = '/data/dataset/scannet'
    train_dataset = ScannetDataset

