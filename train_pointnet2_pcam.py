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
    # Compute TP, FP, FN. This assumes that the second to last axit counts the truths (like the first axis of a
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
    exp_dir = exp_dir.joinpath('pointnet2_pcam')  # 第一个地方
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

    writer = SummaryWriter(log_dir='tensorboard_wypr_conv_3', filename_suffix='test')  # 第二个地方

    """DATA LOADING"""
    log_string('Load dataset ...')
    data_path = '/data/dataset/scannet'
    train_dataset = ScannetDataset(path=data_path, npoints=40000, split='train')
    test_dataset = ScannetDataset(path=data_path, npoints=40000, split='val')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    """MODEL LOADING"""
    num_class = args.num_category
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class)
    criterion = model.get_loss()
    accuracy = model.get_accuracy()

    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        accuracy = accuracy.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    # 按固定的训练epoch数进行学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_iou_acc = 0.0
    num_batches = len(trainDataLoader)

    """TRAIN"""
    logger.info('Start training...')
    iter_count = 0

    test_dirs = './test_pred_wypr_conv_3'  # 第三个地方
    if not os.path.exists(test_dirs):
        os.makedirs(test_dirs)

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = 0
        loss_sum = 0
        classifier = classifier.train()

        scheduler.step()
        with tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9) as pbar:

            for batch_id, (points, cloud_labels_all, target, gt_label, mask, _) in pbar:
                optimizer.zero_grad()

                points = points.data.numpy()
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)[:, :, :6]
                points, target = points.float().cuda(), target.long().cuda()
                # print("points shape:", points.shape, target.shape)
                # points = points.transpose(2, 1)
                target = target.transpose(2, 1).squeeze(-1)

                if not args.use_cpu:
                    points, cloud_labels_all, target = points.cuda(), cloud_labels_all.cuda(), target.cuda()
                pred = classifier(points)
                loss = criterion(pred, target.long())
                acc = accuracy(pred, target)
                pbar.set_postfix(Loss=loss.cpu().detach().numpy(), Accuracy=acc.cpu().detach().numpy())
                mean_correct += acc
                loss.backward()
                optimizer.step()
                global_step += 1
                loss_sum += loss

                writer.add_scalars('Loss', {'Train': loss.item()}, iter_count)
                writer.add_scalars('Acc', {"Train": acc.item()}, iter_count)
                iter_count += 1
                # log_string('Training loss: %f' % (loss.cpu().detach().numpy()))

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Train Accuracy: %f' % (mean_correct / num_batches))

        with torch.no_grad():
            num_batches = len(testDataLoader)
            classifier = classifier.eval()
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            predictions = []
            targets = []
            masks = []
            loss_all = 0.0
            total_files_name = []

            for i, (points, cloud_labels_all, target, gt_label, mask, files_name) in tqdm(enumerate(testDataLoader, 0),
                                                                                          total=len(testDataLoader),
                                                                                          smoothing=0.9):
                if not args.use_cpu:
                    points, cloud_labels_all, target = points.cuda(), cloud_labels_all.cuda(), target.cuda()

                gt_pred = classifier.forward_cam(points[:, :, :6])

                loss = criterion(pred, target.long())
                gt_pred = gt_pred.permute(0, 2, 1)
                gt_pred = torch.mul(gt_pred, cloud_labels_all)
                # print("gt_pred:", cloud_labels_all[0][0])
                gt_pred = gt_pred.cpu().numpy()
                gt_label = gt_label.cpu().numpy()
                # print("gt_pred:", gt_pred[0][0])
                gt_pred = np.argmax(gt_pred, axis=2)

                masks.append(mask)
                predictions.append(gt_pred)
                targets.append(gt_label)
                loss_all += loss.item()
                total_files_name.append(files_name)

            masks = np.concatenate(masks, axis=0)
            total_files_name = np.concatenate(total_files_name, axis=0)
            predictions = np.concatenate(predictions, axis=0).astype(np.int32)
            targets = np.concatenate(targets, axis=0).astype(np.int32)

            Confs = np.zeros((len(predictions), 21, 21), dtype=np.int32)

            for i, (probs, truth, file_name, mask) in enumerate(zip(predictions, targets, total_files_name, masks)):

                preds = test_dataset.label_values[probs + 1][mask]
                truth = truth[mask]
                np.save(os.path.join(test_dirs, file_name), preds)

                Confs[i, :, :] = fast_confusion(truth, preds, test_dataset.label_values).astype(np.int32)

            C = np.sum(Confs, axis=0).astype(np.float32)

            # remove ignored labels from confusions
            C = np.delete(C, 0, axis=0)
            C = np.delete(C, 0, axis=1)

            IoUs = IoU_from_confusions(C)
            loss_mean_epoch = loss_all / num_batches

            mIou = 100 * np.mean(IoUs)
            log_string("Mean IoU = {:.1f}%".format(mIou))
            log_string("Best Mean IoU = {:.1f}%".format(best_iou_acc))
            log_string(IoUs)
            writer.add_scalars("Loss", {"Valid": loss_mean_epoch}, iter_count)
            writer.add_scalars("Miou", {"Valid": mIou}, iter_count)

            if mIou > best_iou_acc:
                best_iou_acc = mIou
                logger.info('Save model...')
                save_path = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s ' % save_path)
                state = {
                    'epoch': epoch,
                    'miou': mIou,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, save_path)

            global_epoch += 1
        writer.close()
        logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
