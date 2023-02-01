import pickle
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# ================
# Define PLY types
# ================
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# ===================
# Numpy reader format
# ===================
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}


# =========
# Functions
# =========
def parse_header(plyfile, ext):
    # Variable
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])
        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variable
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)
    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, 'rb') as plyfile:
        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')
        # Get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')
        # Get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:
            # parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)
            # get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)
            # get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)
            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:
            # Parse header
            num_points, properties = parse_header(plyfile, ext)
            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, filed_names):
    # List of lines to write
    lines = []
    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])
    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1
    return lines


def collate_fn_points(data):
    print(data[0][0].shape)
    points_stack = np.concatenate([d[0] for d in data], axis=0).astype(np.float32)
    cloud_labels_all = np.concatenate([d[1] for d in data], axis=0)
    weakly_labels_stack = np.concatenate([d[2] for d in data], axis=0)
    gt_labels_stack = np.concatenate([d[3] for d in data], axis=0)
    mask = np.concatenate([d[4] for d in data], axis=0)
    return torch.from_numpy(points_stack), torch.from_numpy(cloud_labels_all), torch.from_numpy(
        weakly_labels_stack), torch.from_numpy(gt_labels_stack), torch.from_numpy(mask)


# ============================
# Define scannet dataset class
# ============================
class ScannetDataset(Dataset):
    def __init__(self, path, npoints, split='train'):
        super().__init__()
        self.label_to_names = {0: 'unclassified',
                               1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refridgerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'otherfurniture'}
        self.label_values = np.sort(np.sort([k for k, v in self.label_to_names.items()]))
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.data_root = path
        self.split = split
        self.block_size = 1.0
        self.npoints = npoints
        self.ignored_label = np.sort([0])
        self.num_pos = 0
        self.num_neg = 0

        if split == 'train':
            self.clouds_path = np.loadtxt(os.path.join(self.data_root, 'scannetv2_train.txt'), dtype=np.str)
        else:
            self.clouds_path = np.loadtxt(os.path.join(self.data_root, 'scannetv2_val.txt'), dtype=np.str)
        self.train_path = '/data/dataset/scannet/input_0.040'
        self.files = np.sort([os.path.join(self.train_path, f + '.ply') for f in self.clouds_path])

    def __getitem__(self, index):
        path = self.files[index]
        # print(path)
        data = read_ply(path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        num_points = points.shape[0]
        if num_points > 70000:
            self.num_neg += 1
        else:
            self.num_pos += 1
        # print("poins shape:", points.shape)
        colors = np.vstack((data['red'], data['green'], data['blue'])).T

        gt_label = data['class'].astype(np.int32)
        # print("gt label:", np.unique(gt_label))
        # gt_label = np.array([self.label_to_idx[i] for i in gt_label])
        mask = np.ones([self.npoints], dtype=np.bool)
        num_points = points.shape[0]

        if num_points > self.npoints:
            choice = np.random.choice(num_points, self.npoints, replace=True)
            points_set = points[choice, :]
            colors_set = colors[choice, :]
            gt_labels_set = gt_label[choice]
        else:
            points_set = np.zeros([self.npoints, points.shape[1]], dtype=np.float32)
            colors_set = np.zeros([self.npoints, colors.shape[1]], dtype=np.float32)
            gt_labels_set = np.zeros([self.npoints], dtype=np.int32)
            choice = np.random.choice(num_points, self.npoints - num_points, replace=True)
            points_set[:num_points, :] = points
            points_set[num_points:, :] = points[choice, :]
            colors_set[:num_points, :] = colors
            colors_set[num_points:, :] = colors[choice, :]
            gt_labels_set[:num_points] = gt_label
            gt_labels_set[num_points:] = gt_label[choice]
            mask[num_points:] = False

        points_all = np.zeros((self.npoints, 9))
        points_all[:, 0:3] = points_set
        points_all[:, 3:6] = colors_set
        points_all[:, 6:9] = pc_normalize(points_set)
        gt_labels_set_temp = np.array([self.label_to_idx[i] for i in gt_labels_set])
        cloud_labels_idx = np.unique(gt_labels_set_temp)
        cloud_labels_idx = cloud_labels_idx[cloud_labels_idx != 0].astype(np.int32)

        cloud_labels = np.zeros((1, 20))
        cloud_labels[0][cloud_labels_idx - 1] = 1
        cloud_labels_all = np.ones((points_all.shape[0], 20))
        cloud_labels_all = cloud_labels_all * cloud_labels

        # data = np.concatenate((points_set, colors_set), axis=1)
        data = points_all
        # print(data.dtype, cloud_labels_all.dtype, cloud_labels.dtype, gt_labels_set.dtype, mask.dtype)
        file_name = path.split('/')
        file_name = file_name[-1].split('.')[0]
        # print("gt_labels set unique:", np.unique(gt_labels_set))
        # print("cloud_labels shape:", cloud_labels)
        return (data.astype(np.float32), cloud_labels_all.astype(np.int32), cloud_labels.astype(np.int32),
                gt_labels_set.astype(np.int32), mask.astype(np.bool), file_name)

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    data = ScannetDataset('/data/dataset/scannet', npoints=40000, split='val')
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=20, shuffle=False)
    Max = -1
    Min = 200000
    total_files_name = []
    masks = []
    labels_total = []
    val_prob = np.zeros(20, dtype=np.float32)
    for points, cloud_labels_all, weakly_label, gt_label, mask, files_name in train_dataloader:
        total_files_name.append(files_name)
        masks.append(mask)
        labels_total += [gt_label]
        num_points = points.shape[0]

        print(points.shape, weakly_label.shape, gt_label.shape)
    num_pos = data.num_pos
    num_neg = data.num_neg
    print("num_pos:", num_pos)
    print("num_neg:", num_neg)
    print(num_pos / (num_pos + num_neg))
