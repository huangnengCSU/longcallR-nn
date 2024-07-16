import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import os


# import tables
# from options import gt_decoded_labels, zy_decoded_labels

# TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
# shuffle_bin_size = 50000
#
# no_flanking_bases = 16
# no_of_positions = 2 * no_flanking_bases + 1
# channel = ('A', 'C', 'G', 'T', 'I', 'I1', 'D', 'D1', '*',
#            'a', 'c', 'g', 't', 'i', 'i1', 'd', 'd1', '#')
# channel_size = len(channel)
# ont_input_shape = input_shape = [no_of_positions, channel_size]
# label_shape = [21, 3, no_of_positions, no_of_positions]
# label_size = sum(label_shape)
#
#
# def calculate_percentage(ts):
#     # ts: L, N, C
#     # return: L, N, 5
#     ts_A = np.expand_dims(((ts == 1).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
#     ts_C = np.expand_dims(((ts == 2).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
#     ts_G = np.expand_dims(((ts == 3).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
#     ts_T = np.expand_dims(((ts == 4).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
#     ts_D = np.expand_dims(((ts == -1).sum(2) / ((ts != -2).sum(2) + 1e-9)), 2)
#     return np.concatenate((ts_A, ts_C, ts_G, ts_T, ts_D), axis=2)


# def balance_dataset(gt_label, zy_label):
#     """
#     position_matrix.shape   (495703, 33, 18)
#     gt_label.shape  (495703,)
#     zy_label.shape  (495703,)
#     """
#     num_gts = len(gt_decoded_labels)
#     num_zys = len(zy_decoded_labels)
#     indexes_for_gt_zy = {}
#     num_of_max_categories = 0
#     for i in range(num_gts):
#         for j in range(num_zys):
#             findex = np.where((gt_label == i) & (zy_label == j))[0]
#             if len(findex) >= num_of_max_categories:
#                 num_of_max_categories = len(findex)
#             indexes_for_gt_zy[(i, j)] = findex
#
#     # 每个类上采样
#     non_zero_categories = 0
#     for k in indexes_for_gt_zy.keys():
#         size_of_categories = len(indexes_for_gt_zy[k])
#         if size_of_categories > 0 and size_of_categories < num_of_max_categories:
#             sample_num = num_of_max_categories - size_of_categories
#             sampling_index = np.random.choice(indexes_for_gt_zy[k], size=sample_num, replace=True)
#             indexes_for_gt_zy[k] = indexes_for_gt_zy[k].tolist() + sampling_index.tolist()
#             non_zero_categories += 1
#
#     total_indexes = []
#     for k in indexes_for_gt_zy.keys():
#         total_indexes.extend(indexes_for_gt_zy[k])
#
#     # 所有类的总和随机下采样
#     np.random.shuffle(total_indexes)
#     final_index = np.random.choice(total_indexes, size=int(len(total_indexes) / non_zero_categories))
#     return final_index
#
#
# def filter_sample(position_matrix, gt_label, zy_label):
#     pass


# class TrainDataset(Dataset):
#     def __init__(self, datapath, use_balance=False, for_evaluate=False):
#         # tables.set_blosc_max_threads(16)
#         table_file = tables.open_file(datapath, 'r')
#         position_matrix = np.array(table_file.root.position_matrix)  # [N,33,18]
#         label = table_file.root.label  # [N,90]
#         gt_label = np.array(label[:, :21].argmax(1))  # [N]
#         zy_label = np.array(label[:, 21:24].argmax(1))  # [N]
#         indel1_label = np.array(label[:, 24:57].argmax(1))  # [N]
#         indel2_label = np.array(label[:, 57:].argmax(1))  # [N]
#         table_file.close()
#         if use_balance:
#             train_idx = balance_dataset(gt_label=gt_label, zy_label=zy_label)
#             self.position_matrix = position_matrix[train_idx]
#             self.gt_label = gt_label[train_idx]
#             self.zy_label = zy_label[train_idx]
#             self.indel1_label = indel1_label[train_idx]
#             self.indel2_label = indel2_label[train_idx]
#         else:
#             self.position_matrix = position_matrix
#             self.gt_label = gt_label
#             self.zy_label = zy_label
#             self.indel1_label = indel1_label
#             self.indel2_label = indel2_label
#
#         if for_evaluate:
#             variant_idx = np.where(self.zy_label > 0)[0]  # zy_decoded_labels = ['0/0', '1/1', '0/1']
#             self.position_matrix = self.position_matrix[variant_idx]
#             self.gt_label = self.gt_label[variant_idx]
#             self.zy_label = self.zy_label[variant_idx]
#             self.indel1_label = self.indel1_label[variant_idx]
#             self.indel2_label = self.indel2_label[variant_idx]
#
#     def __getitem__(self, i):
#         position_matrix = self.position_matrix[i]
#         gt_label = self.gt_label[i]
#         zy_label = self.zy_label[i]
#         indel1_label = self.indel1_label[i]
#         indel2_label = self.indel2_label[i]
#         return position_matrix, gt_label, zy_label, indel1_label, indel2_label
#
#     def __len__(self):
#         return len(self.gt_label)


def sort_3d_array(feature_matrix):
    col_idx = int((feature_matrix.shape[0] - 1) / 2)
    slice_to_sort = feature_matrix[col_idx, :, 1]
    sorted_indices = np.argsort(-slice_to_sort)
    feature_matrix = feature_matrix[:, sorted_indices, :]
    return feature_matrix


class TrainDataset(Dataset):
    def __init__(self, data_paths, flanking_size):
        ## data_paths: list of file paths
        training_matrix = []
        label = []
        for datapath in data_paths:
            with open(datapath, 'r') as fin:
                single_feature_matrix = []
                idx = 0
                for line in fin:
                    idx += 1
                    if idx <= flanking_size * 2 + 1:
                        line = [float(v) for v in line.strip().rstrip(',').split(',')]
                        single_feature_matrix.append(line)
                    elif idx == flanking_size * 2 + 2:
                        label.append(int(line.strip().split('\t')[0]))
                        training_matrix.append(single_feature_matrix)
                        single_feature_matrix = []
                        idx = 0
        assert len(label) > 0
        nfeatures = len(training_matrix[0][0])
        self.training_matrix = np.array(training_matrix).reshape(-1, flanking_size * 2 + 1, nfeatures)  # [N,41,38]
        assert len(label) == len(self.training_matrix)
        self.label = np.array(label)  # [N,]

    def __getitem__(self, i):
        feature_matrix = self.training_matrix[i]
        label = self.label[i]
        return feature_matrix, label

    def __len__(self):
        return len(self.label)


class TrainDataset2(Dataset):
    def __init__(self, datapath, max_depth=1000):
        data = np.load(datapath)
        feature_positions = data.files

        ## filter out the data with depth > max_depth
        indices = []
        for i, feature_pos in enumerate(feature_positions):
            if data[feature_pos].shape[1] <= max_depth:
                indices.append(i)

        labelpath = os.path.splitext(datapath)[0] + '.label'
        label_positions = []
        labels = []
        with open(labelpath, 'r') as fin:
            for line in fin:
                fields = line.strip().split('\t')
                labels.append(int(fields[0]))
                label_positions.append(fields[1])

        feature_positions = [feature_positions[i] for i in indices]
        labels = [labels[i] for i in indices]
        label_positions = [label_positions[i] for i in indices]

        self.data = data
        self.feature_positions = feature_positions
        self.labels = labels
        self.label_positions = label_positions

    def __getitem__(self, i):
        feature_pos = self.feature_positions[i]
        feature_matrix = self.data[feature_pos]  # [flanking_size * 2 + 1, depth, nfeatures], depth not fixed
        label_pos = self.label_positions[i]
        label = self.labels[i]
        assert feature_pos == label_pos
        return feature_matrix, label

    def __len__(self):
        return len(self.labels)


class TrainDataset3(Dataset):
    def __init__(self, data_folder, max_depth=1000):

        all_data = {}
        all_feature_positions = []
        all_labels = []
        all_label_positions = []

        for fn in os.listdir(data_folder):
            if not fn.endswith('.npz'):
                continue
            datapath = data_folder + '/' + fn
            data = np.load(datapath)
            feature_positions = data.files

            ## filter out the data with depth > max_depth
            indices = []
            for i, feature_pos in enumerate(feature_positions):
                if data[feature_pos].shape[1] <= max_depth:
                    indices.append(i)

            labelpath = os.path.splitext(datapath)[0] + '.label'
            label_positions = []
            labels = []
            with open(labelpath, 'r') as fin:
                for line in fin:
                    fields = line.strip().split('\t')
                    labels.append(int(fields[0]))
                    label_positions.append(fields[1])

            all_feature_positions.extend([feature_positions[i] for i in indices])
            all_labels.extend([labels[i] for i in indices])
            all_label_positions.extend([label_positions[i] for i in indices])
            for i in indices:
                all_data[feature_positions[i]] = data[feature_positions[i]]

        self.data = all_data
        self.feature_positions = all_feature_positions
        self.labels = all_labels
        self.label_positions = all_label_positions

    def __getitem__(self, i):
        feature_pos = self.feature_positions[i]
        feature_matrix = self.data[feature_pos]  # [flanking_size * 2 + 1, depth, nfeatures], depth not fixed
        label_pos = self.label_positions[i]
        label = self.labels[i]
        assert feature_pos == label_pos
        return feature_matrix, label

    def __len__(self):
        return len(self.labels)


class EvalDataset(Dataset):
    def __init__(self, data_paths, flanking_size):
        ## data_paths: list of file paths
        training_matrix = []
        label = []
        positions = []
        for datapath in data_paths:
            with open(datapath, 'r') as fin:
                single_feature_matrix = []
                idx = 0
                for line in fin:
                    idx += 1
                    if idx <= flanking_size * 2 + 1:
                        line = [float(v) for v in line.strip().rstrip(',').split(',')]
                        single_feature_matrix.append(line)
                    elif idx == flanking_size * 2 + 2:
                        label.append(int(line.strip().split('\t')[0]))
                        positions.append(line.strip().split('\t')[1])
                        training_matrix.append(single_feature_matrix)
                        single_feature_matrix = []
                        idx = 0
        assert len(label) > 0
        nfeatures = len(training_matrix[0][0])
        self.training_matrix = np.array(training_matrix).reshape(-1, flanking_size * 2 + 1, nfeatures)  # [N,41,38]
        assert len(label) == len(self.training_matrix)
        self.label = np.array(label)  # [N,]
        self.positions = positions  # [N,]

    def __getitem__(self, i):
        feature_matrix = self.training_matrix[i]
        label = self.label[i]
        position = self.positions[i]
        return position, feature_matrix, label

    def __len__(self):
        return len(self.label)


class EvalDataset2(Dataset):
    def __init__(self, datapath):
        data = np.load(datapath)
        feature_positions = data.files

        labelpath = os.path.splitext(datapath)[0] + '.label'
        label_positions = []
        labels = []
        with open(labelpath, 'r') as fin:
            for line in fin:
                fields = line.strip().split('\t')
                labels.append(int(fields[0]))
                label_positions.append(fields[1])

        self.data = data
        self.feature_positions = feature_positions
        self.labels = labels
        self.label_positions = label_positions

    def __getitem__(self, i):
        feature_pos = self.feature_positions[i]
        feature_matrix = sort_3d_array(
            self.data[feature_pos])  # [flanking_size * 2 + 1, depth, nfeatures], depth not fixed
        label_pos = self.label_positions[i]
        label = self.labels[i]
        assert feature_pos == label_pos
        return feature_pos, feature_matrix, label

    def __len__(self):
        return len(self.labels)


# class PredictDataset(Dataset):
#     def __init__(self, datapath):
#         # tables.set_blosc_max_threads(16)
#         table_file = tables.open_file(datapath, 'r')
#         position_matrix = np.array(table_file.root.position_matrix)  # [N,33,18]
#         position = table_file.root.position
#         positions = []
#         reference_bases = []
#         contig_names = []
#         for item in position:
#             ctg_name, pos, seq = str(item[0], encoding="utf-8").strip().split(':')
#             contig_names.append(ctg_name)
#             pos = int(pos)
#             positions.append(pos)
#             reference_bases.append(ord(seq[16]))  # ASCII
#         positions = np.array(positions)
#         reference_bases = np.array(reference_bases)
#         table_file.close()
#         self.contig_names = contig_names
#         self.position_matrix = position_matrix
#         self.positions = positions
#         self.reference_bases = reference_bases
#
#     def __getitem__(self, i):
#         contig_names = self.contig_names[i]
#         positions = self.positions[i]
#         reference_bases = self.reference_bases[i]
#         position_matrix = self.position_matrix[i]
#         return contig_names, positions, reference_bases, position_matrix
#
#     def __len__(self):
#         return len(self.position_matrix)


class PredictDataset(Dataset):
    def __init__(self, data_paths, flanking_size):
        predict_matrix = []
        positions = []
        for datapath in data_paths:
            with open(datapath, 'r') as fin:
                idx = 0
                single_feature_matrix = []
                for line in fin:
                    idx += 1
                    if idx <= flanking_size * 2 + 1:
                        line = [float(v) for v in line.strip().rstrip(',').split(',')]
                        single_feature_matrix.append(line)
                    elif idx == flanking_size * 2 + 2:
                        positions.append(line.strip())
                        predict_matrix.append(single_feature_matrix)
                        single_feature_matrix = []
                        idx = 0
        assert len(positions) > 0
        nfeatures = len(predict_matrix[0][0])
        self.predict_matrix = np.array(predict_matrix).reshape(-1, flanking_size * 2 + 1, nfeatures)  # [N,41,38]
        assert len(positions) == len(self.predict_matrix)
        self.positions = positions  # [N,]

    def __getitem__(self, i):
        position = self.positions[i]
        feature_matrix = self.predict_matrix[i]
        return position, feature_matrix

    def __len__(self):
        return len(self.predict_matrix)


class PredictDataset2(Dataset):
    def __init__(self, datapath):
        data = np.load(datapath)
        feature_positions = data.files
        self.data = data
        self.feature_positions = feature_positions

    def __getitem__(self, i):
        feature_pos = self.feature_positions[i]
        feature_matrix = sort_3d_array(
            self.data[feature_pos])  # [flanking_size * 2 + 1, depth, nfeatures], depth not fixed
        return feature_pos, feature_matrix

    def __len__(self):
        return len(self.feature_positions)


def train_pad_collate(batch, max_depth_threshold=10000):
    # Separate data and labels
    data, labels = zip(*batch)

    # Find the max depth in the batch
    max_depth = min(max([x.shape[1] for x in data]), max_depth_threshold)

    # Pad sequences to the same depth
    padded_data = []
    for x in data:
        depth_padding = max_depth - x.shape[1]
        padded_x = F.pad(torch.tensor(x), (0, 0, 0, depth_padding))
        padded_data.append(padded_x)

    # Stack them into a tensor
    padded_data = torch.stack(padded_data)
    labels = torch.tensor(labels)

    return padded_data, labels


def eval_pad_collate(batch, max_depth_threshold=10000):
    # Separate data and labels
    pos, data, labels = zip(*batch)

    # Find the max depth in the batch
    max_depth = min(max([x.shape[1] for x in data]), max_depth_threshold)

    # Pad sequences to the same depth
    padded_data = []
    for x in data:
        depth_padding = max_depth - x.shape[1]
        padded_x = F.pad(torch.tensor(x), (0, 0, 0, depth_padding))
        padded_data.append(padded_x)

    # Stack them into a tensor
    padded_data = torch.stack(padded_data)
    labels = torch.tensor(labels)

    return pos, padded_data, labels


# def predict_pad_collate(batch, max_depth_threshold=10000):
#     # Separate data and labels
#     pos, data = zip(*batch)
#
#     # Find the max depth in the batch
#     max_depth = min(max([x.shape[1] for x in data]), max_depth_threshold)
#
#     # Pad sequences to the same depth
#     padded_data = []
#     for x in data:
#         depth_padding = max_depth - x.shape[1]
#         padded_x = F.pad(torch.tensor(x), (0, 0, 0, depth_padding))
#         padded_data.append(padded_x)
#
#     # Stack them into a tensor
#     padded_data = torch.stack(padded_data)
#
#     return pos, padded_data

def predict_pad_collate(batch):
    # Separate data and labels
    pos, data = zip(*batch)

    # Find the max depth in the batch
    max_depth = max([x.shape[1] for x in data])

    # Pad sequences to the same depth
    padded_data = []
    for x in data:
        depth_padding = max_depth - x.shape[1]
        padded_x = F.pad(torch.tensor(x), (0, 0, 0, depth_padding))
        padded_data.append(padded_x)

    # Stack them into a tensor
    padded_data = torch.stack(padded_data)

    return pos, padded_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True, help='directory of bin files')
    opt = parser.parse_args()

    epoch = 0
    # filepaths = [opt.data + '/' + file for file in os.listdir(opt.data) if file.endswith('.npz')]
    # for file in filepaths:
    #     dataset = TrainDataset2(datapath=file)
    #     while epoch < 1:
    #         dl = DataLoader(dataset, batch_size=500, shuffle=True, collate_fn=train_pad_collate)
    #         for batch in dl:
    #             feature_matrices, labels = batch
    #             print("Epoch ", epoch, ":", feature_matrices.shape)
    #         epoch += 1

    # filepaths = [opt.data + '/' + file for file in os.listdir(opt.data) if file.endswith('.npz')]
    # for file in filepaths:
    #     dataset = EvalDataset2(datapath=file)
    #     while epoch < 1:
    #         dl = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=eval_pad_collate)
    #         for batch in dl:
    #             positions, feature_matrices, labels = batch
    #             print("Epoch ", epoch, ":", feature_matrices.shape)
    #         epoch += 1

    # filepaths = [opt.data + '/' + file for file in os.listdir(opt.data) if file.endswith('.npz')]
    # for file in filepaths:
    #     dataset = PredictDataset2(datapath=file)
    #     while epoch < 1:
    #         dl = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=predict_pad_collate)
    #         for batch in dl:
    #             positions, feature_matrices = batch
    #             print("Epoch ", epoch, ":", feature_matrices.shape)
    #         epoch += 1

    dataset = TrainDataset3(data_folder=opt.data, max_depth=200)
    while epoch < 2:
        dl = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=train_pad_collate)
        for batch in dl:
            feature_matrices, labels = batch
            print("Epoch ", epoch, ":", feature_matrices.shape)
        epoch += 1
