import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from nn import ResNetwork
from dataset import PredictDataset2, predict_collate
from utils import AttrDict
from vcf import write_vcf_header
import math

GT_MAP = {
    0: "AA",
    1: "AC",
    2: "AG",
    3: "AT",
    4: "CA",
    5: "CC",
    6: "CG",
    7: "CT",
    8: "GA",
    9: "GC",
    10: "GG",
    11: "GT",
    12: "TA",
    13: "TC",
    14: "TG",
    15: "TT"
}

ZY_MAP = {
    0: "0/0",
    1: "0/1",
    2: "1/1",
    3: "1/2",
    4: "0/0"
}


def calculate_score(probabilities):
    scores = []
    for probability in probabilities:
        p = probability
        tmp = max((-10 * math.log10(math.e)) * math.log(((1.0 - p) + 1e-10) / (p + 1e-10)) + 10, 0)
        scores.append(float(round(tmp, 2)))
    return scores


# def calculate_phred_scores(probs):
#     """
#     Calculate Phred scores for an array of probabilities.
#
#     Parameters:
#     probs (array-like): An array of error probabilities.
#
#     Returns:
#     np.ndarray: An array of Phred quality scores.
#     """
#     # Convert probabilities to Phred scores using the formula Q = -10 * log10(P)
#     phred_scores = -10 * np.log10(probs + 1e-30)
#     return phred_scores


# def predict(model, test_dataset, batch_size, output_file, device):
#     model.eval()
#     dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     fout = open(output_file, 'w')
#     for batch in dl:
#         positions, feature_matrices = batch
#         feature_tensor = feature_matrices.type(torch.FloatTensor).to(device)
#
#         zy_output_ = model.predict(feature_tensor)
#         zy_output_ = zy_output_.detach().cpu().numpy()
#         zy_prob = np.max(zy_output_, axis=1)  # [N]
#         zy_output = np.argmax(zy_output_, axis=1)  # [N]
#         zy_qual = calculate_score(zy_prob)
#         # zy_qual = calculate_phred_scores(1 - zy_prob)
#
#         ## write to csv file
#         ## position, zygosity, quality
#         for i in range(len(positions)):
#             fout.write(
#                 positions[i] + ',' + str(zy_output[i]) + ',' + str(zy_qual[i]) + ',' + str(
#                     zy_output_[i][0]) + ',' + str(
#                     zy_output_[i][1]) + ',' + str(zy_output_[i][2]) + ',' + str(zy_output_[i][3]) + '\n')
#     fout.close()


def predict2(model, test_paths, batch_size, max_depth_threshold, output_file, device):
    model.eval()
    write_vcf_header(output_file)
    fout = open(output_file, 'a')
    for file in test_paths:
        test_dataset = PredictDataset2(file)
        dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: predict_collate(batch, max_depth_threshold=max_depth_threshold))
        for batch in dl:
            positions, feature_matrices = batch
            feature_tensor = feature_matrices.type(torch.FloatTensor).to(device)
            feature_tensor = feature_tensor.permute(0, 3, 1, 2)  # [batch, dim, L, W]
            zy_output_, gt_output_ = model.predict(feature_tensor)
            zy_output_ = zy_output_.detach().cpu().numpy()
            zy_prob = np.max(zy_output_, axis=1)  # [N]
            zy_output = np.argmax(zy_output_, axis=1)  # [N]
            zy_qual = calculate_score(zy_prob)  # [N]
            gt_output_ = gt_output_.detach().cpu().numpy()
            gt_output = np.argmax(gt_output_, axis=1)
            gt_prob = np.max(gt_output_, axis=1)
            gt_qual = calculate_score(gt_prob)

            ## write to vcf file
            ## position, zygosity, quality
            for i in range(len(positions)):
                chr = positions[i].split(':')[0]
                pos = positions[i].split(':')[1]
                if zy_output[i] == 0:
                    gt = "0/0"
                    [ref_base, alt_base] = GT_MAP[gt_output[i]]
                    if ref_base != alt_base:
                        alt_base = ref_base
                elif zy_output[i] == 1:
                    gt = "0/1"
                    [ref_base, alt_base] = GT_MAP[gt_output[i]]
                    if ref_base == alt_base:
                        gt = "0/0"
                elif zy_output[i] == 2:
                    gt = "1/1"
                    [ref_base, alt_base] = GT_MAP[gt_output[i]]
                    if ref_base == alt_base:
                        gt = "0/0"
                elif zy_output[i] == 3:
                    gt = "1/2"
                    continue
                elif zy_output[i] == 4:
                    gt = "0/0"
                    [ref_base, alt_base] = GT_MAP[gt_output[i]]
                    if GT_MAP[gt_output[i]] != "AG" and GT_MAP[gt_output[i]] != "TC":
                        print("Warning: {}:{}\t GT:{}\tZY:{}".format(
                            chr, pos, GT_MAP[gt_output[i]], ZY_MAP[zy_output[i]]))
                        continue
                    alt_base = ref_base
                qual = min(zy_qual[i], gt_qual[i])
                if zy_output[i] == 1 or zy_output[i] == 2 or zy_output[i] == 3:
                    fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        chr, pos, '.', ref_base, alt_base, qual, 'PASS', '.', 'GT', gt))
                elif zy_output[i] == 0:
                    fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        chr, pos, '.', ref_base, alt_base, qual, 'RefCall', '.', 'GT', gt))
                elif zy_output[i] == 4:
                    fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        chr, pos, '.', ref_base, ref_base, qual, 'RnaEdit', '.', 'GT', gt))
                else:
                    raise ValueError("Unexpected zygosity output")
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='path to config file')
    parser.add_argument('-model', required=True, help='path to trained model')
    parser.add_argument('-data', required=True, help='directory of feature files')
    # parser.add_argument('-contig', required=True, help='contig name of the input bin files')
    parser.add_argument('-output', required=True, help='output vcf file')
    parser.add_argument('-max_depth', type=int, default=2000, help='max depth threshold')
    parser.add_argument('-batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    pred_model = ResNetwork(config.model).to(device)
    checkpoint = torch.load(opt.model, map_location=device)
    if config.model.spp:
        pred_model.resnet.load_state_dict(checkpoint['resnet'])
        pred_model.spp.load_state_dict(checkpoint['spp'])
        pred_model.zy_fc.load_state_dict(checkpoint['zy_fc'])
        pred_model.gt_fc.load_state_dict(checkpoint['gt_fc'])
    else:
        pred_model.resnet.load_state_dict(checkpoint['resnet'])
        pred_model.zy_fc.load_state_dict(checkpoint['zy_fc'])
        pred_model.gt_fc.load_state_dict(checkpoint['gt_fc'])
    testing_paths = [opt.data + '/' + fname for fname in os.listdir(opt.data) if fname.endswith('.npz')]
    # predict_dataset = PredictDataset(testing_paths, config.data.flanking_size)
    predict2(pred_model, testing_paths, opt.batch_size, opt.max_depth, opt.output, device)


if __name__ == '__main__':
    main()
