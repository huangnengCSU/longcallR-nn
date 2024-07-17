import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from math import log, e
from nn import ResNetwork
from dataset import EvalDataset2, eval_collate
from utils import AttrDict


def calculate_score(probability):
    p = probability
    tmp = max((-10 * log(e, 10)) * log(((1.0 - p) + 1e-300) / (p + 1e-300)) + 10, 0)
    return float(round(tmp, 2))


def calculate_phred_scores(probs):
    """
    Calculate Phred scores for an array of probabilities.

    Parameters:
    probs (array-like): An array of error probabilities.

    Returns:
    np.ndarray: An array of Phred quality scores.
    """
    # Convert probabilities to Phred scores using the formula Q = -10 * log10(P)
    phred_scores = -10 * np.log10(probs + 1e-300)
    return phred_scores


def eval(model, eval_dataset, batch_size, output_file, device):
    model.eval()
    dl = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    fout = open(output_file, 'w')
    for batch in dl:
        positions, feature_matrices, labels = batch
        feature_tensor = feature_matrices.type(torch.FloatTensor).to(device)

        zy_output_ = model.predict(feature_tensor)
        zy_output_ = zy_output_.detach().cpu().numpy()
        zy_prob = np.max(zy_output_, axis=1)  # [N]
        zy_output = np.argmax(zy_output_, axis=1)  # [N]
        zy_qual = calculate_phred_scores(1 - zy_prob)

        ## write to csv file
        ## position, zygosity, quality
        for i in range(len(positions)):
            if zy_output[i] != labels[i]:
                fout.write(
                    positions[i] + ',' + str(labels[i].item()) + ',' + str(zy_output[i]) + ',' + str(zy_qual[i]) + ','
                    + str(zy_output_[i][0]) + ',' + str(zy_output_[i][1]) + ',' + str(zy_output_[i][2]) + ','
                    + str(zy_output_[i][3]) + '\n')
    fout.close()


def eval2(model, eval_paths, batch_size, max_depth_threshold, output_file, device):
    model.eval()
    fout = open(output_file, 'w')
    for file in eval_paths:
        eval_dataset = EvalDataset2(file)
        dl = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: eval_collate(batch, max_depth_threshold=max_depth_threshold))
        for batch in dl:
            positions, feature_matrices, labels = batch
            feature_tensor = feature_matrices.type(torch.FloatTensor).to(device)
            feature_tensor = feature_tensor.permute(0, 3, 1, 2)  # [batch, ndim, L, W]

            zy_output_ = model.predict(feature_tensor)
            zy_output_ = zy_output_.detach().cpu().numpy()
            zy_prob = np.max(zy_output_, axis=1)  # [N]
            zy_output = np.argmax(zy_output_, axis=1)  # [N]
            zy_qual = calculate_phred_scores(1 - zy_prob)

            ## write to csv file
            ## position, zygosity, quality
            for i in range(len(positions)):
                if zy_output[i] != labels[i]:
                    fout.write(positions[i] + ',' + str(labels[i].item()) + ',' + str(zy_output[i]) + ',' + str(
                        zy_qual[i]) + ','
                               + str(zy_output_[i][0]) + ',' + str(zy_output_[i][1]) + ',' + str(zy_output_[i][2]) + ','
                               + str(zy_output_[i][3]) + '\n')
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='path to config file')
    parser.add_argument('-model', required=True, help='path to trained model')
    parser.add_argument('-data', required=True, help='directory of feature files')
    # parser.add_argument('-contig', required=True, help='contig name of the input bin files')
    parser.add_argument('-output', required=True, help='output vcf file')
    parser.add_argument('-batch_size', type=int, default=10, help='batch size')
    parser.add_argument('-max_depth', type=int, default=2000, help='max depth threshold')
    parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    pred_model = ResNetwork(config.model).to(device)
    checkpoint = torch.load(opt.model, map_location=device)
    pred_model.resnet.load_state_dict(checkpoint['resnet'])
    eval_paths = [opt.data + '/' + fname for fname in os.listdir(opt.data) if fname.endswith('.npz')]
    # eval_dataset = EvalDataset(eval_paths, config.data.flanking_size)
    eval2(pred_model, eval_paths, opt.batch_size, opt.max_depth, opt.output, device)


if __name__ == '__main__':
    main()
