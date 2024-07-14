import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from math import log, e
from nn import LSTMNetwork, ResNetwork
from dataset import PredictDataset2, predict_pad_collate
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
    phred_scores = -10 * np.log10(probs)
    return phred_scores


def predict(model, test_dataset, batch_size, output_file, device):
    model.eval()
    dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    fout = open(output_file, 'w')
    for batch in dl:
        positions, feature_matrices = batch
        feature_tensor = feature_matrices.type(torch.FloatTensor).to(device)

        zy_output_ = model.predict(feature_tensor)
        zy_output_ = zy_output_.detach().cpu().numpy()
        zy_prob = np.max(zy_output_, axis=1)  # [N]
        zy_output = np.argmax(zy_output_, axis=1)  # [N]
        zy_qual = calculate_phred_scores(1 - zy_prob)

        ## write to csv file
        ## position, zygosity, quality
        for i in range(len(positions)):
            fout.write(
                positions[i] + ',' + str(zy_output[i]) + ',' + str(zy_qual[i]) + ',' + str(
                    zy_output_[i][0]) + ',' + str(
                    zy_output_[i][1]) + ',' + str(zy_output_[i][2]) + ',' + str(zy_output_[i][3]) + '\n')
    fout.close()


def predict2(model, test_paths, batch_size, output_file, device):
    model.eval()
    fout = open(output_file, 'w')
    for file in test_paths:
        test_dataset = PredictDataset2(file)
        dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=predict_pad_collate)
        for batch in dl:
            positions, feature_matrices = batch
            feature_tensor = feature_matrices.type(torch.FloatTensor).to(device)
            feature_tensor = feature_tensor.permute(0, 3, 1, 2)  # [batch, dim, L, W]
            zy_output_ = model.predict(feature_tensor)
            zy_output_ = zy_output_.detach().cpu().numpy()
            zy_prob = np.max(zy_output_, axis=1)  # [N]
            zy_output = np.argmax(zy_output_, axis=1)  # [N]
            zy_qual = calculate_phred_scores(1 - zy_prob)

            ## write to csv file
            ## position, zygosity, quality
            for i in range(len(positions)):
                fout.write(
                    positions[i] + ',' + str(zy_output[i]) + ',' + str(zy_qual[i]) + ',' + str(
                        zy_output_[i][0]) + ',' + str(
                        zy_output_[i][1]) + ',' + str(zy_output_[i][2]) + ',' + str(zy_output_[i][3]) + '\n')
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='path to config file')
    parser.add_argument('-model_path', required=True, help='path to trained model')
    parser.add_argument('-data', required=True, help='directory of feature files')
    # parser.add_argument('-contig', required=True, help='contig name of the input bin files')
    parser.add_argument('-output', required=True, help='output vcf file')
    parser.add_argument('-batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    pred_model = ResNetwork(config.model).to(device)
    checkpoint = torch.load(opt.model_path, map_location=device)
    pred_model.resnet18.load_state_dict(checkpoint['resnet18'])
    testing_paths = [opt.data + '/' + fname for fname in os.listdir(opt.data) if fname.endswith('.npz')]
    # predict_dataset = PredictDataset(testing_paths, config.data.flanking_size)
    predict2(pred_model, testing_paths, opt.batch_size, opt.output, device)


if __name__ == '__main__':
    main()
