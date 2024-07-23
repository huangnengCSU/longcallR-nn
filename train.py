import os
import shutil
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from nn import ResNetwork
from dataset import TrainDataset3, train_collate
from utils import AttrDict, init_logger, count_parameters, save_model2
from tensorboardX import SummaryWriter
from torchmetrics import Accuracy, Recall, Precision, F1Score, ConfusionMatrix

torch.set_printoptions(
    precision=2,  # 精度，保留小数点后几位，默认4
    threshold=1000,
    edgeitems=3,
    linewidth=300,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=False  # 用科学技术法显示数据，默认True
)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# def train(epoch, config, model, train_dataset, batch_size, optimizer, logger, visualizer=None):
#     model.train()
#     start = time.process_time()
#     total_loss = 0
#     total_images = 0
#     optimizer.epoch()
#     zy_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_class)
#     zy_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_class)
#     zy_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_class)
#     dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
#     for batch in dl:
#         feature_tensor, zygosity_label = batch
#         feature_tensor = feature_tensor.type(torch.FloatTensor)  # [batch, 2*flanking_size+1, dim]
#         zygosity_label = zygosity_label.type(torch.LongTensor)  # [batch,]
#         if config.training.num_gpu > 0:
#             feature_tensor = feature_tensor.cuda()
#             zygosity_label = zygosity_label.cuda()
#
#         loss, zy_out = model(feature_tensor, zygosity_label)
#         optimizer.zero_grad()
#         loss.backward()
#         total_loss += loss.item()
#         grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
#         optimizer.step()
#         total_images += feature_tensor.shape[0]
#         zy_out = zy_out.cpu().data.contiguous().view(-1, config.model.num_class)
#         zygosity_label = zygosity_label.cpu().data.contiguous().view(-1)
#         zy_acc_metric.update(zy_out, zygosity_label)
#         zy_f1_metric.update(zy_out, zygosity_label)
#         zy_conf_metric.update(zy_out, zygosity_label)
#
#         print('\r-Training-Epoch:%d, Global Step:%d | Zygosity Accuracy:%.5f F1-Score:%.5f' % (
#             epoch, optimizer.global_step, float(zy_acc_metric.compute()), float(zy_f1_metric.compute())), end="",
#               flush=True)
#     if visualizer is not None:
#         visualizer.add_scalar('train_loss', loss.item(), optimizer.global_step)
#         visualizer.add_scalar('learn_rate', optimizer.lr, optimizer.global_step)
#     batch_zy_acc = float(zy_acc_metric.compute())
#     batch_zy_f1 = float(zy_f1_metric.compute())
#     print('\r-Training-Epoch:%d, Global Step:%d | Zygosity Accuracy:%.5f F1-Score:%.5f' % (
#         epoch, optimizer.global_step, batch_zy_acc, batch_zy_f1), end="", flush=True)
#     print()
#
#
# def train2(epoch, config, model, train_paths, batch_size, optimizer, logger, visualizer=None):
#     model.train()
#     start = time.process_time()
#     total_loss = 0
#     total_images = 0
#     optimizer.epoch()
#     zy_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_class)
#     zy_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_class)
#     zy_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_class)
#     for file in train_paths:
#         train_dataset = TrainDataset2(file, config.data.max_depth)
#         dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate)
#         for batch in dl:
#             feature_tensor, zygosity_label = batch
#             feature_tensor = feature_tensor.type(torch.FloatTensor)  # [batch, 2*flanking_size+1, dim]
#             feature_tensor = feature_tensor.permute(0, 3, 1, 2)  # [batch, dim, L, W]
#             zygosity_label = zygosity_label.type(torch.LongTensor)  # [batch,]
#             if config.training.num_gpu > 0:
#                 feature_tensor = feature_tensor.cuda()
#                 zygosity_label = zygosity_label.cuda()
#
#             loss, zy_out = model(feature_tensor, zygosity_label)
#             optimizer.zero_grad()
#             loss.backward()
#             total_loss += loss.item()
#             grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
#             optimizer.step()
#             total_images += feature_tensor.shape[0]
#             zy_out = zy_out.cpu().data.contiguous().view(-1, config.model.num_class)
#             zygosity_label = zygosity_label.cpu().data.contiguous().view(-1)
#             zy_acc_metric.update(zy_out, zygosity_label)
#             zy_f1_metric.update(zy_out, zygosity_label)
#             zy_conf_metric.update(zy_out, zygosity_label)
#
#             print('\r-Training-Epoch:%d, Global Step:%d | Zygosity Accuracy:%.5f F1-Score:%.5f' % (
#                 epoch, optimizer.global_step, float(zy_acc_metric.compute()), float(zy_f1_metric.compute())), end="",
#                   flush=True)
#     if visualizer is not None:
#         visualizer.add_scalar('train_loss', loss.item(), optimizer.global_step)
#         visualizer.add_scalar('learn_rate', optimizer.lr, optimizer.global_step)
#     batch_zy_acc = float(zy_acc_metric.compute())
#     batch_zy_f1 = float(zy_f1_metric.compute())
#     print('\r-Training-Epoch:%d, Global Step:%d | Zygosity Accuracy:%.5f F1-Score:%.5f' % (
#         epoch, optimizer.global_step, batch_zy_acc, batch_zy_f1), end="", flush=True)
#     print()


def train3(epoch, global_step, config, model, train_dataset, batch_size, optimizer, logger, visualizer=None):
    model.train()
    total_loss = 0
    total_images = 0
    zy_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_zy_class)
    zy_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_zy_class)
    zy_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_zy_class)
    gt_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_gt_class)
    gt_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_gt_class)
    gt_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_gt_class)
    dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda batch: train_collate(batch, max_depth_threshold=config.data.max_depth))
    for batch in dl:
        feature_tensor, zygosity_label, genotype_label = batch
        feature_tensor = feature_tensor.type(torch.FloatTensor)  # [batch, 2*flanking_size+1, dim]
        feature_tensor = feature_tensor.permute(0, 3, 1, 2)  # [batch, dim, L, W]
        zygosity_label = zygosity_label.type(torch.LongTensor)  # [batch,]
        genotype_label = genotype_label.type(torch.LongTensor)  # [batch,]
        if config.training.num_gpu > 0:
            feature_tensor = feature_tensor.cuda()
            zygosity_label = zygosity_label.cuda()
            genotype_label = genotype_label.cuda()

        loss, zy_out, gt_out = model(feature_tensor, zygosity_label, genotype_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        global_step += 1
        total_loss += loss.item()
        total_images += feature_tensor.shape[0]
        zy_out = zy_out.cpu().data.contiguous().view(-1, config.model.num_zy_class)
        zygosity_label = zygosity_label.cpu().data.contiguous().view(-1)
        zy_acc_metric.update(zy_out, zygosity_label)
        zy_f1_metric.update(zy_out, zygosity_label)
        zy_conf_metric.update(zy_out, zygosity_label)
        gt_out = gt_out.cpu().data.contiguous().view(-1, config.model.num_gt_class)
        genotype_label = genotype_label.cpu().data.contiguous().view(-1)
        gt_acc_metric.update(gt_out, genotype_label)
        gt_f1_metric.update(gt_out, genotype_label)
        gt_conf_metric.update(gt_out, genotype_label)

        print(
            '\r-Training-Epoch:%d, Global Step:%d | Zygosity Accuracy:%.5f F1-Score:%.5f | Genotype Accuracy:%.5f F1-Score:%.5f' % (
                epoch, global_step, float(zy_acc_metric.compute()), float(zy_f1_metric.compute()),
                float(gt_acc_metric.compute()), float(gt_f1_metric.compute())), end="", flush=True)
    if visualizer is not None:
        visualizer.add_scalar('train_loss', loss.item(), global_step)
        visualizer.add_scalar('learn_rate', get_lr(optimizer), global_step)
    batch_zy_acc = float(zy_acc_metric.compute())
    batch_zy_f1 = float(zy_f1_metric.compute())
    batch_gt_acc = float(gt_acc_metric.compute())
    batch_gt_f1 = float(gt_f1_metric.compute())
    print(
        '\r-Training-Epoch:%d, Global Step:%d | Zygosity Accuracy:%.5f F1-Score:%.5f | Genotype Accuracy:%.5f F1-Score:%.5f' % (
            epoch, global_step, batch_zy_acc, batch_zy_f1, batch_gt_acc, batch_gt_f1), end="", flush=True)
    print()
    return global_step


# def eval(epoch, config, model, validate_dataset, batch_size, logger, visualizer=None):
#     model.eval()
#     total_loss = 0
#     total_images = 0
#     zy_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_class)
#     zy_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_class)
#     zy_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_class)
#     dl = DataLoader(validate_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
#     for batch in dl:
#         feature_tensor, zygosity_label = batch
#         feature_tensor = feature_tensor.type(torch.FloatTensor)  # [batch, 33, dim]
#         zygosity_label = zygosity_label.type(torch.LongTensor)  # [batch,]
#         if config.training.num_gpu > 0:
#             feature_tensor = feature_tensor.cuda()
#             zygosity_label = zygosity_label.cuda()
#
#         loss, zy_out = model(feature_tensor, zygosity_label)
#         total_loss += loss.item()
#         total_images += feature_tensor.shape[0]
#
#         zy_out = zy_out.cpu().data.contiguous().view(-1, config.model.num_class)
#         zygosity_label = zygosity_label.cpu().data.contiguous().view(-1)
#         zy_acc_metric.update(zy_out, zygosity_label)
#         zy_f1_metric.update(zy_out, zygosity_label)
#         zy_conf_metric.update(zy_out, zygosity_label)
#
#     avg_loss = total_loss / total_images
#     batch_zy_acc = float(zy_acc_metric.compute())
#     batch_zy_f1 = float(zy_f1_metric.compute())
#     if visualizer is not None:
#         visualizer.add_scalar('eval_loss', avg_loss, epoch)
#         visualizer.add_scalar('zy_accuracy', batch_zy_acc, epoch)
#         visualizer.add_scalar('zy_f1score', batch_zy_f1, epoch)
#     print('-Validating-Epoch:%d, | Zygosity Accuracy:%.5f F1-Score:%.5f' % (epoch, batch_zy_acc, batch_zy_f1))
#     print(zy_conf_metric.compute())
#
#
# def eval2(epoch, config, model, validate_paths, batch_size, logger, visualizer=None):
#     model.eval()
#     total_loss = 0
#     total_images = 0
#     zy_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_class)
#     zy_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_class)
#     zy_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_class)
#     for file in validate_paths:
#         validate_dataset = TrainDataset2(file, config.data.max_depth)
#         dl = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collate)
#         for batch in dl:
#             feature_tensor, zygosity_label = batch
#             feature_tensor = feature_tensor.type(torch.FloatTensor)  # [batch, 33, dim]
#             feature_tensor = feature_tensor.permute(0, 3, 1, 2)  # [batch, dim, L, W]
#             zygosity_label = zygosity_label.type(torch.LongTensor)  # [batch,]
#             if config.training.num_gpu > 0:
#                 feature_tensor = feature_tensor.cuda()
#                 zygosity_label = zygosity_label.cuda()
#
#             loss, zy_out = model(feature_tensor, zygosity_label)
#             total_loss += loss.item()
#             total_images += feature_tensor.shape[0]
#
#             zy_out = zy_out.cpu().data.contiguous().view(-1, config.model.num_class)
#             zygosity_label = zygosity_label.cpu().data.contiguous().view(-1)
#             zy_acc_metric.update(zy_out, zygosity_label)
#             zy_f1_metric.update(zy_out, zygosity_label)
#             zy_conf_metric.update(zy_out, zygosity_label)
#
#     avg_loss = total_loss / total_images
#     batch_zy_acc = float(zy_acc_metric.compute())
#     batch_zy_f1 = float(zy_f1_metric.compute())
#     if visualizer is not None:
#         visualizer.add_scalar('eval_loss', avg_loss, epoch)
#         visualizer.add_scalar('zy_accuracy', batch_zy_acc, epoch)
#         visualizer.add_scalar('zy_f1score', batch_zy_f1, epoch)
#     print('-Validating-Epoch:%d, | Zygosity Accuracy:%.5f F1-Score:%.5f' % (epoch, batch_zy_acc, batch_zy_f1))
#     print(zy_conf_metric.compute())


def eval3(epoch, config, model, validate_dataset, batch_size, logger, visualizer=None):
    model.eval()
    total_loss = 0
    total_images = 0
    zy_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_zy_class)
    zy_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_zy_class)
    zy_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_zy_class)
    gt_acc_metric = Accuracy(task='multiclass', num_classes=config.model.num_gt_class)
    gt_f1_metric = F1Score(task='multiclass', num_classes=config.model.num_gt_class)
    gt_conf_metric = ConfusionMatrix(task='multiclass', num_classes=config.model.num_gt_class)
    dl = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda batch: train_collate(batch, max_depth_threshold=config.data.max_depth))
    for batch in dl:
        feature_tensor, zygosity_label, genotype_label = batch
        feature_tensor = feature_tensor.type(torch.FloatTensor)  # [batch, 33, dim]
        feature_tensor = feature_tensor.permute(0, 3, 1, 2)  # [batch, dim, L, W]
        zygosity_label = zygosity_label.type(torch.LongTensor)  # [batch,]
        genotype_label = genotype_label.type(torch.LongTensor)  # [batch,]
        if config.training.num_gpu > 0:
            feature_tensor = feature_tensor.cuda()
            zygosity_label = zygosity_label.cuda()
            genotype_label = genotype_label.cuda()

        loss, zy_out, gt_out = model(feature_tensor, zygosity_label, genotype_label)
        total_loss += loss.item()
        total_images += feature_tensor.shape[0]

        zy_out = zy_out.cpu().data.contiguous().view(-1, config.model.num_zy_class)
        zygosity_label = zygosity_label.cpu().data.contiguous().view(-1)
        zy_acc_metric.update(zy_out, zygosity_label)
        zy_f1_metric.update(zy_out, zygosity_label)
        zy_conf_metric.update(zy_out, zygosity_label)

        gt_out = gt_out.cpu().data.contiguous().view(-1, config.model.num_gt_class)
        genotype_label = genotype_label.cpu().data.contiguous().view(-1)
        gt_acc_metric.update(gt_out, genotype_label)
        gt_f1_metric.update(gt_out, genotype_label)
        gt_conf_metric.update(gt_out, genotype_label)

    avg_loss = total_loss / total_images
    batch_zy_acc = float(zy_acc_metric.compute())
    batch_zy_f1 = float(zy_f1_metric.compute())
    batch_gt_acc = float(gt_acc_metric.compute())
    batch_gt_f1 = float(gt_f1_metric.compute())
    if visualizer is not None:
        visualizer.add_scalar('eval_loss', avg_loss, epoch)
        visualizer.add_scalar('zy_accuracy', batch_zy_acc, epoch)
        visualizer.add_scalar('zy_f1score', batch_zy_f1, epoch)
        visualizer.add_scalar('gt_accuracy', batch_gt_acc, epoch)
        visualizer.add_scalar('gt_f1score', batch_gt_f1, epoch)
    print('-Validating-Epoch:%d, | Zygosity Accuracy:%.5f F1-Score:%.5f | Genotype Accuracy:%.5f F1-Score:%.5f' % (
        epoch, batch_zy_acc, batch_zy_f1, batch_gt_acc, batch_gt_f1))
    print(zy_conf_metric.compute())
    print(gt_conf_metric.compute())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='path to config file', required=True)
    parser.add_argument('-log', type=str, default='train.log', help='name of log file')
    parser.add_argument('-mode', type=str, default='retrain', help="training mode: retrain or continue")
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('egs', config.configname, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    num_workers = config.training.num_gpu * 2
    if config.data.dev != "None":
        # training_paths = [config.data.train + '/' + fn for fn in os.listdir(config.data.train) if fn.endswith('.npz')]
        # validating_paths = [config.data.dev + '/' + fn for fn in os.listdir(config.data.dev) if fn.endswith('.npz')]
        validating_dataset = TrainDataset3(config.data.dev)
        logger.info("Load validating dataset from %s" % config.data.dev)
        training_dataset = TrainDataset3(config.data.train)
        logger.info("Load training dataset from %s" % config.data.train)
        # train_dataset = TrainDataset(training_paths, config.data.flanking_size)
        # validate_dataset = TrainDataset(validating_paths, config.data.flanking_size)
    else:
        filelist = [config.data.train + '/' + fb for fb in os.listdir(config.data.train) if fb.endswith('.npz')]
        np.random.shuffle(filelist)
        filelist_size = len(filelist)
        training_paths = filelist[:int(filelist_size * 0.9)]  # 90% for training
        validating_paths = filelist[int(filelist_size * 0.9):]  # 10% for testing
        assert len(validating_paths) > 0
        # train_dataset = TrainDataset(training_paths, config.data.flanking_size)
        # validate_dataset = TrainDataset(validating_paths, config.data.flanking_size)

    model = ResNetwork(config.model)

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.forward_layer.load_state_dict(checkpoint['forward_layer'])
        logger.info('Loaded model from %s' % config.training.load_model)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    n_params, enc, fow = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the ForwardLayer: %d' % fow)

    # optimizer = Optimizer(model.parameters(), config.optim)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    logger.info('Created a %s optimizer.' % config.optim.type)

    if opt.mode == 'continue':
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 0

    # create a visualizer
    if config.training.visualization:
        visual_log = os.path.join(exp_name, 'log')
        visualizer = SummaryWriter(os.path.join(visual_log, 'train'))
        dev_visualizer = SummaryWriter(os.path.join(visual_log, 'dev'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    global_step = 0
    for epoch in range(start_epoch, config.training.epochs):

        global_step = train3(epoch, global_step, config, model, training_dataset, config.training.batch_size, optimizer,
                             logger, visualizer)

        scheduler.step()

        if config.training.eval_or_not:
            eval3(epoch, config, model, validating_dataset, config.training.batch_size, logger, dev_visualizer)

        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        save_model2(model, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)

    logger.info('The training process is OVER!')


if __name__ == '__main__':
    main()
