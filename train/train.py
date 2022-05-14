import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# torch library
import torch
import torch.nn as nn
# libs
from config import cfg
from models import ModelBuilder, SegmentationModule, ClassificationModule
from utils import AverageMeter, parse_devices, setup_logger, RES_save, RES_plot, vis_log, cls_count, get_best_acc
import transforms
#from torch.optim.lr_scheduler import StepLR
#from transforms import autoaugment

lr_step = False

# train one epoch
def train_module(_module, dataloader, optimizers, history, epoch, cfg):
    # switch to train mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_score = AverageMeter()
    ave_dice = AverageMeter()
    batch_ave_score = 0

    _module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    c = 0
    for eni in range(cfg.TRAIN.tr_epoch_num_iters):
        for i, data in enumerate(dataloader, 0):
            # load a batch of data
            batch_data = data
            data_time.update(time.time() - tic)
            _module.zero_grad()

            # adjust learning rate
            if not lr_step:
                cur_iter = c + i + (epoch - 1) \
                           * cfg.TRAIN.tr_epoch_iters * cfg.TRAIN.tr_epoch_num_iters
                adjust_learning_rate(optimizers, cur_iter, cfg)
            # forward pass
            loss, acc, s_outputs, dice = _module(batch_data)

            if cfg.MODEL.type.lower() == 'classification':
                acc1, acc5 = acc
                acc1 = acc1.mean()

                # get ave score
                # """
                score_list, sc_idx = torch.max(s_outputs, dim=1)
                # print (score_list)
                batch_ave_score = sum(score_list.tolist()) / len(sc_idx)
                # """
            elif cfg.MODEL.type.lower() == 'segmentation':
                acc1 = acc.mean()

            loss = loss.mean()

            # Backward
            loss.backward()

            for optimizer in optimizers:
                if optimizer is not None:
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss and acc
            ave_total_loss.update(loss.data.item())
            ave_acc.update(acc1.data.item())
            ave_score.update(batch_ave_score)
            ave_dice.update(dice)

            fractional_epoch = 1. * cur_iter / cfg.TRAIN.max_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc1.data.item())
            if cfg.MODEL.type.lower() == 'segmentation':
                history['train']['dice'].append(dice)

        c = i

        if cfg.MODEL.type.lower() == 'classification':
            tr_log = '[%d, %5d] TRAIN lr: %.8f ave_score: %.5f loss: %.5f acc: %.5f' % \
                     (epoch, i + 1, cfg.TRAIN.running_lr_encoder, ave_score.average(), ave_total_loss.average(),
                      ave_acc.average())

        elif cfg.MODEL.type.lower() == 'segmentation':
            tr_log = '[%d, %5d] TRAIN lr_enc: %.6f lr_dec: %.6f ave_score: %.5f loss: %.5f acc: %.5f, dice: %.5f' % \
                     (epoch, i + 1, cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                      ave_score.average(), ave_total_loss.average(),
                      ave_acc.average(), ave_dice.average())

        logger.info(tr_log)

    get_best_acc(float(ave_acc.average()), cfg, 0)

    cfg.TRAIN.train_dict_elem = [epoch,
                                 ave_acc.average(),
                                 ave_total_loss.average(),
                                 ave_score.average(),
                                 cfg.TRAIN.running_lr_encoder]
    if cfg.MODEL.type.lower() == 'segmentation':
        cfg.TRAIN.train_dict_elem.append(cfg.TRAIN.running_lr_decoder)
        cfg.TRAIN.train_dict_elem.append(ave_dice.average())

    ####################################################################


def val_module(_module, dataloader, epoch, cfg):
    # switch to train mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_score = AverageMeter()
    ave_dice = AverageMeter()
    batch_ave_score = 0

    _module.eval()

    # main loop
    tic = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # load a batch of data
            batch_data = data
            data_time.update(time.time() - tic)
            _module.zero_grad()

            # forward pass
            loss, acc, s_outputs, dice = _module(batch_data)

            if cfg.MODEL.type.lower() == 'classification':
                acc1, acc5 = acc
                acc1 = acc1.mean()

                # get ave score
                # """
                score_list, sc_idx = torch.max(s_outputs, dim=1)
                batch_ave_score = sum(score_list.tolist()) / len(sc_idx)
                # """
            elif cfg.MODEL.type.lower() == 'segmentation':
                acc1 = acc.mean()

            loss = loss.mean()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss and acc
            ave_total_loss.update(loss.data.item())
            ave_acc.update(acc1.data.item())
            ave_score.update(batch_ave_score)
            ave_dice.update(dice)

    if cfg.MODEL.type.lower() == 'classification':

        val_log = '[%d, %5d] VAL lr: %.8f ave_score: %.5f loss: %.5f acc: %.5f' % \
                  (epoch, i + 1, cfg.TRAIN.running_lr_encoder, ave_score.average(), ave_total_loss.average(),
                   ave_acc.average())
    elif cfg.MODEL.type.lower() == 'segmentation':
        val_log = '[%d, %5d] VAL lr_enc: %.6f lr_dec: %.6f ave_score: %.5f loss: %.5f acc: %.5f dice: %.5f' % \
                  (epoch, i + 1, cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                   ave_score.average(), ave_total_loss.average(), ave_acc.average(), ave_dice.average())
    # print(val_log)
    logger.info(val_log)
    get_best_acc(float(ave_acc.average()), cfg, 1)

    cfg.TRAIN.val_dict_elem = [epoch,
                               ave_acc.average(),
                               ave_total_loss.average(),
                               ave_score.average(),
                               cfg.TRAIN.running_lr_encoder]
    if cfg.MODEL.type.lower() == 'segmentation':
        cfg.TRAIN.val_dict_elem.append(cfg.TRAIN.running_lr_decoder)
        cfg.TRAIN.val_dict_elem.append(ave_dice.average())


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    _optim = cfg.TRAIN.tr_optim
    optimizer_decoder = None
    if _optim.lower() == 'sgd':
        optimizer_encoder = torch.optim.SGD(
            group_weight(net_encoder),
            lr=cfg.TRAIN.tr_lr_encoder,
            momentum=cfg.TRAIN.tr_momentum,
            weight_decay=cfg.TRAIN.tr_weight_decay)
        if net_decoder is not None:
            optimizer_decoder = torch.optim.SGD(
                group_weight(net_decoder),
                lr=cfg.TRAIN.tr_lr_decoder,
                momentum=cfg.TRAIN.tr_momentum,
                weight_decay=cfg.TRAIN.tr_weight_decay)
    elif _optim.lower() == 'adam':
        optimizer_encoder = torch.optim.Adam(
            group_weight(net_encoder),
            lr=cfg.TRAIN.tr_lr_encoder,
            # betas=cfg.TRAIN.tr_momentum,
            weight_decay=cfg.TRAIN.tr_weight_decay)
        if net_decoder is not None:
            optimizer_decoder = torch.optim.Adam(
                group_weight(net_decoder),
                lr=cfg.TRAIN.tr_lr_decoder,
                # betas=cfg.TRAIN.tr_momentum,
                weight_decay=cfg.TRAIN.tr_weight_decay)

    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    # scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.tr_lr_pow)
    scale_running_lr = ((cfg.TRAIN.max_iters / (9.0 * float(cur_iter) + cfg.TRAIN.max_iters)) ** cfg.TRAIN.tr_lr_pow)

    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.tr_lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.tr_lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    if optimizer_decoder is not None:
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr_decoder


def adjust_learning_rate_step(optimizers):
    (optimizer_encoder, optimizer_decoder) = optimizers

    cfg.TRAIN.running_lr_encoder = optimizer_encoder.param_groups[0]['lr']
    cfg.TRAIN.running_lr_decoder = optimizer_decoder.param_groups[0]['lr']
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder

    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def checkpoint(nets, cfg, epoch):
    (enc, dec, crit) = nets
    if cfg.TRAIN.ckpt_interval < 0:
        if cfg.TRAIN.best_acc_cfg[0] and cfg.TRAIN.best_acc_cfg[1] and \
                -cfg.TRAIN.ckpt_interval <= epoch:
            if -cfg.TRAIN.ckpt_interval != epoch:
                print('removing previous model...')
                os.remove('{}/model_epoch_{}_best.pth' \
                          .format(cfg.CKPT_DIR, cfg.TRAIN.best_acc_cfg[2]))
                cfg.TRAIN.best_acc_cfg[2] = epoch

            print('Saving best model...')
            if dec is None:
                dict_model = enc.module.state_dict()
                torch.save(
                    dict_model,
                    '{}/model_epoch_{}_best.pth'.format(cfg.CKPT_DIR, epoch))
            else:
                dict_enc = enc.module.state_dict()
                torch.save(
                    dict_enc,
                    '{}/enc_epoch_{}_best.pth'.format(cfg.CKPT_DIR, epoch))
                dict_dec = dec.module.state_dict()
                torch.save(
                    dict_dec,
                    '{}/dec_epoch_{}_best.pth'.format(cfg.CKPT_DIR, epoch))

            cfg.TRAIN.best_acc = cfg.TRAIN.tmp_acc.copy()

        cfg.TRAIN.best_acc_cfg[0] = False
        cfg.TRAIN.best_acc_cfg[1] = False

    elif epoch % cfg.TRAIN.ckpt_interval == 0 or epoch == cfg.TRAIN.tr_num_epochs:
        print('Saving checkpoints...')
        if dec is None:
            dict_model = enc.module.state_dict()
            torch.save(
                dict_model,
                '{}/model_epoch_{}.pth'.format(cfg.CKPT_DIR, epoch))
        else:
            dict_enc = enc.module.state_dict()
            torch.save(
                dict_enc,
                '{}/enc_epoch_{}.pth'.format(cfg.CKPT_DIR, epoch))
            dict_dec = dec.module.state_dict()
            torch.save(
                dict_dec,
                '{}/dec_epoch_{}.pth'.format(cfg.CKPT_DIR, epoch))


def main(cfg, gpus):
    # Network Builders
    GPU_ids = gpus
    print('GPU(s) = ', GPU_ids)

    if cfg.MODEL.type.lower() == 'classification':
        # Data loading code
        import dataset_cls as datasets

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # aa_policy = autoaugment.AutoAugmentPolicy("imagenet")
        # autoaugment.AutoAugment(policy=aa_policy)
        trainset = datasets.DatasetList2d(cfg.DATASET.list_train, cfg.DATASET, transforms.Compose([
            # transforms.RandomResizedCrop(512),
            # transforms.RandomHorizontalFlip(),
            # autoaugment.AutoAugment(policy=aa_policy),
            transforms.HEDJitter(theta=0.05),
            transforms.RandomAffineCV2(alpha=0.1),
            transforms.ToTensor(),
            normalize,
        ]))
        valset = datasets.DatasetList2d(cfg.DATASET.list_val, cfg.DATASET, transforms.Compose([
            # transforms.RandomResizedCrop(512),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))
        loader_train = torch.utils.data.DataLoader(trainset,
                                                   batch_size=cfg.TRAIN.tr_batchsize, shuffle=True,
                                                   num_workers=cfg.TRAIN.workers, pin_memory=True)
        loader_val = torch.utils.data.DataLoader(valset,
                                                 batch_size=cfg.VAL.vl_batchsize, shuffle=True,
                                                 num_workers=cfg.TRAIN.workers, pin_memory=True)

        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder,
            num_class=cfg.DATASET.num_class)

        net_encoder = torch.nn.DataParallel(net_encoder, device_ids=GPU_ids)

        ### weighted CE to balance loss
        cls_c_dict_tr = cls_count(cfg.DATASET.list_train)
        cls_c_dict_vl = cls_count(cfg.DATASET.list_val)

        logger.info(cls_c_dict_tr)
        logger.info(cls_c_dict_vl)

        cls_c_dict = dict(sorted(cls_c_dict_tr.items(), key=lambda x: int(x[0])))

        cls_0 = cls_c_dict['11']+cls_c_dict['14']
        cls_1 = cls_c_dict['13']
        cls_2 = cls_c_dict['12']

        summed = cls_0 + cls_1 + cls_2
        weight_arr = [float(cls_0)/float(summed),
                      float(cls_1)/float(summed),
                      float(cls_2)/float(summed)]
        weight = torch.tensor(weight_arr,
                              dtype=torch.float)
        # crit = nn.NLLLoss(ignore_index=-1)
        crit = nn.CrossEntropyLoss(weight=weight).cuda()

        if cfg.MODEL.arch_encoder.lower()[:9] == 'inception':
            _module = ClassificationModule(
                net_encoder, crit, incpt=True)
        else:
            _module = ClassificationModule(
                net_encoder, crit)

        print('Data count = {}'.format(len(loader_train)))

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda")
        _module.to(device)
        nets = (net_encoder, None, crit)

    elif cfg.MODEL.type.lower() == 'segmentation':
        import dataset_seg as datasets

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # aa_policy = autoaugment.AutoAugmentPolicy("imagenet")
        # autoaugment.AutoAugment(policy=aa_policy)
        trainset = datasets.TrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET, )
        valset = datasets.TrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET, )

        loader_train = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.TRAIN.tr_batchsize,
            shuffle=False,
            num_workers=cfg.TRAIN.workers,
            pin_memory=True)
        loader_val = torch.utils.data.DataLoader(
            valset,
            batch_size=cfg.VAL.vl_batchsize,
            shuffle=False,
            num_workers=cfg.TRAIN.workers,
            pin_memory=True)

        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder)

        net_encoder = torch.nn.DataParallel(net_encoder, device_ids=GPU_ids)
        net_decoder = torch.nn.DataParallel(net_decoder, device_ids=GPU_ids)

        crit = nn.NLLLoss(ignore_index=-1)
        # crit = nn.CrossEntropyLoss().cuda()

        if cfg.MODEL.arch_decoder.endswith('deepsup'):
            _module = SegmentationModule(
                net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
        else:
            _module = SegmentationModule(
                net_encoder, net_decoder, crit)

        print('Data count = {}'.format(len(loader_train)))

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda")
        _module.to(device)
        nets = (net_encoder, net_decoder, crit)

    optimizers = create_optimizers(nets, cfg)

    ### init additional cfg ###

    num_classes = cfg.DATASET.num_class
    num_tr_data = len(trainset)
    num_vl_data = len(valset)

    cfg.TRAIN.num_data = num_tr_data
    cfg.VAL.num_data = num_vl_data

    cfg.TRAIN.tr_epoch_iters = num_tr_data
    cfg.TRAIN.max_iters = cfg.TRAIN.tr_epoch_iters * cfg.TRAIN.tr_num_epochs \
                          * cfg.TRAIN.tr_epoch_num_iters
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.tr_lr_encoder
    cfg.TRAIN.timestamp = [0, 0, 0, 0, 0]
    cfg.TRAIN.best_acc = [0.0, 0.0]
    cfg.TRAIN.tmp_acc = [0.0, 0.0]
    cfg.TRAIN.best_acc_cfg = [False, False, 1]

    cfg.TRAIN.LAsave = [0.0, 0.0, 0.0,  # tr_loss,tr_acc,tr_score
                        False, False, False]  # vl_loss,vl_acc,vl_score

    cfg.TRAIN.train_dict_elem = list()
    cfg.TRAIN.val_dict_elem = list()
    train_dict = {}
    val_dict = {}

    # LA_save_list = []
    cfg.TRAIN.timestamp[0] = int(time.time())
    cfg.TRAIN.timestamp[3] = cfg.TRAIN.tr_num_epochs

    history = {'train': {'epoch': [], 'loss': [], 'acc': [], 'dice': []}}
    # Main loop
    for epoch in range(cfg.TRAIN.tr_num_epochs):
        if lr_step:
            adjust_learning_rate_step(optimizers)

        train_module(_module, loader_train, optimizers, history, epoch + 1, cfg)
        val_module(_module, loader_val, epoch + 1, cfg)

        print(cfg.TRAIN.best_acc_cfg)

        # checkpointing
        checkpoint(nets, cfg, epoch + 1)

        train_dict[epoch + 1] = {'ep': cfg.TRAIN.train_dict_elem[0],
                                 'acc': cfg.TRAIN.train_dict_elem[1],
                                 'loss': cfg.TRAIN.train_dict_elem[2],
                                 'score': cfg.TRAIN.train_dict_elem[3]}

        val_dict[epoch + 1] = {'ep': cfg.TRAIN.val_dict_elem[0],
                               'acc': cfg.TRAIN.val_dict_elem[1],
                               'loss': cfg.TRAIN.val_dict_elem[2],
                               'score': cfg.TRAIN.val_dict_elem[3]}

        if cfg.MODEL.type.lower() == 'classification':
            train_dict[epoch + 1]['lr'] = cfg.TRAIN.train_dict_elem[4]
            val_dict[epoch + 1]['lr'] = cfg.TRAIN.val_dict_elem[4]
        elif cfg.MODEL.type.lower() == 'segmentation':
            train_dict[epoch + 1]['lr_enc'] = cfg.TRAIN.train_dict_elem[4]
            train_dict[epoch + 1]['lr_dec'] = cfg.TRAIN.train_dict_elem[5]
            train_dict[epoch + 1]['dice'] = cfg.TRAIN.train_dict_elem[6]
            val_dict[epoch + 1]['lr_enc'] = cfg.TRAIN.val_dict_elem[4]
            val_dict[epoch + 1]['lr_dec'] = cfg.TRAIN.val_dict_elem[5]
            val_dict[epoch + 1]['dice'] = cfg.TRAIN.val_dict_elem[6]

        cfg.TRAIN.timestamp[1] = int(time.time())
        cfg.TRAIN.timestamp[2] = epoch + 1
        cfg.TRAIN.timestamp[3] -= 1
        cum_t = int((cfg.TRAIN.timestamp[1] - cfg.TRAIN.timestamp[0])
                    / cfg.TRAIN.timestamp[2] * cfg.TRAIN.timestamp[3])
        cfg.TRAIN.timestamp[4] = cum_t

        res_dict = {}
        res_dict['train'] = train_dict
        res_dict['val'] = val_dict

        ### save result info

        RES_save(res_dict, cfg.CKPT_DIR)
        RES_plot(res_dict, cfg.CKPT_DIR)

        vis_log.vis_html(cfg, res_dict)

    import shutil
    shutil.copyfile(log_file, os.path.join(cfg.CKPT_DIR, 'result.log'))

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Training Application"
    )
    parser.add_argument(
        "--cfg",
        default="config/tmp-densenet161trf_3c_CV1_cls.yaml",

        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--logs",
        default="./logs",
        help="path to logs dir"
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if not os.path.exists(args.logs):
        os.makedirs(args.logs)
    cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    log_file = os.path.join(args.logs, cur_time + '.log')

    logger = setup_logger(distributed_rank=0, filename=log_file)  # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.CKPT_DIR):
        os.makedirs(cfg.CKPT_DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.CKPT_DIR))
    with open(os.path.join(cfg.CKPT_DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.CKPT_DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.CKPT_DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
               os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    # GPU_ids = [int(g) for g in args.gpus.split(',')]

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
