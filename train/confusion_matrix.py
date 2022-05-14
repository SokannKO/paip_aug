# System libs
import os
import glob
import argparse
from distutils.version import LooseVersion
import torch
import torch.nn as nn
from models import ModelBuilder, SegmentationModule, ClassificationModule
import transforms
import utils
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import shutil

Image.MAX_IMAGE_PIXELS = 5000000000

lr_step = False
SVS_LEVEL = 1

def test_cls(classification_module, dataloader, tags, cfgs):
    # switch to train mode

    cfg_cls, cfg_seg = cfgs

    classes = ('Non_Tumor', 'Tumor_WO_Nerve', 'Tumor_W_Nerve')
    #classes = ('NonTumor', 'Tumor')
    dest_dir = os.path.join('CM', cfg_cls.TEST.result)
    os.makedirs(dest_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(dest_dir, cls), exist_ok=True)

    classification_module.eval()

    y_pred = []
    y_true = []
    # main loop

    with torch.no_grad():

        for i, data in enumerate(dataloader, 0):
            # load a batch of data
            batch_data = data
            classification_module.zero_grad()

            # forward pass
            loss, acc, s_outputs, _ = classification_module(batch_data)
            acc1, acc5 = acc
            loss = loss.mean()
            acc1 = acc1.mean()

            # """
            score_list, sc_idx = torch.max(s_outputs, dim=1)
            # """

            pred = sc_idx.cpu().numpy()
            y_pred.extend(pred)  # Save Prediction

            labels = batch_data[1].cpu().numpy()
            y_true.extend(labels)  # Save Truth

            #'''
            if pred[0] != labels[0]:
                img_path = batch_data[2][0]
                pred_dir = os.path.join(dest_dir,classes[labels[0]],'pred_'+classes[pred[0]])
                os.makedirs(pred_dir, exist_ok=True)
                shutil.copy(img_path, pred_dir)
            #'''

        # Build confusion matrix
        plt.figure(figsize=(12, 7))
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                             columns=[i for i in classes])
        plt.title(cfg_cls.CKPT_DIR, size=18)
        sn.heatmap(df_cm, annot=True)
        plt.ylabel('True', size=14)
        plt.xlabel('Prediction', size=14)
        plt.savefig(os.path.join(dest_dir,'CM.png'))

        print(classification_report(y_true, y_pred, target_names=classes))
        print(roc_auc_score(y_true, y_pred, average=None))

    # test_log = '[%5d] TEST lr: %.8f loss: %.5f acc: %.5f' % \
    #         (i + 1, cfg.TRAIN.running_lr_encoder, ave_total_loss.average(), ave_acc.average())

    # print(val_log)
    # logger.info(TEST)

def main(cfgs, gpus):
    # Network Builders

    cfg_cls, cfg_seg = cfgs
    import dataset_cls as datasets

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    testset = datasets.DatasetList2d(cfg_cls.DATASET.list_val, cfg_cls.DATASET, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    loader_test = torch.utils.data.DataLoader(testset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    num_classes = cfg_cls.DATASET.num_class
    num_tr_data = len(loader_test)

    tags = {}
    tags['mask_line'] = 'mask_line'
    tags['mask_seg'] = 'tile_mask_seg'
    tags['ovly'] = 'ovly'
    tags['image'] = 'tile_image'
    tags['mask_subm'] = 'mask_subm'

    ### load cls model
    cls_model_path = os.path.join(cfg_cls.CKPT_DIR, cfg_cls.TEST.checkpoint)
    if cfg_cls.TRAIN.ckpt_interval < 0:
        cls_model_path = glob.glob(os.path.join(cfg_cls.CKPT_DIR, 'model_*_best.pth'))[0]

    print(cls_model_path)

    cls_net_encoder = ModelBuilder.build_encoder(
        arch=cfg_cls.MODEL.arch_encoder.lower(),
        fc_dim=cfg_cls.MODEL.fc_dim,
        weights=cls_model_path,
        num_class=cfg.DATASET.num_class)
        
    cls_net_encoder = torch.nn.DataParallel(cls_net_encoder, device_ids=[0])

    #cls_crit = nn.NLLLoss(ignore_index=-1)
    cls_crit = nn.CrossEntropyLoss().cuda()
    classification_module = ClassificationModule(
        cls_net_encoder, cls_crit)

    print('Data count = {}'.format(len(loader_test)))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    classification_module.to(device)

    # Main loop
    test_cls(classification_module, loader_test, tags, cfgs)

    print('Testing Done!')

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Paip Testing"
    )
    parser.add_argument(
        "--cfgcls",
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

    from config import cfg
    cfg.merge_from_file(args.cfgcls)
    cfg_cls = cfg.clone()


    # Parse gpu ids
    gpus = utils.parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)

    cfgs = [cfg_cls, None]

    main(cfgs, gpus)

