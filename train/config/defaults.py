from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.CKPT_DIR = "ckpt/tmp"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/train.txt"
_C.DATASET.list_val = "./data/val.txt"
_C.DATASET.list_test = "./data/test.txt"
_C.DATASET.md_classes = ['A','B']
_C.DATASET.num_class = 2
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (512,512)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True
_C.DATASET.stain_norm = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# type
_C.MODEL.type = "Segmentation"
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50"
# architecture of net_decoder
_C.MODEL.arch_decoder = ""
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.tr_batchsize = 2
# epochs to train for
_C.TRAIN.tr_num_epochs = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.tr_epoch_num_iters = 1
_C.TRAIN.ckpt_interval = 1

_C.TRAIN.tr_optim = "SGD"
_C.TRAIN.tr_lr_encoder = 0.02
_C.TRAIN.tr_lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.tr_lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.tr_momentum = 0.9
# weights regularizer
_C.TRAIN.tr_weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.vl_batchsize = 1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.ts_batchsize = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
