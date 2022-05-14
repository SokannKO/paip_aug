import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
import cv2 as cv
import spams
import torch
import collections

def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj

def setup_logger(distributed_rank=0, filename="debug.log"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)

    # don't log results for the non-master process
    fmt = "[%(asctime)s %(levelname)s] %(message)s"

    if distributed_rank == 0:
        fmt = "[%(asctime)s %(levelname)s] %(message)s"

    elif distributed_rank == 1:
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    return logger


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret

def cls_count(path):
    list_path = path
    with open(list_path, 'r') as f:
        lines = f.readlines()
        cls_dict = {}

        for line in lines:
            line = line.split('\n')[0]
            if '\t' in line:
                line = line.split('\t')
            else:
                line = line.split(' ')

            if line[-1] in cls_dict:
                cls_dict[line[-1]] += 1
            else:
                cls_dict[line[-1]] = 1
    #print (cls_dict)
    return cls_dict

def get_best_acc(acc, cfg, idx):
    if acc >= cfg.TRAIN.best_acc[idx]:
        cfg.TRAIN.tmp_acc[idx] = acc
        cfg.TRAIN.best_acc_cfg[idx] = True

def sec2dhms(t):
    days = t // (24*3600)
    t = t % (24*3600)
    hours = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t

    return days,hours,minutes,seconds

def LA_save(lst, path):
    np.save(os.path.join(path,'LA_save.npy'), lst)

def LA_plot(path):
    import matplotlib.pyplot as plt

    input_path = os.path.join(path, 'LA_save.npy')
    save_path = os.path.join(path, 'LA_plot.png')
    LA_arr = np.load(input_path)

    train_losses = LA_arr[:, 0]
    train_acc = LA_arr[:, 1]
    val_losses = LA_arr[:, 3]
    val_acc = LA_arr[:, 4]

    ep = len(train_losses)
    dim = np.arange(1, ep + 1)

    f = plt.figure(figsize=(16, 9))

    ax1 = f.add_subplot(111)
    ax1.set_xticks(dim)
    ax1.yaxis.set_label_position("left")
    ax1.set_ylabel('Loss')
    ax1.plot(dim, train_losses, label="train/loss", color='#0000c0')
    ax1.plot(dim, val_losses, label="val/loss", color='#c0c000')
    ax1.set_xlabel('epochs')
    # ax1.legend(loc=3)

    ax2 = ax1.twinx()
    ax2.set_xticks(dim)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 100])
    ax2.plot(dim, train_acc, label="train/acc", color='#c00000')
    ax2.plot(dim, val_acc, label="val/acc", color='#00c000')
    # ax2.legend(loc=2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=3)

    plt.xticks(np.arange(0, ep + 1, ep / 20))
    plt.title(path)
    # plt.show()
    plt.savefig(save_path)

def RES_save(lst, path):
    np.save(os.path.join(path,'result.npy'), lst)

def RES_plot(res_dict, path):
    d = 0
    import matplotlib.pyplot as plt

    #input_path = os.path.join(path, 'result.npy')
    save_path = os.path.join(path, 'LA_plot.png')
    #res_arr = np.load(input_path, allow_pickle=True)[()]
    res_arr = res_dict
    dice_opt = False

    train_losses = [f['loss'] for f in res_arr['train'].values()]
    train_acc = [f['acc'] for f in res_arr['train'].values()]
    val_losses = [f['loss'] for f in res_arr['val'].values()]
    val_acc = [f['acc'] for f in res_arr['val'].values()]

    try:
        train_dice = [f['dice']*100 for f in res_arr['train'].values()]
        val_dice = [f['dice']*100 for f in res_arr['val'].values()]
        dice_opt = True
    except:
        pass


    ep = len(train_losses)
    dim = np.arange(1, ep + 1)

    f = plt.figure(figsize=(16, 9))

    ax1 = f.add_subplot(111)
    ax1.set_xticks(dim)
    ax1.yaxis.set_label_position("left")
    ax1.set_ylabel('Loss')
    ax1.plot(dim, train_losses, label="train/loss", color='#0000c0')
    ax1.plot(dim, val_losses, label="val/loss", color='#c0c000')
    ax1.set_xlabel('epochs')
    # ax1.legend(loc=3)

    ax2 = ax1.twinx()
    ax2.set_xticks(dim)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 100])
    ax2.plot(dim, train_acc, label="train/acc", color='#c00000')
    ax2.plot(dim, val_acc, label="val/acc", color='#00c000')

    if dice_opt:
        ax2.plot(dim, train_dice, label="train/dice", color='#c000c0')
        ax2.plot(dim, val_dice, label="val/dice", color='#00c0c0')
    # ax2.legend(loc=2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=3)

    #plt.xticks(np.arange(0, ep + 1, ep / 20))
    plt.title(path)
    #plt.show()
    plt.savefig(save_path)
    plt.close()

def crop_pad(img_size, crop_size=(512,512)):
    ori_size = img_size
    _crop_size = check_crop_size(ori_size, crop_size)
    crop_loc = get_crop_loc(ori_size, _crop_size)

    return crop_loc, _crop_size

def get_pad_val(crop_size, actual_size):
    pad_x, pad_y = 0, 0
    cfg = False
    if crop_size[0] != actual_size[0]:
        pad_x = (crop_size[0] - actual_size[0]) / 2
        cfg = True
    if crop_size[1] != actual_size[1]:
        pad_y = (crop_size[1] - actual_size[1]) / 2
        cfg = True

    return cfg, (-pad_x,-pad_y,actual_size[0]+pad_x,actual_size[1]+pad_y)

def check_crop_size(ori_size, crop_size):
    _x, _y = crop_size
    if ori_size[0] < crop_size[0]:
        _x = ori_size[0]
    if ori_size[1] < crop_size[1]:
        _y = ori_size[1]
    return (_x, _y)

def dv(a, b):
    dv_cnt = 0
    if a%b: dv_cnt = a//b + 1
    else: dv_cnt = a//b

    return dv_cnt

def ov_size(a, b, cnt):
    if cnt == 1:
        return 0
    return int((b*cnt-a)/(cnt-1))

def get_crop_loc(ori_size, crop_size):
    res_loc = list()

    x_dv_cnt = dv(ori_size[0], crop_size[0])
    y_dv_cnt = dv(ori_size[1], crop_size[1])
    x_ov_size = ov_size(ori_size[0], crop_size[0], x_dv_cnt)
    y_ov_size = ov_size(ori_size[1], crop_size[1], y_dv_cnt)

    y_loc = 0
    for y in range(y_dv_cnt):
        x_loc = 0
        for x in range(x_dv_cnt):
            res_loc.append([x_loc, y_loc])
            x_loc = x_loc+(crop_size[0] - x_ov_size)
        y_loc = y_loc+(crop_size[1] - y_ov_size)

    return res_loc


##### stain_norm

def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)

def get_concentrations(I, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T


def get_stain_matrix(I, threshold=0.8, lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = normalize_rows(dictionary)
    return dictionary


def VahadaneNormalizer_transform(I, stain_matrix_target):
    I = standardize_brightness(I)
    stain_matrix_source = get_stain_matrix(I)
    source_concentrations = get_concentrations(I, stain_matrix_source)
    return (255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target).reshape(I.shape))).astype(
        np.uint8)

