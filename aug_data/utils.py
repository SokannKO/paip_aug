import numpy as np
import spams
import cv2 as cv

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