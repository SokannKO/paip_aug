import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 5000000000
import cv2
import random
import utils
from skimage.filters import threshold_multiotsu
from skimage import data, io, img_as_ubyte
from datetime import datetime

'''
seg translation
'''
def PNI_prediction2(img, _):
    """
    :param img: for test, img is a tile image (such as .png), but np.array directly
    :return: pni_line : perineural invasion line
    """
    import random as rng
    rng.seed(12345)

    kernel0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # tumor open
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  # tumor_seg dilate
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # nerve_seg  dilate
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # pni_line

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply multi-Otsu threshold
    try:
        thresholds = threshold_multiotsu(image, classes=3)
    except:
        try:
            thresholds = threshold_multiotsu(image, classes=2)
        except:
            thresholds = threshold_multiotsu(image, classes=1)

    # Digitize (segment) original image into multiple classes.
    # np.digitize assign values 0, 1, 2, 3, ... to pixels in each class.
    regions = np.digitize(image, bins=thresholds)

    # 2d label value (0, 1, 2) 3-classes
    output = img_as_ubyte(regions)  # Convert 64 bit integer values to uint8
    tumor_mask = output == 0
    tumor = Image.fromarray(tumor_mask)
    tumor = tumor.convert('L')
    tumor = np.array(tumor)

    tumor = remove_small_objects(tumor, min_size=230)
    tumor_mask = cv2.morphologyEx(tumor, cv2.MORPH_OPEN, kernel)
    tumor_mask = cv2.dilate(tumor_mask, kernel1, iterations=1)

    ####
    # subt_mask = np.array(seg).astype(bool)*1 - tumor_mask.astype(bool)*1
    # nerv_mask = ((subt_mask == 1)*255).astype(np.uint8)
    # nerv_mask = cv2.dilate(nerv_mask, kernel2, iterations=1)
    ####

    #####################################

    pni_line1 = tumor_mask
    # pni_line2 = tumor_mask2 * nerv_mask

    # pni_line1 = cv2.dilate(pni_line1, kernel2, iterations=1)
    # pni_line1 = cv2.erode(pni_line1, kernel0, iterations=1)
    # pni_line1 = cv2.morphologyEx(pni_line1, cv2.MORPH_OPEN, kernel)
    # pni_line1 = cv2.HoughLinesP(pni_line1, rho=1, theta=np.pi/360, threshold=10, minLineLength=10, maxLineGap=20)

    return pni_line1


def stain_normalization(img, stain_matrix_target):
    img = np.array(img)
    img = utils.VahadaneNormalizer_transform(img, stain_matrix_target)

    return img


def remove_small_objects(img, min_size=120):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    img2 = img
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2


def process_one(cfg, cp_tum, img_seg_path_list):
    '''
    :param cp_tum:
    :param img_seg_path_list:
    :return: RGB Image format
    '''
    def _attempt_one(cp_img_tum, L_cp_seg_mask, ov_cp_img):
        val_dist = 0
        loop_cnt = 1
        while val_dist <= 0:
            ### set seg position
            rand_seed = datetime.now().microsecond
            random.seed(rand_seed)

            contour_x_rand = random.randint(x_pos_min, x_pos_max - img_seg_size[0])
            contour_y_rand = random.randint(y_pos_min, y_pos_max - img_seg_size[1])

            # print((contour_x_rand, contour_y_rand))

            val_dist = cv2.pointPolygonTest(contour, (contour_x_rand, contour_y_rand), True)
            '''
            if val_dist > 0:
                # if True:
                print(val_dist)
            '''
            if loop_cnt > 100:
                print('exceed max loop count', (contour_x_rand, contour_y_rand),
                      (x_pos_min, x_pos_max, y_pos_min, y_pos_max))
                break
            loop_cnt += 1
        # print (loop_cnt)

        ### rotation

        for rot in range(rotation_cnt):
            degree = 360 / rotation_cnt
            d = 0

            rot_img_seg = img_seg.rotate(degree, expand=True)
            cp_seg_mask = Image.new('RGBA', cp_img_tum.size)
            cp_seg_mask.paste(rot_img_seg, (contour_x_rand, contour_y_rand), rot_img_seg)

            np_cp_seg_mask = np.array(cp_seg_mask)[:, :, 3]
            if np_cp_seg_mask.max() == 0:
                continue

            _valid = (np_cp_tum_mask == 0) * (np_cp_seg_mask != 0)
            # valid_ov.save('tmp/valid.png')
            # Image.fromarray(np_cp_seg_mask).save('tmp/seg_mask.png')

            ### validation types PI : ==False, FS : ==True
            if _valid.max() == type_stmt:
                pass
            else:
                # for PI type
                if type == 'PI':
                    # '''
                    mask_gth = (np_cp_tum_mask != 0)
                    mask_seg = (np_cp_seg_mask != 0)

                    # area of contour in label
                    a_p = np.sum(mask_gth)
                    a_l = np.sum(mask_seg)
                    # area of intersection |T ∩ N|
                    a_pl = np.sum(mask_gth * mask_seg)
                    # min(|T|,|N|)
                    min_a_pl = min(a_p, a_l)
                    # |T ∩ N| / min(|T|,|N|)
                    sub_dice = a_pl / min_a_pl

                    if sub_dice < 0.3 or sub_dice > 0.7:
                        continue
                    # '''

                cp_img_tum.paste(cp_seg_mask, (0, 0), cp_seg_mask)
                L_cp_seg_mask = Image.fromarray(np_cp_seg_mask)

                overlay_cfg = True
                if overlay_cfg:
                    np_img_seg = np.array(rot_img_seg)
                    L_img_seg = np_img_seg[:, :, 3]
                    mask_overlay = np.zeros((rot_img_seg.size[1], rot_img_seg.size[0]), dtype=np.uint8)
                    seg_contours, hierarchy = cv2.findContours(L_img_seg, cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(mask_overlay, seg_contours, -1, 255, 2)
                    cv2.drawContours(np_img_seg, seg_contours, -1, (0, 255, 0), 2)
                    np_img_seg[:, :, 3] += mask_overlay
                    ov_img_seg = Image.fromarray(np_img_seg)

                    ov_cp_img.paste(ov_img_seg, (contour_x_rand, contour_y_rand), ov_img_seg)
                    # ov_cp_img.save('tmp/otseg.png')

                # print((contour_x_rand, contour_y_rand))
                c_idx_list.append(c_idx)
                return cp_img_tum, L_cp_seg_mask, ov_cp_img, True

        return cp_img_tum, L_cp_seg_mask, ov_cp_img, False

    cp_img_tum, cp_tum_mask = cp_tum
    rotation_cnt = cfg['rotation_cnt']
    type = cfg['type']

    if type == 'FS':
        type_stmt = True
    elif type == 'PI':
        type_stmt = False

    np_cp_img = np.array(cp_img_tum)
    np_cp_tum_mask = np.array(cp_tum_mask)
    tum_contours, hierarchy = cv2.findContours(np.array(cp_tum_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(np_cp_img, tum_contours, -1, (0, 255, 255), 1)

    # Image.fromarray(np_cp_img).save('tmp/contours.png')

    tum_contours = sorted(tum_contours, key=len)[::-1]
    print(len(tum_contours))

    L_cp_seg_mask = None
    ov_cp_img = Image.fromarray(np_cp_img)
    c_idx_list = []
    for img_seg_path in img_seg_path_list:
        img_seg = Image.open(img_seg_path)
        for c_idx, contour in enumerate(tum_contours):
            if len(c_idx_list) == len(img_seg_path_list):
                continue
            if c_idx in c_idx_list:
                continue
            if len(contour) < 10:
                continue

            x_pos = contour[:, 0, 0]
            y_pos = contour[:, 0, 1]

            img_seg_size = img_seg.rotate(360 / rotation_cnt, expand=True).size
            x_pos_min, x_pos_max = x_pos.min(), x_pos.max()
            y_pos_min, y_pos_max = y_pos.min(), y_pos.max()

            if (x_pos_max - img_seg_size[0] <= x_pos_min) or (y_pos_max - img_seg_size[1] <= y_pos_min):
                continue

            ### attempt count
            suc_config = False
            for _t in range(20):
                if suc_config:
                    continue
                cp_img_tum, L_cp_seg_mask, ov_cp_img, suc_config =\
                    _attempt_one(cp_img_tum, L_cp_seg_mask, ov_cp_img)
                d = 0

    return cp_img_tum, L_cp_seg_mask, ov_cp_img
    # return None, None, None


def main():
    _save_dir_image = os.path.join(out_aug_dir, 'image')
    _save_dir_mask = os.path.join(out_aug_dir, 'mask')
    _save_dir_ovly = os.path.join(out_aug_dir, 'ovly')
    _save_dir_null = os.path.join(out_aug_dir, 'null')
    _save_list_txt = os.path.join(out_aug_dir, 'list.txt')

    os.makedirs(out_aug_dir, exist_ok=True)
    os.makedirs(_save_dir_ovly, exist_ok=True)
    os.makedirs(_save_dir_mask, exist_ok=True)
    os.makedirs(_save_dir_image, exist_ok=True)
    os.makedirs(_save_dir_null, exist_ok=True)
    f = open(_save_list_txt, 'a')

    otseg_dict = np.load(otseg_dict_path, allow_pickle=True)[()]
    for _ot_name in otseg_dict:
        _ot_path = os.path.join(_ori_tumor_dir, _ot_name)
        _ot_tag = _ot_name[:-4]
        _ot_tag_2 = _ot_name[:-8]

        img_ot = Image.open(_ot_path)

        img_ot_x = img_ot.size[0]
        img_ot_y = img_ot.size[1]
        quad = [[0, 0, int(img_ot_x / 2), int(img_ot_y / 2)],
                [int(img_ot_x / 2), int(img_ot_y / 2), img_ot_x, img_ot_y],
                [int(img_ot_x / 2), 0, img_ot_x, int(img_ot_y / 2)],
                [0, int(img_ot_y / 2), int(img_ot_x / 2), img_ot_y]
                ]

        sorted_otseg = sorted([f for f in otseg_dict[_ot_name]], reverse=True, key=lambda x: x[1][0] * x[1][1])

        if len(sorted_otseg) == 0:
            continue
        # for idx, _s_info in enumerate(sorted_otseg):

        cp_size = cfg['cp_size']
        img_tum = Image.open(_ot_path)
        # stain_norm_dataset = {}
        if stain_norm_cfg:
            stain_target_img = np.array(Image.open(SN_target1_path))
            target = utils.standardize_brightness(stain_target_img)
            stain_matrix_target = utils.get_stain_matrix(target)
            print('Processing stain normalization..')

            # stain normalization
            img = stain_normalization(img_tum, stain_matrix_target)
            img_tum = Image.fromarray(img)
            # stain_norm_dataset[_ot_path] = img
            debug = 0

        np_img = np.array(img_tum).astype(np.uint8)
        cp_loc, _cp_size = utils.crop_pad(img_tum.size, crop_size=cp_size)
        pni = PNI_prediction2(np_img, '')

        rand_seg_count = random.randint(1, cfg['max_seg_count'])

        for itr in range(cfg['itr_no']):
            _s_info_list = random.choices(sorted_otseg, k=rand_seg_count)
            _seg_path_list = []
            for _s_info in _s_info_list:
                _s_name = _s_info[0]
                _seg_tag = _s_name[:-8]
                _seg_tag_s = _s_name[:-4]
                _seg_path = os.path.join(_seg_dir, _seg_tag, _s_name)
                _seg_path_list.append(_seg_path)

            ### for distributing in quad
            '''
            img_seg = Image.open(_seg_path)
            if idx != 0:
                if max(0, quad[idx][2] - img_seg.size[0]) - quad[idx][0] < img_seg.size[0] or\
                        max(0, quad[idx][3] - img_seg.size[1]) - quad[idx][1] < img_seg.size[1]:
                    continue
            '''
            # img_seg = Image.open(_seg_path)

            for c_idx, c_l in enumerate(cp_loc):
                cp_img_tum = img_tum.crop((c_l[0], c_l[1], c_l[0] + _cp_size[0], c_l[1] + _cp_size[1]))
                cp_tum_mask = Image.fromarray(pni).crop((c_l[0], c_l[1], c_l[0] + _cp_size[0], c_l[1] + _cp_size[1]))

                pad_val = utils.get_pad_val(cp_size, _cp_size)
                if pad_val[0]:
                    cp_img_tum = cp_img_tum.crop((pad_val[1]))
                    cp_tum_mask = cp_tum_mask.crop((pad_val[1]))

                cp_tum = (cp_img_tum, cp_tum_mask)
                """
                if cp_img_tum.size[0] < 400 and cp_img_tum.size[1] < 400:
                    continue
                """

                cp_img, L_cp_seg_mask, ov_cp_img = process_one(cfg, cp_tum, _seg_path_list)
                _ot_tag_itr = _ot_tag_2 + '_{:03d}_i{:02d}.png'.format(c_idx, itr)

                for _s_info in _s_info_list:
                    _s_name = _s_info[0]
                    print(_ot_tag_itr, _s_name)

                if L_cp_seg_mask is None:
                    null_tum_contours, hierarchy = cv2.findContours(pni, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(np_img, null_tum_contours, -1, (0, 255, 255), 1)
                    ov_null_img = Image.fromarray(np_img)
                    for _s_info in _s_info_list:
                        _s_name = _s_info[0]
                        _seg_tag = _s_name[:-8]
                        _seg_tag_s = _s_name[:-4]
                        _seg_path = os.path.join(_seg_dir, _seg_tag, _s_name)
                        _seg_path_list.append(_seg_path)
                        img_seg = Image.open(_seg_path)
                        ov_null_img.paste(img_seg, (0, 0), img_seg)

                    print('::Null process::')
                    ov_null_img.save(os.path.join(_save_dir_null, _ot_tag_itr))
                else:
                    if image_cfg:
                        cp_img.save(os.path.join(_save_dir_image, _ot_tag_itr))
                    if mask_cfg:
                        L_cp_seg_mask.save(os.path.join(_save_dir_mask, _ot_tag_itr))
                    if overlay_cfg:
                        ov_cp_img.save(os.path.join(_save_dir_ovly, _ot_tag_itr))
                    f.writelines(os.path.join(_save_dir_image, _ot_tag_itr) + ' 12\n')

        ### process only one nerve seg
        # continue

    d = 0


def create_list():
    d = 0


if __name__ == '__main__':
    # input
    _ori_tumor_dir = '../DATABANK/PAIP2021-dataset/imgnmask_l1_ori/bbox/13'
    _seg_dir = './seg_150'
    otseg_dict_path = 'npy/otseg_dict150.npy'
    SN_target1_path = './data/SN/target1.png'

    # output
    out_aug_dir = './data/paip_aug/Aug12'

    # process params
    cfg = {}
    cfg['itr_no'] = 1               # iteration count
    cfg['type'] = 'FS'              # FS or PI
    cfg['rotation_cnt'] = 10        # rotation count
    cfg['max_seg_count'] = 1        # max seg count
    cfg['cp_size'] = (512, 512)     # crop size

    overlay_cfg = True
    mask_cfg = True
    image_cfg = True
    stain_norm_cfg = True

    stain_norm_tag = ''
    if stain_norm_cfg:
        stain_norm_tag = 'SN'
    out_aug_dir = out_aug_dir +'_i'+str(cfg['itr_no'])+stain_norm_tag+'_s'+str(cfg['max_seg_count'])+'_'+cfg['type']
    main()
    # process_one(tumor_img_path, seg_img_path)

    d = 0
