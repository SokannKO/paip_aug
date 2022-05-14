import os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 5000000000
import cv2
import random

'''
seg composition
'''
class seg_comp():
    def __init__(self, path_params):
        self.path_params = path_params

        # input info
        self._seg_img_dir = path_params['_seg_img_dir']
        self._seg_mask_dir = path_params['_seg_mask_dir']
        self._ori_tumor_dir = path_params['_ori_tumor_dir']

        # output info
        self.npy_out_dir = path_params['npy_out_dir']
        self.seg_out_dir = path_params['seg_out_dir']
        self.ot_c_dict_path = os.path.join(self.npy_out_dir, path_params['ot_c_dict_f_name'])
        self.s_c_dict_path = os.path.join(self.npy_out_dir, path_params['s_c_dict_f_name'])
        self.otseg_dict_path = os.path.join(self.npy_out_dir, path_params['otseg_dict_f_name'])

        d = 0

    def run(self):
        # create npy dir
        os.makedirs(self.npy_out_dir, exist_ok=True)

        self.create_seg()
        self.create_seg_npy()
        self.distribute_seg3()

    # create seg only for dict
    def create_seg(self):
        seg_img_list = os.listdir(self._seg_img_dir)

        for _img_name in seg_img_list:
            img_path = os.path.join(self._seg_img_dir, _img_name)
            mask_path = os.path.join(self._seg_mask_dir, _img_name)
            tag = os.path.splitext(_img_name)[0]

            np_mask = np.array(Image.open(mask_path))
            contours, hierarchy = cv2.findContours(np_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            np_img_RGBA = np.array(Image.open(img_path).convert('RGBA'))
            np_img_RGBA[:,:,3] = np_mask*255

            new_img_RGBA = Image.fromarray(np_img_RGBA)

            _img_name_dir = os.path.join(self.seg_out_dir, tag)
            if not os.path.exists(_img_name_dir):
                os.makedirs(_img_name_dir)

            for idx, contour in enumerate(contours):
                x_flatten = contour[:,0][:,0]
                y_flatten = contour[:,0][:,1]
                x_min = x_flatten.min()
                x_max = x_flatten.max()
                y_min = y_flatten.min()
                y_max = y_flatten.max()

                cp_img = new_img_RGBA.crop((x_min, y_min, x_max, y_max))
                cp_img_name = '{0}_s{1:02}.png'.format(tag,idx)
                cp_img.save(os.path.join(_img_name_dir,cp_img_name))
                d = 0

            d = 0

    def create_seg_npy(self):
        #"""
        list_ot_dir = sorted(os.listdir(self._ori_tumor_dir))
        ot_c_dict = {}
        for _ot_img in list_ot_dir:
            _ot_img_path = os.path.join(self._ori_tumor_dir,_ot_img)
            _ot = Image.open(_ot_img_path)
            rz_11 = _ot.resize((1,1))
            ot_color = np.array(rz_11)[0][0].astype(int)
            ot_c_dict[_ot_img] = [ot_color, _ot.size]
            d = 0
        np.save(self.ot_c_dict_path, ot_c_dict)
        #"""


        list_s_dir = [f for f in sorted(os.listdir(self.seg_out_dir)) if f[0] != '.']
        s_c_dict = {}
        for _s_img_dir in list_s_dir:
            _s_img_dir_path = os.path.join(self.seg_out_dir, _s_img_dir)
            _s_img_dir_list = os.listdir(_s_img_dir_path)

            for _s_img in _s_img_dir_list:
                _s_img_path = os.path.join(_s_img_dir_path,_s_img)
                _s = Image.open(_s_img_path).convert('RGB')
                rz_11 = _s.resize((1,1))
                s_color = np.array(rz_11)[0][0].astype(int)
                s_c_dict[_s_img] = [s_color, _s.size]
            d = 0
        np.save(self.s_c_dict_path, s_c_dict)

        d = 0

    def distribute_seg3(self):
        '''
        process in the same slice
        distribute random multiple count of segs
        :return:
        '''

        ot_c_dict = np.load(self.ot_c_dict_path, allow_pickle=True)[()]
        s_c_dict = np.load(self.s_c_dict_path, allow_pickle=True)[()]

        list_ot_dir = sorted(ot_c_dict.keys())
        list_s_dir = sorted(s_c_dict.keys())
        #random.shuffle(list_s_dir)
        d = 0

        # init otseg_dict
        otseg_dict = {}
        for ot_info in ot_c_dict:
            otseg_dict[ot_info] = []

        init_len_s_c_list = len(list_s_dir)

        total_success = 0

        info_cfg = ('','')
        #"""

        for ot_info in list_ot_dir:
            if len(otseg_dict[ot_info]) > 2:
                continue

            ot_c_val = ot_c_dict[ot_info]

            if info_cfg != (ot_info.split('_')[0], ot_info.split('_')[3]):
                ot_slice_s_list = [f for f in list_s_dir if f.split('_')[0] == ot_info.split('_')[0] and\
                                f.split('_')[3] == ot_info.split('_')[3]]
                info_cfg = (ot_info.split('_')[0], ot_info.split('_')[3])
            join_list = []
            random.shuffle(ot_slice_s_list)
            for s_info in ot_slice_s_list:
                if len(join_list) > 3:
                    continue
                s_c_val = s_c_dict[s_info]
                x_sz, y_sz = (ot_c_val[1][0] - s_c_val[1][0], ot_c_val[1][1] - s_c_val[1][1])

                if x_sz >= 0 and y_sz >= 0:
                    #join_list.append(ot_c_val[0])
                    join_list.append([s_info,s_c_val[1]])
                    total_success += 1

            otseg_dict[ot_info] = otseg_dict[ot_info] + join_list

        print ('init_len = ' + str(init_len_s_c_list))
        print ('remain = ' + str(len(list_s_dir)))
        print ('success = ' + str(total_success))

        #for _elem in otseg_dict:
        #    print (str(ot_c_dict[_elem][1])+'   '+str(otseg_dict[_elem]))

        d = 0
        np.save(self.otseg_dict_path, otseg_dict)



if __name__ == '__main__':
    path_params = dict()
    path_params['_seg_img_dir'] = '../DATABANK/PAIP2021-dataset/imgnmask_l1/cntr/1/image'
    path_params['_seg_mask_dir'] = '../DATABANK/PAIP2021-dataset/imgnmask_l1/cntr/1/mask'
    path_params['_ori_tumor_dir'] = '../DATABANK/PAIP2021-dataset/imgnmask_l1_ori/bbox/13'

    path_params['npy_out_dir'] = './npy'    # save dir for dict
    path_params['seg_out_dir'] = './seg_150'    # save dir for seg images
    path_params['ot_c_dict_f_name'] = 'ot_c_list150.npy'    # tumor tile dict
    path_params['s_c_dict_f_name'] = 's_c_list150.npy'      # seg tile dict
    path_params['otseg_dict_f_name'] = 'otseg_dict150.npy'  # tumor/seg loc dict

    s = seg_comp(path_params)
    s.run()

    d = 0