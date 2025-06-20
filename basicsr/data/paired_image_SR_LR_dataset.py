# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize, resize
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_hw, unpaired_random_crop
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img
import os
import math
import numpy as np
import glob
from scipy.io import loadmat
import torch
import random
import cv2
from degradation_pool.imresize import imresize
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.data.noise_dataset import noiseDataset

class PairedImageSRLRDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageSRLRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            import os
            # print(self.lq_folder)
            nums_lq = len(os.listdir(self.lq_folder))
            nums_gt = len(os.listdir(self.gt_folder))

            # nums_lq = sorted(nums_lq)
            # nums_gt = sorted(nums_gt)

            # print('lq gt ... opt')
            # print(nums_lq, nums_gt, opt)
            # assert nums_gt == nums_lq

            self.nums = nums_lq
            # {:04}_L   {:04}_R


            # self.paths = paired_paths_from_folder(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #     self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # print(index)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']

        gt_path_L = os.path.join(self.gt_folder, '{:04}_L.png'.format(index + 1))
        gt_path_R = os.path.join(self.gt_folder, '{:04}_R.png'.format(index + 1))


        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))


        lq_path_L = os.path.join(self.lq_folder, '{:04}_L.png'.format(index + 1))
        lq_path_R = os.path.join(self.lq_folder, '{:04}_R.png'.format(index + 1))

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))



        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size
            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # padding
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
            #                                     gt_path_L)
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
                                                'gt_path_L_and_R')
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
            #                          self.opt['use_rot'])
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=False)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # if scale != 1:
        #     c, h, w = img_lq.shape
        #     img_lq = resize(img_lq, [h*scale, w*scale])
            # print('img_lq .. ', img_lq.shape, img_gt.shape)

        # print('img_lq shape is:', img_lq.shape)
        # print('img_gt shape is:', img_gt.shape)
        # img_lq_tensor = torch.cat((img_lq, img_lq), dim=0)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': f'{index+1:04}',
            'gt_path': f'{index+1:04}',
        }

    def __len__(self):
        return self.nums // 2


class PairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(PairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)
        # print('nums of pictures:',self.nums)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # print(index)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')
        # gt_path_L = os.path.join(self.gt_folder, f'{index+1:04d}_L.png')
        # gt_path_R = os.path.join(self.gt_folder, f'{index+1:04d}_R.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')
        # lq_path_L = os.path.join(self.lq_folder, f'{index+1:04d}_L.png')
        # lq_path_R = os.path.join(self.lq_folder, f'{index+1:04d}_R.png')

        # lq_path = self.paths[index]['lq_path']

        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
            #                                     gt_path_L)


            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
                                                'gt_path_L_and_R')

            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


            img_gt, img_lq = imgs

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # print('lq111111111111111', type(img_lq), img_lq.shape) # tensor, 6*30*90

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
            # 'folder_name_index':self.lq_files[index]
        }

    def __len__(self):
        return self.nums


###### original ########
class PairedStereoImageDataset0(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(PairedStereoImageDataset0, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)
        # print('nums of pictures:',self.nums)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        #print(index)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')
        # gt_path_L = os.path.join(self.gt_folder, f'{index+1:04d}_L.png')
        # gt_path_R = os.path.join(self.gt_folder, f'{index+1:04d}_R.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')
        # lq_path_L = os.path.join(self.lq_folder, f'{index+1:04d}_L.png')
        # lq_path_R = os.path.join(self.lq_folder, f'{index+1:04d}_R.png')

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
                                                'gt_path_L_and_R')
            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


            img_gt, img_lq = imgs

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_folder, self.lq_files[index]),
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index]),
        }

    def __len__(self):
        return self.nums


class UnPairedStereoImageDataset(data.Dataset):
    '''
    unPaired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(UnPairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        # self.lq_files = os.listdir(self.lq_folder)
        self.gt_files = os.listdir(self.gt_folder)

        self.nums = len(self.gt_files)

############################################################
        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']

        self.blur_kernel_size3 = opt['blur_kernel_size3']
        self.kernel_list3 = opt['kernel_list3']
        self.kernel_prob3 = opt['kernel_prob3']
        self.blur_sigma3 = opt['blur_sigma3']
        self.betag_range3 = opt['betag_range3']
        self.betap_range3 = opt['betap_range3']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')
        # gt_path_L = os.path.join(self.gt_folder, f'{index+1:04d}_L.png')
        # gt_path_R = os.path.join(self.gt_folder, f'{index+1:04d}_R.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)

        scale = self.opt['scale']
        # augmentation for training
        # if self.opt['phase'] == 'train':
        #     if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
        #         gt_size_h = int(self.opt['gt_size_h'])
        #         gt_size_w = int(self.opt['gt_size_w'])
        #     else:
        #         gt_size = int(self.opt['gt_size'])
        gt_size = self.opt['gt_size']

        if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
            idx = [
                [0, 1, 2, 3, 4, 5],
                [0, 2, 1, 3, 5, 4],
                [1, 0, 2, 4, 3, 5],
                [1, 2, 0, 4, 5, 3],
                [2, 0, 1, 5, 3, 4],
                [2, 1, 0, 5, 4, 3],
            ][int(np.random.rand() * 6)]
            img_gt = img_gt[:, :, idx]

        # flip, rotation
        img_gt, status = augment(img_gt, self.opt['use_hflip'],
                               self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


################  realesrgan ####################
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < 0.1:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        if kernel.shape[0] == 21:
            kernel = kernel
        else:
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < 0.1:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        if kernel2.shape[0] == 21:
            kernel2 = kernel2
        else:
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

################## kernel 3 #################
        kernel_size = random.choice(self.kernel_range)
        kernel3 = random_mixed_kernels(
            self.kernel_list3,
            self.kernel_prob3,
            kernel_size,
            self.blur_sigma3,
            self.blur_sigma3, [-math.pi, math.pi],
            self.betag_range3,
            self.betap_range3,
            noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        if kernel3.shape[0] == 21:
            kernel3 = kernel3
        else:
            kernel3 = np.pad(kernel3, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < 0.8:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor


        img_gt = unpaired_random_crop(img_gt, gt_size, scale, 'gt_path_L_and_R')
        img_gt = img2tensor(img_gt, bgr2rgb=False, float32=True)

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        kernel3 = torch.FloatTensor(kernel3)

        return {
            'gt': img_gt,
            'kernel1': kernel,
            'kernel2': kernel2,
            'kernel3': kernel3,
            'sinc_kernel': sinc_kernel,
            'gt_path': os.path.join(self.gt_folder, self.gt_files[index])
        }

    def __len__(self):
        return self.nums

