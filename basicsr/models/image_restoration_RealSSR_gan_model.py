# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import numpy as np
import random
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.data.noise_dataset import noiseDataset
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.dist_util import get_dist_info
from basicsr.losses.LDL_loss import get_refined_artifact_map

loss_module = importlib.import_module('basicsr.losses.basic_loss')
metric_module = importlib.import_module('basicsr.metrics')

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    # cut_w = np.int(W * cut_rat)   #####
    # cut_h = np.int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


@MODEL_REGISTRY.register()
class ImageRestorationRealSSRGANModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationRealSSRGANModel, self).__init__(opt)

        # define network
        self.net_g = build_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        # define network net_d
        self.net_d = build_network(deepcopy(self.opt['network_d']))
        self.net_d = self.model_to_device(self.net_d)
        # self.print_network(self.net_d)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

##############################artifact##################################
        if train_opt.get('artifacts_opt'):
            artifacts_type = train_opt['artifacts_opt'].pop('type')
            cri_artifacts_cls = getattr(loss_module, artifacts_type)
            self.cri_artifacts = cri_artifacts_cls(
                **train_opt['artifacts_opt']).to(self.device)
        else:
            self.cri_artifacts = None

########################## Degradation Restoration loss ####################################

        if train_opt.get('DR_opt'):
            DR_type = train_opt['DR_opt'].pop('type')
            cri_DR_cls = getattr(loss_module, DR_type)
            self.cri_DR = cri_DR_cls(
                **train_opt['DR_opt']).to(self.device)
        else:
            self.cri_DR = None


        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data2(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    @torch.no_grad()
    def feed_data(self, data, is_val=False):
        self.gt = data['gt'].to(self.device)

        self.kernel1 = data['kernel1'].to(self.device)
        self.kernel2 = data['kernel2'].to(self.device)
        self.kernel3 = data['kernel3'].to(self.device)
        self.sinc_kernel = data['sinc_kernel'].to(self.device)

        self.gt_L, self.gt_R = self.gt.chunk(2, dim=1)
        self.gt_concat = torch.cat([self.gt_L, self.gt_R], dim=2)  # concat along H dimension,竖着拼

        # USM sharpen the GT images
        if self.opt['gt_usm'] is True:
            self.gt_concat = self.usm_sharpener(self.gt_concat)

        ori_h, ori_w = self.gt_concat.size()[2:4]
        # ----------------------- The first degradation process ----------------------- #
        # blur
        if np.random.uniform() < 0.5:
            out = filter2D(self.gt_concat, self.kernel1)
        else:
            out = self.gt_concat
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < 0.5:
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression
        if np.random.uniform() < 0.5:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            if np.random.uniform() < 0.5:  # left and right different degradation p=0.5?
                lq_l, lq_r = out.chunk(2, dim=2)
                out_l = filter2D(lq_l, self.kernel2)
                out_r = filter2D(lq_r, self.kernel3)
                out = torch.cat([out_l, out_r], dim=2)
            else:
                out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
        # add noise
        if np.random.uniform() < 0.5:
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < 0.5:  ### different
                lq_l, lq_r = out.chunk(2, dim=2)
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    lq_l = random_add_gaussian_noise_pt(lq_l, sigma_range=self.opt['noise_range2'], clip=True,
                                                        rounds=False,
                                                        gray_prob=gray_noise_prob)
                    lq_r = random_add_gaussian_noise_pt(lq_r, sigma_range=self.opt['noise_range2'], clip=True,
                                                        rounds=False,
                                                        gray_prob=gray_noise_prob)
                else:
                    lq_l = random_add_poisson_noise_pt(
                        lq_l,
                        scale_range= self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    lq_r = random_add_poisson_noise_pt(
                        lq_r,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

            else:
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            if np.random.uniform() < 0.5:
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            if np.random.uniform() < 0.5:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)

        # clamp and round
        self.lq_concat = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        self.lq_concat = self.lq_concat.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        self.lq_L, self.lq_R = self.lq_concat.chunk(2, dim=2)
        self.lq = torch.cat([self.lq_L, self.lq_R], dim=1)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_g_ema.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq)
        preds_ema = self.net_g_ema(self.lq)

        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(preds_ema, list):
            preds_ema = [preds_ema]

        self.output = preds[-1]
        self.output_ema = preds_ema[-1]


        l_g_total = 0
        loss_dict = OrderedDict()
        # if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
        if (current_iter % self.net_d_iters == 0):
            # pixel loss
            if self.cri_pix:
                l_pix = 0.
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)

                # print('l pix ... ', l_pix)
                l_g_total += l_pix
                loss_dict['l_pix'] = l_pix

            # perceptual loss
            if self.cri_perceptual:
                l_percep_l, l_style_l = self.cri_perceptual(self.output[:,:3], self.gt[:,:3])
                l_percep_r, l_style_r = self.cri_perceptual(self.output[:,3:], self.gt[:,3:])

                if l_percep_l is not None:
                    l_percep = l_percep_l + l_percep_r
                    l_g_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style_l is not None:
                    l_style = l_style_l + l_style_r
                    l_g_total += l_style
                    loss_dict['l_style'] = l_style

            # artifact loss(LDL)
            if self.cri_artifacts:
                pixel_weight_l = get_refined_artifact_map(self.gt[:,:3], self.output[:,:3], self.output_ema[:,:3], 7)
                l_g_artifacts_l = self.cri_artifacts(torch.mul(pixel_weight_l, self.output[:,:3]), torch.mul(pixel_weight_l, self.gt[:,:3]))
                pixel_weight_r = get_refined_artifact_map(self.gt[:,3:], self.output[:,3:], self.output_ema[:,3:], 7)
                l_g_artifacts_r = self.cri_artifacts(torch.mul(pixel_weight_r, self.output[:,3:]), torch.mul(pixel_weight_r, self.gt[:,3:]))
                if l_g_artifacts_l is not None and l_g_artifacts_r is not None:
                    l_g_artifacts = l_g_artifacts_l + l_g_artifacts_r
                l_g_total += l_g_artifacts
                loss_dict['l_g_artifacts'] = l_g_artifacts

#####################   DR loss  ##################################
            if self.cri_DR:
                l_DR_left = self.cri_DR(self.output[:, :3], self.gt[:, :3])
                l_DR_right = self.cri_DR(self.output[:, 3:], self.gt[:, 3:])
                l_DR = l_DR_left + l_DR_right
                l_g_total += l_DR

                loss_dict['l_DR'] = l_DR


            # gan loss (relativistic gan)
            real_d_mid, real_d_pred, e_f_h, d_f_h = self.net_d(self.gt)
            real_d_mid, real_d_pred = real_d_mid.detach(), real_d_pred.detach()
            fake_g_mid, fake_g_pred, e_f_s, d_f_s = self.net_d(self.output)
            l_g_real_mid = self.cri_gan(real_d_mid - torch.mean(fake_g_mid), False, is_disc=False)
            l_g_fake_mid = self.cri_gan(fake_g_mid - torch.mean(real_d_mid), True, is_disc=False)
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2 + (l_g_real_mid + l_g_fake_mid) / 2

            l_g_fms = []
            for f in range(3):
                l_g_fms += [self.cri_pix(e_f_s[f], e_f_h[f])]
                l_g_fms += [self.cri_pix(d_f_s[f], d_f_h[f])]
            l_g_fm = torch.mean(torch.stack(l_g_fms))

            l_g_total += l_g_gan + l_g_fm
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['l_g_fm'] = l_g_fm

            l_g_total = l_g_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_g_total.backward()
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        fake_d_mid, fake_d_pred, _, _ = self.net_d(self.output)
        fake_d_mid, fake_d_pred = fake_d_mid.detach(), fake_d_pred.detach()
        real_d_mid, real_d_pred, _, _ = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real += self.cri_gan(real_d_mid - torch.mean(fake_d_mid), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        output_cutmix = self.output.clone()
        if torch.rand(1) <= 0.5:  ### cutmix
            r_mix = torch.rand(1)   # real/fake ratio

            bbx1, bby1, bbx2, bby2 = rand_bbox(output_cutmix.size(), r_mix)
            output_cutmix[:, :, bbx1:bbx2, bby1:bby2] = self.gt[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (output_cutmix.size()[-1] * output_cutmix.size()[-2]))

            fake_d_mid_mix, fake_d_pred_mix, _, _ = self.net_d(output_cutmix.detach())
            l_d_fake = self.cri_gan(fake_d_pred_mix - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            l_d_fake += self.cri_gan(fake_d_mid_mix - torch.mean(real_d_mid.detach()), False, is_disc=True) * 0.5

            fake_d_pred[:,:,bbx1:bbx2, bby1:bby2] = real_d_pred[:,:,bbx1:bbx2, bby1:bby2]
            l_d_fake += F.mse_loss(fake_d_pred_mix.detach(), fake_d_pred.detach())
        else:
            fake_d_mid, fake_d_pred, _, _ = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            l_d_fake += self.cri_gan(fake_d_mid - torch.mean(real_d_mid.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        #### used in swin transformer ####
        # pad to multiplication of window_size
        window_size = self.opt['window_size'] # 16
        scale = self.opt.get('scale', 1)
        _, _, h, w = self.lq.size()
        mod_pad_h = (h // window_size + 1) * window_size - h
        mod_pad_w = (w // window_size + 1) * window_size - w
        img = torch.cat([self.lq, torch.flip(self.lq, [2])], 2)[:, :, :h + mod_pad_h, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        #####  used in swin transformer ######
        #######  very important ########
        self.output = self.output[..., :h * scale, :w * scale]

    # def test_selfensemble(self):
    #     # TODO: to be tested
    #     # 8 augmentations
    #     # modified from https://github.com/thstkdgus35/EDSR-PyTorch
    #
    #     def _transform(v, op):
    #         # if self.precision != 'single': v = v.float()
    #         v2np = v.data.cpu().numpy()
    #         if op == 'v':
    #             tfnp = v2np[:, :, :, ::-1].copy()
    #         elif op == 'h':
    #             tfnp = v2np[:, :, ::-1, :].copy()
    #         elif op == 't':
    #             tfnp = v2np.transpose((0, 1, 3, 2)).copy()
    #
    #         ret = torch.Tensor(tfnp).to(self.device)
    #         # if self.precision == 'half': ret = ret.half()
    #
    #         return ret
    #
    #     # prepare augmented data
    #     lq_list = [self.lq]
    #     for tf in 'v', 'h', 't':
    #         lq_list.extend([_transform(t, tf) for t in lq_list])
    #
    #     # inference
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug)[0] for aug in lq_list]
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g(aug)[0] for aug in lq_list]
    #         self.net_g.train()
    #
    #     # merge results
    #     for i in range(len(out_list)):
    #         if i > 3:
    #             out_list[i] = _transform(out_list[i], 't')
    #         if i % 4 > 1:
    #             out_list[i] = _transform(out_list[i], 'h')
    #         if (i % 4) % 2 == 1:
    #             out_list[i] = _transform(out_list[i], 'v')
    #     output = torch.cat(out_list, dim=0)
    #
    #     self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            if 'lpips' in self.opt['val']['metrics'] or 'score' in self.opt['val']['metrics']:
                from basicsr.metrics.ntire_score import NTIRE_score
                score_metric = NTIRE_score(self.opt['val']['metrics']['score']['fast']).cuda()
                self.opt['val']['metrics']['lpips']['calculate_lpips'] = score_metric.cal_lpips
                self.opt['val']['metrics']['score']['calculate_score'] = score_metric

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data2(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        if 'lpips' in metric_type or 'score' in metric_type:
                            with torch.no_grad():
                            #     self.metric_results[name] += getattr(
                            #     metric_module, metric_type)(visuals['result'], visuals['gt'], lpips_vgg, **opt_).mean().cpu().numpy()
                                self.metric_results[name] += self.opt['val']['metrics'][name][metric_type](visuals['result'], visuals['gt'])\
                                    .mean().detach().cpu().numpy()
                        else:
                            self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)

                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

            keys = []
            metrics = []
            for name, value in self.collected_metrics.items():
                keys.append(name)
                metrics.append(value)
            metrics = torch.stack(metrics, 0)
            torch.distributed.reduce(metrics, dst=0)
            if self.opt['rank'] == 0:
                metrics_dict = {}
                cnt = 0
                for key, metric in zip(keys, metrics):
                    if key == 'cnt':
                        cnt = float(metric)
                        continue
                    metrics_dict[key] = float(metric)

                for key in metrics_dict:
                    metrics_dict[key] /= cnt

                self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                                tb_logger, metrics_dict)
        return 0.


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(ImageRestorationRealSSRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
