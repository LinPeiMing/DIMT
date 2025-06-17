import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.local_arch import Local_Base
from basicsr.archs.arch_util import trunc_normal_, LayerNorm2d, default_init_weights
from basicsr.archs.DIMT_utils import PatchEmbed, PatchUnEmbed, Upsample, HAB, OCAB, window_partition, SFB
from basicsr.archs.DIEM_arch import get_intensity


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class Fusion(nn.Module):
    def __init__(self, c):
        super(Fusion, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,groups=1, bias=True)
        )

    def forward(self, c, i):
        x = self.sca(i) * c + i
        return x


class ConditionNet(nn.Module):
    def __init__(self, num_feat=96):
        super(ConditionNet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(True))
        
        self.fc = nn.Sequential(
            nn.Linear(512, (512 + num_feat * 2) // 2), # 512 to 352
            nn.ReLU(True),
            nn.Linear((512 + num_feat * 2) // 2, (512 + num_feat * 2) // 2), # 352 to 352
            nn.ReLU(True),
            nn.Linear((512 + num_feat * 2) // 2, num_feat * 2), # 352 to 192
        )
        default_init_weights([self.fc], 0.1)
        default_init_weights([self.embedding], 0.1)

    def forward(self, d):
        # now d is b*2
        d = self.embedding(d) # 2 to 512, now d is b*512
        d = self.fc(d) # 512 to 192, now d is b*192
        return d


class AttenBlocks(nn.Module):
    """ A series of attention blocks for one SFTB.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # OCAB
        self.overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.patch_embed = PatchEmbed(img_size=48, patch_size=1, in_chans=3, embed_dim = 64,norm_layer = nn.LayerNorm)
        self.patch_unembed = PatchUnEmbed(img_size=48, patch_size=1, in_chans=3, embed_dim = 64, norm_layer = nn.LayerNorm)

    def forward(self, x, x_size, params):

        for blk in self.blocks:

            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])
        x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SFTB(nn.Module):
    """ Swin Fourier Transformer Block (SFTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='SFB'):
        super(SFTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'SFB':
            self.conv = SFB(dim)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class DMB(nn.Module):
    def __init__(self, c):
        super(DMB, self).__init__()
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.sg = nn.GELU()

        self.conv_down = nn.Conv2d(c, c // 8, kernel_size=1, stride=1, padding=0)
        self.conv_up_l = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0)
        self.conv_up_r = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.encode_l = ConditionNet(num_feat=c)
        self.encode_r = ConditionNet(num_feat=c)

    def forward(self, x_l, x_r, d_l, d_r):
        d_l = self.encode_l(d_l)
        d_r = self.encode_r(d_r)
        d_l = d_l.view(d_l.size(0), d_r.size(1), 1, 1) # now d is b*192*1*1
        d_r = d_r.view(d_r.size(0), d_r.size(1), 1, 1)

        gamma1, beta1 = torch.chunk(d_l, chunks=2, dim=1)
        gamma2, beta2 = torch.chunk(d_r, chunks=2, dim=1)
        x_l = (1 + gamma1) * self.norm_l(x_l) + beta1
        x_r = (1 + gamma2) * self.norm_r(x_r) + beta2

        x = self.sg(self.conv_down(self.gap(x_l + x_r)))
        x_l2 = torch.sigmoid(self.conv_up_l(x)) * x_l + x_l
        x_r2 = torch.sigmoid(self.conv_up_r(x)) * x_r + x_r
        return x_l2, x_r2


class DASIM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.criterion = nn.L1Loss()

        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.patch_embed = PatchEmbed(img_size=256, patch_size=1, in_chans=0, embed_dim=c, norm_layer=None)
        self.fusion_L = Fusion(c)
        self.fusion_R = Fusion(c)
        self.selection = DMB(c)

    def forward(self, x, x_size, d_l, d_r):
        h, w = x_size
        b, _, c = x.shape

        x = x.view(b, h, w, c).permute(0, 3, 1, 2)  # [2b, c, h, w]
        x_l = x[:b // 2]
        x_r = x[b // 2:]

        x_left, x_right = self.selection(x_l, x_r, d_l, d_r)
        Q_l = x_left.permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = x_right.permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c
        # scale
        F_l = self.fusion_L(F_r2l.permute(0, 3, 1, 2), x_l)
        F_r = self.fusion_R(F_l2r.permute(0, 3, 1, 2), x_r)

        out = torch.cat([F_l, F_r], 0)  # original is 0
        out = self.patch_embed(out)
        return out


@ARCH_REGISTRY.register()
class DIMTnet(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(4, 4, 4, 4), #setting here
                 num_heads=(4, 4, 4, 4),
                 window_size=16,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='SFB',
                 **kwargs):
        super(DIMTnet, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64 
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Swin Fourier Transformer Block (SFTB)

        self.layers = nn.ModuleList()
        self.scam_layer = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SFTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)

            scam_layer = DASIM(embed_dim)
            self.scam_layer.append(scam_layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, d_l, d_r):
        x_size = (x.shape[2], x.shape[3])
        # p = self.compress(p) # p:2*256->2*96
        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-cosuming for large window size.
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, x_size, params)
            x = self.scam_layer[i](x, x_size, d_l, d_r)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, input):

        x = input[0] #1*6*64*64
        d_l = input[1]
        d_r = input[2]

        b, c, h, w = x.shape # 1*6*64*64

        # [B, 2C, H, W] -> [2B, C, H, W]
        x_l = x[:, :c // 2]
        x_r = x[:, c // 2:]
        x = torch.cat([x_l, x_r], 0)


        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, d_l, d_r)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, d_l, d_r)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        #[2B, C, H, W] -> [B, 2C, H, W]  2*3*256*256 -> 1*6*256*256
        x = torch.cat([x[:b], x[b:]], 1)

        return x


# @ARCH_REGISTRY.register()
class DIMTnet_Local(Local_Base, DIMTnet):
    def __init__(self, *args, train_size=(1, 6, 64, 64), d_l_size=(1, 2), d_r_size=(1, 2), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        DIMTnet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, d_l_size=d_l_size, d_r_size=d_r_size, fast_imp=fast_imp)


@ARCH_REGISTRY.register()
class DIMT(nn.Module):
    def __init__(self):
        super(DIMT, self).__init__()
        self.G = DIMTnet_Local() 

    def forward(self, x):
        #x: lq_left concat lq_right
        x_l, x_r = x.chunk(2, dim=1)
        score_lr_l = get_intensity(x_l) # score_lr_l is a list with 2 tensor
        score_lr_r = get_intensity(x_r)
        d_lr_l = torch.stack([score_lr_l[0], score_lr_l[1]], dim=1) # now shape is b*2
        d_lr_r = torch.stack([score_lr_r[0], score_lr_r[1]], dim=1)

        sr = self.G([x, d_lr_l, d_lr_r])
        return sr


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    x1 = torch.randn(1, 6, 128, 128).cuda()
    d_l = torch.randn(1, 2).cuda()
    d_r = torch.randn(1, 2).cuda()
    x = [x1, d_l, d_r]
    model = DIMT().cuda()
    print({sum(map(lambda x: x.numel(), model.parameters()))})
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))


