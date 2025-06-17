import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.archs.arch_util import ResidualBlockNoBN, default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DIEM(nn.Module):
    """Degradation Estimator with ResNetNoBN arch. v2.1, no vector anymore

    As shown in paper 'Towards Flexible Blind JPEG Artifacts Removal',
    resnet arch works for image quality estimation.

    Args:
        num_in_ch (int): channel number of inputs. Default: 3.
        num_degradation (int): num of degradation the DE should estimate. Default: 2(blur+noise).
        degradation_embed_size (int): embedding size of each degradation vector.
        degradation_degree_actv (int): activation function for degradation degree scalar. Default: sigmoid.
        num_feats (list): channel number of each stage.
        num_blocks (list): residual block of each stage.
        downscales (list): downscales of each stage.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_degradation=2,
                 degradation_degree_actv='sigmoid',
                 num_feats=[64, 64, 64, 128],
                 num_blocks=[2, 2, 2, 2],
                 downscales=[1, 1, 2, 1]):
        super(DIEM, self).__init__()

        assert isinstance(num_feats, list)
        assert isinstance(num_blocks, list)
        assert isinstance(downscales, list)
        assert len(num_feats) == len(num_blocks) and len(num_feats) == len(downscales)

        num_stage = len(num_feats)

        self.conv_first = nn.ModuleList()
        for _ in range(num_degradation):
            self.conv_first.append(nn.Conv2d(num_in_ch, num_feats[0], 3, 1, 1))
        self.body = nn.ModuleList()
        for _ in range(num_degradation):
            body = list()
            for stage in range(num_stage):
                for _ in range(num_blocks[stage]):
                    body.append(ResidualBlockNoBN(num_feats[stage]))
                if downscales[stage] == 1:
                    if stage < num_stage - 1 and num_feats[stage] != num_feats[stage + 1]:
                        body.append(nn.Conv2d(num_feats[stage], num_feats[stage + 1], 3, 1, 1))
                    continue
                elif downscales[stage] == 2:
                    body.append(nn.Conv2d(num_feats[stage], num_feats[min(stage + 1, num_stage - 1)], 3, 2, 1))
                else:
                    raise NotImplementedError
            self.body.append(nn.Sequential(*body))

        # self.body = nn.Sequential(*body)

        self.num_degradation = num_degradation
        self.fc_degree = nn.ModuleList()
        if degradation_degree_actv == 'sigmoid':
            actv = nn.Sigmoid
        elif degradation_degree_actv == 'tanh':
            actv = nn.Tanh
        else:
            raise NotImplementedError(f'only sigmoid and tanh are supported for degradation_degree_actv, '
                                      f'{degradation_degree_actv} is not supported yet.')
        for _ in range(num_degradation):
            self.fc_degree.append(
                nn.Sequential(
                    nn.Linear(num_feats[-1], 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 1),
                    actv(),
                ))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        default_init_weights([self.conv_first, self.body, self.fc_degree], 0.1)

    def forward(self, x):
        degrees = []
        for i in range(self.num_degradation):
            x_out = self.conv_first[i](x)
            feat = self.body[i](x_out)
            feat = self.avg_pool(feat)
            feat = feat.squeeze(-1).squeeze(-1)
            degrees.append(self.fc_degree[i](feat).squeeze(-1))

        return degrees


@ARCH_REGISTRY.register()
class DIEM_get_intensity(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 scale=4,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 de_net_type='DIEM',
                 **kwargs):
        super(DIEM_get_intensity, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        num_degradation = kwargs['num_degradation']
        self.num_degradation = num_degradation
        self.num_block = num_block
        # degradation net
        kwargs['num_in_ch'] = num_in_ch
        self.de_net = ARCH_REGISTRY.get(de_net_type)(**kwargs)

    def forward(self, x):
        intensity = self.de_net(x)
        return intensity
    

def get_intensity(img):
    ### input image is already a tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DIEM_get_intensity(num_in_ch=3, num_out_ch=3, num_feat=64, scale=4, num_degradation=2)
    loadnet = torch.load('/experiments/pretrained_models/DIEM.pth', map_location=torch.device('cuda:0')) # DIEM MODEL
    model.load_state_dict(loadnet['params'], strict=False)
    model.to('cuda:0')
    model.eval()
    with torch.no_grad():
        intensity = model(img)  ###intensity is a list,containing intensity_blur(tensor) and intensity_noise(tensor)
    return intensity    
    

if __name__ == '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    x = torch.randn(1, 3, 128, 128)
    model = DIEM()
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)