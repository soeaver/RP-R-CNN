import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.net import make_conv
from models.ops import Conv2d, ConvTranspose2d, interpolate, Scale
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_HIER_OUTPUTS.register("hier_output")
class Hier_output(nn.Module):
    def __init__(self, dim_in):
        super(Hier_output, self).__init__()

        num_classes = cfg.HRCNN.NUM_CLASSES
        num_convs = cfg.HRCNN.OUTPUT_NUM_CONVS
        conv_dim = cfg.HRCNN.OUTPUT_CONV_DIM
        use_lite = cfg.HRCNN.OUTPUT_USE_LITE
        use_bn = cfg.HRCNN.OUTPUT_USE_BN
        use_gn = cfg.HRCNN.OUTPUT_USE_GN
        use_dcn = cfg.HRCNN.OUTPUT_USE_DCN
        prior_prob = cfg.HRCNN.PRIOR_PROB

        self.norm_reg_targets = cfg.HRCNN.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.HRCNN.CENTERNESS_ON_REG

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            conv_type = 'deform' if use_dcn and i == num_convs - 1 else 'normal'
            cls_tower.append(
                make_conv(dim_in, conv_dim, kernel=3, stride=1, dilation=1, use_dwconv=use_lite,
                          conv_type=conv_type, use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=use_lite)
            )
            bbox_tower.append(
                make_conv(dim_in, conv_dim, kernel=3, stride=1, dilation=1, use_dwconv=use_lite,
                          conv_type=conv_type, use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=use_lite)
            )
            dim_in = conv_dim

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_deconv = ConvTranspose2d(conv_dim, conv_dim, 2, 2, 0)
        self.bbox_deconv = ConvTranspose2d(conv_dim, conv_dim, 2, 2, 0)
        self.cls_logits = Conv2d(
            conv_dim, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = Conv2d(
            conv_dim, 4, kernel_size=3, stride=1, padding=1
        )
        self.centerness = Conv2d(
            conv_dim, 1, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize the bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(1)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        cls_tower = self.cls_tower(x)
        box_tower = self.bbox_tower(x)

        _cls_tower = F.relu(self.cls_deconv(cls_tower))
        _box_tower = F.relu(self.bbox_deconv(box_tower))
        logits.append(self.cls_logits(_cls_tower))

        if self.centerness_on_reg:
            centerness.append(self.centerness(_box_tower))
        else:
            centerness.append(self.centerness(_cls_tower))

        bbox_pred = self.scales[0](self.bbox_pred(_box_tower))
        if self.norm_reg_targets:
            bbox_pred = F.relu(bbox_pred)
            if self.training:
                bbox_reg.append(bbox_pred)
            else:
                bbox_reg.append(bbox_pred * 0.5)
        else:
            bbox_reg.append(torch.exp(bbox_pred))

        locations = compute_locations(centerness, [0.5])

        return logits, bbox_reg, centerness, locations


def compute_locations(features, strides):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locations_per_level = compute_locations_per_level(
            h, w, strides[level],
            feature.device
        )
        locations.append(locations_per_level)
    return locations


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations
