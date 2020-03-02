import torch

from rcnn.modeling.hier_rcnn import heads
from rcnn.modeling.hier_rcnn import outputs
from rcnn.modeling.hier_rcnn.inference import hier_post_processor
from rcnn.modeling.hier_rcnn.loss import hier_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class HierRCNN(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(HierRCNN, self).__init__()
        if len(cfg.HRCNN.ROI_STRIDES) == 0:
            self.spatial_scale = spatial_scale
        else:
            self.spatial_scale = [1. / stride for stride in cfg.HRCNN.ROI_STRIDES]

        head = registry.ROI_HIER_HEADS[cfg.HRCNN.ROI_HIER_HEAD]
        self.Head = head(dim_in, self.spatial_scale)
        output = registry.ROI_HIER_OUTPUTS[cfg.HRCNN.ROI_HIER_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = hier_post_processor()
        self.loss_evaluator = hier_loss_evaluator()

    def forward(self, conv_features, proposals, targets=None):
        if self.training:
            return self._forward_train(conv_features, proposals, targets)
        else:
            return self._forward_test(conv_features, proposals)

    def _forward_train(self, conv_features, proposals, targets=None):
        all_proposals = proposals
        with torch.no_grad():
            proposals = self.loss_evaluator.subsample(proposals, targets)

        x = self.Head(conv_features, proposals)
        box_cls, box_regression, centerness, locations = self.Output(x)

        loss_cls, loss_reg, loss_centerness = self.loss_evaluator(box_cls, box_regression, centerness, locations)
        loss_dict = dict(loss_hier_cls=loss_cls, loss_hier_reg=loss_reg, loss_hier_centerness=loss_centerness)

        return None, all_proposals, loss_dict

    def _forward_test(self, conv_features, proposals):
        x = self.Head(conv_features, proposals)
        box_cls, box_regression, centerness, locations = self.Output(x)

        result = self.post_processor(box_cls, box_regression, centerness, locations, proposals)
        return None, result, {}
