import numpy as np
import cv2

import torch
from torch import nn

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.boxlist_ops import cat_boxlist
from utils.data.structures.boxlist_ops import boxlist_nms, boxlist_ml_nms, boxlist_soft_nms, boxlist_box_voting
from utils.data.structures.boxlist_ops import remove_small_boxes
from rcnn.core.config import cfg


class HierPostProcessor(nn.Module):
    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, min_size, num_classes,
                 resolution, eval_hier, hier_th):
        super(HierPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.resolution = resolution
        self.eval_hier = eval_hier
        self.hier_th = hier_th

    def forward_for_single_feature_map(self, locations, box_cls, box_regression, centerness, boxes):
        """
        Arguments:
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, self.num_classes).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = self.get_det_result(locations, box_cls, box_regression, boxes)
        if self.eval_hier:
            hier_boxes, hier_scores = self.get_hier_result(locations, box_cls, box_regression, boxes)
            return results, hier_boxes, hier_scores
        else:
            return results, None, None

    def get_det_result(self, locations, box_cls, box_regression, boxes):
        N = len(box_cls)
        h, w = self.resolution

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        _boxes = boxes.bbox
        size = boxes.size
        boxes_scores = boxes.get_field("scores")

        results = []
        for i in range(N):
            box = _boxes[i]
            boxes_score = boxes_scores[i]
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 2

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            _x1 = per_locations[:, 0] - per_box_regression[:, 0]
            _y1 = per_locations[:, 1] - per_box_regression[:, 1]
            _x2 = per_locations[:, 0] + per_box_regression[:, 2]
            _y2 = per_locations[:, 1] + per_box_regression[:, 3]

            _x1 = _x1 / w * (box[2] - box[0]) + box[0]
            _y1 = _y1 / h * (box[3] - box[1]) + box[1]
            _x2 = _x2 / w * (box[2] - box[0]) + box[0]
            _y2 = _y2 / h * (box[3] - box[1]) + box[1]

            detections = torch.stack([_x1, _y1, _x2, _y2], dim=-1)

            boxlist = BoxList(detections, size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(torch.sqrt(per_box_cls) * boxes_score))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        results = cat_boxlist(results)

        return results

    def get_hier_result(self, locations, box_cls, box_regression, boxes):
        N = len(box_cls)
        h, w = self.resolution

        _boxes = boxes.bbox
        boxes_scores = boxes.get_field("scores")

        _hier_boxes = []
        _hier_scores = []
        for i in range(N):
            box = _boxes[i]
            boxes_score = boxes_scores[i]
            per_box_cls = box_cls[i]
            per_box_cls, per_box_loc = per_box_cls.max(dim=0)

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            _x1 = per_locations[:, 0] - per_box_regression[:, 0]
            _y1 = per_locations[:, 1] - per_box_regression[:, 1]
            _x2 = per_locations[:, 0] + per_box_regression[:, 2]
            _y2 = per_locations[:, 1] + per_box_regression[:, 3]

            _x1 = _x1 / w * (box[2] - box[0]) + box[0]
            _y1 = _y1 / h * (box[3] - box[1]) + box[1]
            _x2 = _x2 / w * (box[2] - box[0]) + box[0]
            _y2 = _y2 / h * (box[3] - box[1]) + box[1]

            detections = torch.stack([_x1, _y1, _x2, _y2], dim=-1)

            per_box_cls[per_box_cls < self.hier_th] = 0
            _hier_boxes.append(detections)
            _hier_scores.append(torch.sqrt(torch.sqrt(per_box_cls) * boxes_score))

        _hier_boxes = torch.stack(_hier_boxes, dim=0)
        _hier_scores = torch.stack(_hier_scores, dim=0)
        return _hier_boxes, _hier_scores

    def forward(self, box_cls_all, box_reg_all, centerness_all,  locations, boxes_all):
        fea_level_num = len(box_cls_all)
        boxes_per_image = [len(box) for box in boxes_all]
        cls = [box_cls.split(boxes_per_image, dim=0) for box_cls in box_cls_all]
        reg = [box_reg.split(boxes_per_image, dim=0) for box_reg in box_reg_all]
        center = [centerness.split(boxes_per_image, dim=0) for centerness in centerness_all]
        cls = list(zip(*cls))
        reg = list(zip(*reg))
        center = list(zip(*center))

        results = []
        hier_results = []
        for box_cls, box_regression, centerness, boxes in zip(cls, reg, center, boxes_all):
            sampled_boxes = []
            hier_boxes = []
            hier_scores = []
            for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
                _results, _hier_boxes, _hier_scores = self.forward_for_single_feature_map(l, o, b, c, boxes)
                sampled_boxes.append(_results)
                hier_boxes.append(_hier_boxes)
                hier_scores.append(_hier_scores)

            sampled_boxes = cat_boxlist(sampled_boxes)
            results.append(sampled_boxes)

            if self.eval_hier:
                hier_boxes = torch.stack(hier_boxes, dim=2)
                hier_scores = torch.stack(hier_scores, dim=2)

                hier_boxes = hier_boxes.reshape(-1, fea_level_num, 4)
                hier_scores = hier_scores.reshape(-1, fea_level_num)

                per_box_cls, per_box_ind = hier_scores.max(dim=1)
                detections = hier_boxes[range(len(hier_boxes)), per_box_ind]

                hier_boxes = detections.cpu().numpy()
                hier_scores = per_box_cls.cpu().numpy()
                boxes_scores = boxes.get_field("scores")

                hier_results.append([hier_boxes, hier_scores, boxes_scores])

        results = self.select_over_all_levels(results)
        results = [cat_boxlist([result, boxes]) for result, boxes in zip(results, boxes_all)]

        if self.eval_hier:
            for result, hier_result in zip(results, hier_results):
                result.add_field("hier_boxes", hier_result[0])
                result.add_field("hier_scores", hier_result[1])
                result.add_field("pboxes_scores", hier_result[2])

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            if not cfg.TEST.SOFT_NMS.ENABLED and not cfg.TEST.BBOX_VOTE.ENABLED:
                # multiclass nms
                result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            else:
                scores = boxlists[i].get_field("scores")
                labels = boxlists[i].get_field("labels")
                boxes = boxlists[i].bbox
                boxlist = boxlists[i]
                result = []
                # skip the background
                for j in range(2, self.num_classes + 1):
                    inds = (labels == j).nonzero().view(-1)

                    scores_j = scores[inds]
                    boxes_j = boxes[inds, :].view(-1, 4)
                    boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                    boxlist_for_class.add_field("scores", scores_j)
                    boxlist_for_class_old = boxlist_for_class
                    if cfg.TEST.SOFT_NMS.ENABLED:
                        boxlist_for_class = boxlist_soft_nms(
                            boxlist_for_class,
                            sigma=cfg.TEST.SOFT_NMS.SIGMA,
                            overlap_thresh=self.nms_thresh,
                            score_thresh=0.0001,
                            method=cfg.TEST.SOFT_NMS.METHOD
                        )
                    else:
                        boxlist_for_class = boxlist_nms(
                            boxlist_for_class, self.nms_thresh,
                            score_field="scores"
                        )
                    # Refine the post-NMS boxes using bounding-box voting
                    if cfg.TEST.BBOX_VOTE.ENABLED and boxes_j.shape[0] > 0:
                        boxlist_for_class = boxlist_box_voting(
                            boxlist_for_class,
                            boxlist_for_class_old,
                            cfg.TEST.BBOX_VOTE.VOTE_TH,
                            scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                        )
                    num_labels = len(boxlist_for_class)
                    boxlist_for_class.add_field(
                        "labels", torch.full((num_labels,), j, dtype=torch.int64, device=scores.device)
                    )
                    result.append(boxlist_for_class)

                result = cat_boxlist(result)

            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def hier_post_processor():
    pre_nms_thresh = cfg.HRCNN.INFERENCE_TH
    pre_nms_top_n = cfg.HRCNN.PRE_NMS_TOP_N
    nms_thresh = cfg.HRCNN.NMS_TH
    fpn_post_nms_top_n = cfg.HRCNN.DETECTIONS_PER_IMG
    num_classes = cfg.HRCNN.NUM_CLASSES
    resolution = cfg.HRCNN.ROI_XFORM_RESOLUTION
    eval_hier = cfg.HRCNN.EVAL_HIER
    hier_th = cfg.HRCNN.HIER_TH

    hier_post_processor = HierPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=num_classes,
        resolution=resolution,
        eval_hier=eval_hier,
        hier_th=hier_th
    )
    return hier_post_processor
