import torch
from torch.nn import functional as F

from models.ops import IOULoss
from models.ops import SigmoidFocalLoss
from utils.data.structures.bounding_box import BoxList
from utils.data.structures.hier import Hier
from utils.data.structures.boxlist_ops import boxlist_iou
from utils.data.structures.boxlist_ops import cat_boxlist
from rcnn.utils.matcher import Matcher
from rcnn.utils.misc import cat, keep_only_positive_boxes, get_num_gpus, reduce_sum
from rcnn.core.config import cfg

INF = 100000000


def center_within_box(points, boxes):
    """Validate which hier are contained inside a given box.
    points: NxKx5
    boxes: Nx4
    output: NxK
    """
    center_x = (points[..., 0] + points[..., 2]) / 2
    w = points[..., 2] - points[..., 0]
    center_y = (points[..., 1] + points[..., 3]) / 2
    h = points[..., 3] - points[..., 1]
    x_within = ((center_x - 0.25 * w) >= boxes[:, 0, None]) & (
        (center_x + 0.25 * w) <= boxes[:, 2, None]
    )
    y_within = ((center_y - 0.25 * h) >= boxes[:, 1, None]) & (
        (center_y + 0.25 * h) <= boxes[:, 3, None]
    )
    return x_within & y_within


class HierRCNNLossComputation(object):
    def __init__(self, proposal_matcher, resolution):
        """
        Arguments:
            proposal_matcher (Matcher)
            resolution (int)
        """
        self.proposal_matcher = proposal_matcher
        self.resolution = resolution
        self.roi_size_per_img = cfg.HRCNN.ROI_SIZE_PER_IMG
        self.loss_weight = cfg.HRCNN.LOSS_WEIGHT
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.HRCNN.LOSS_GAMMA,
            cfg.HRCNN.LOSS_ALPHA
        )
        self.loc_loss_type = cfg.HRCNN.LOC_LOSS_TYPE
        self.box_reg_loss_func = IOULoss(self.loc_loss_type)
        self.centerness_loss_func = torch.nn.BCEWithLogitsLoss(reduction="sum")

        self.center_sample = cfg.HRCNN.CENTER_SAMPLE
        self.radius = cfg.HRCNN.POS_RADIUS
        self.norm_reg_targets = cfg.HRCNN.NORM_REG_TARGETS
        self.limit_type = cfg.HRCNN.LIMIT_TYPE

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Hier RCNN needs "labels" and "hier "fields for creating the targets
        target = target.copy_with_fields(["labels", "hier"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        positive_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            hier_per_image = matched_targets.get_field("hier")
            within_box = center_within_box(hier_per_image.hier, matched_targets.bbox)
            vis_hier = hier_per_image.hier[..., 4] > 0
            is_visible = (within_box & vis_hier).sum(1) > 0

            if self.limit_type != 'none':
                if self.limit_type == 'hand_and_foot':
                    has_part = vis_hier[:, 2:].sum(1) == (within_box & vis_hier)[:, 2:].sum(1)
                elif self.limit_type == 'all':
                    has_part = vis_hier[:, 0:].sum(1) == (within_box & vis_hier)[:, 0:].sum(1)
                else:
                    raise Exception("Limit type not support: ", self.limit_type)
                is_visible = has_part & is_visible

            labels_per_image[~is_visible] = -1

            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            if self.roi_size_per_img > 0:
                if self.roi_size_per_img < positive_inds.shape[0]:
                    _inds = torch.randperm(positive_inds.shape[0])[:self.roi_size_per_img]
                    positive_inds = positive_inds[_inds]

            proposals_per_image = proposals_per_image[positive_inds]
            hier_per_image = hier_per_image[positive_inds]

            hier_gt_per_image = targets_per_image.get_field("hier")
            hier_gt_parts = hier_gt_per_image.hier[:, 2:]
            vis_hier_parts = hier_gt_parts[..., 4].sum(1) > 1
            parts_nonzeros = vis_hier_parts.nonzero()[:, 0]

            if parts_nonzeros.shape[0] > 0 and self.roi_size_per_img > 0:
                gt_parts_batch_size = self.roi_size_per_img - positive_inds.shape[0]
                if gt_parts_batch_size < parts_nonzeros.shape[0]:
                    _inds = torch.randperm(parts_nonzeros.shape[0])[:gt_parts_batch_size]
                    parts_nonzeros = parts_nonzeros[_inds]

            if parts_nonzeros.shape[0] > 0:
                hier_gt_parts = hier_gt_parts[parts_nonzeros]
                parts_boxes = []
                for i in range(parts_nonzeros.shape[0]):
                    hier_gt_part = hier_gt_parts[i, (hier_gt_parts[i, :, 4] > 0).nonzero()[:, 0], :4]
                    x1 = hier_gt_part[:, 0].min()
                    y1 = hier_gt_part[:, 1].min()
                    x2 = hier_gt_part[:, 2].max()
                    y2 = hier_gt_part[:, 3].max()
                    parts_boxes.append(torch.stack([x1, y1, x2, y2], dim=0))
                parts_boxes = torch.stack(parts_boxes, dim=0)
                parts_hier = hier_gt_per_image[parts_nonzeros]

                boxes = torch.cat([proposals_per_image.bbox, parts_boxes], dim=0)
                hier = torch.cat([hier_per_image.hier, parts_hier.hier], dim=0)

                proposals_per_image = BoxList(boxes, proposals_per_image.size, mode=proposals_per_image.mode)
                hier_per_image = Hier(hier, proposals_per_image.size)

            if len(proposals_per_image) == 0:
                hier_gt_per_image = targets_per_image.get_field("hier")
                vis_hier_parts = hier_gt_per_image.hier[..., 4].sum(1) > 0
                parts_nonzeros = vis_hier_parts.nonzero()[:, 0][:1]
                proposals_per_image = BoxList(
                    targets_per_image[parts_nonzeros].bbox,
                    targets_per_image.size,
                    mode=targets_per_image.mode
                )
                hier_per_image = hier_gt_per_image[parts_nonzeros]

            proposals_per_image.add_field("hier_target", hier_per_image)
            positive_proposals.append(proposals_per_image)
        return positive_proposals

    def subsample(self, proposals, targets):
        positive_proposals = keep_only_positive_boxes(proposals)
        positive_proposals = self.prepare_targets(positive_proposals, targets)
        self.positive_proposals = positive_proposals

        all_num_positive_proposals = 0
        for positive_proposals_per_image in positive_proposals:
            all_num_positive_proposals += len(positive_proposals_per_image)
        if all_num_positive_proposals == 0:
            positive_proposals = [proposals[0][:1]]
        return positive_proposals

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_hier_targets(self, hier, proposals, points):
        object_sizes_of_interest = [[-1, INF]]

        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, expanded_object_sizes_of_interest, hier, proposals
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / 0.5
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, object_sizes_of_interest, hier, proposals):
        proposals = proposals.convert("xyxy")
        device = hier.hier.device
        num_classes = hier.hier.shape[1]
        boxes = proposals.bbox
        visibles = hier.hier[:, :, 4]
        h, w = self.resolution
        _x1 = (hier.hier[:, :, 0] - boxes[:, 0, None]) * w / (boxes[:, 2, None] - boxes[:, 0, None])
        _y1 = (hier.hier[:, :, 1] - boxes[:, 1, None]) * h / (boxes[:, 3, None] - boxes[:, 1, None])
        _x2 = (hier.hier[:, :, 2] - boxes[:, 0, None]) * w / (boxes[:, 2, None] - boxes[:, 0, None])
        _y2 = (hier.hier[:, :, 3] - boxes[:, 1, None]) * h / (boxes[:, 3, None] - boxes[:, 1, None])
        new_hier = torch.stack([_x1, _y1, _x2, _y2], dim=-1)
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(new_hier)):
            visible = visibles[im_i] > 0
            bboxes = new_hier[im_i][visible]
            labels_per_im = torch.range(1, num_classes, dtype=torch.long, device=device)[visible]
            area = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    [0.5],
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, box_cls, box_regression, centerness, locations):
        num_classes = box_cls[0].size(1)
        level = len(locations)
        labels = [[] for _ in range(level)]
        reg_targets = [[] for _ in range(level)]
        for proposals_per_image in self.positive_proposals:
            hier = proposals_per_image.get_field("hier_target")
            labels_per_image, reg_targets_per_image = self.prepare_hier_targets(
                hier, proposals_per_image, locations
            )
            for i in range(level):
                labels[i].append(labels_per_image[i])
                reg_targets[i].append(reg_targets_per_image[i])

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].permute(0, 2, 3, 1).reshape(-1))
            labels_flatten.append(torch.cat(labels[l], dim=0))
            reg_targets_flatten.append(torch.cat(reg_targets[l], dim=0))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
            cls_loss *= self.loss_weight
            reg_loss *= self.loss_weight
            centerness_loss *= self.loss_weight
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def hier_loss_evaluator():
    matcher = Matcher(
        cfg.HRCNN.FG_IOU_THRESHOLD,
        cfg.HRCNN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    resolution = cfg.HRCNN.ROI_XFORM_RESOLUTION
    loss_evaluator = HierRCNNLossComputation(matcher, resolution)
    return loss_evaluator
