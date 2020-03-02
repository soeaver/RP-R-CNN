import cv2
import numpy as np
import os
import pycocotools.mask as mask_util
from collections import defaultdict
import matplotlib.pyplot as plt

from utils.timer import Timer
import utils.colormap as colormap_utils
from rcnn.core.config import cfg

_GRAY = [218, 227, 218]
_GREEN = [18, 127, 15]
_WHITE = [255, 255, 255]


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_bbox(img, bbox, bbox_color):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), bbox_color, thickness=cfg.VIS.SHOW_BOX.BORDER_THICK)

    return img


def vis_class(img, pos, class_str, bg_color):
    """Visualizes the class."""
    font_color = cfg.VIS.SHOW_CLASS.COLOR
    font_scale = cfg.VIS.SHOW_CLASS.FONT_SCALE

    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, bg_color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, font_color, lineType=cv2.LINE_AA)

    return img


def vis_mask(img, mask, bbox_color, show_parss=False):
    """Visualizes a single binary mask."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    border_color = cfg.VIS.SHOW_SEGMS.BORDER_COLOR
    border_thick = cfg.VIS.SHOW_SEGMS.BORDER_THICK

    mask_color = bbox_color if cfg.VIS.SHOW_SEGMS.MASK_COLOR_FOLLOW_BOX else _WHITE
    mask_color = np.asarray(mask_color)
    mask_alpha = cfg.VIS.SHOW_SEGMS.MASK_ALPHA

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if cfg.VIS.SHOW_SEGMS.SHOW_BORDER:
        cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)

    if cfg.VIS.SHOW_SEGMS.SHOW_MASK and not show_parss:
        img[idx[0], idx[1], :] *= 1.0 - mask_alpha
        img[idx[0], idx[1], :] += mask_alpha * mask_color

    return img.astype(np.uint8)


def vis_hier(img, hier, bbox_color):
    border_thick = cfg.VIS.SHOW_HIER.BORDER_THICK
    N = len(hier) // 5
    for i in range(N):
        if hier[i * 5 + 4] > 0:
            cv2.rectangle(
                img,
                (int(hier[i * 5]), int(hier[i * 5 + 1])),
                (int(hier[i * 5 + 2]), int(hier[i * 5 + 3])),
                bbox_color,
                thickness=border_thick
            )

    return img


def get_instance_parsing_colormap(rgb=False):
    instance_colormap = eval('colormap_utils.{}'.format(cfg.VIS.SHOW_BOX.COLORMAP))
    if rgb:
        instance_colormap = colormap_utils.dict_bgr2rgb(instance_colormap)

    return instance_colormap, None


def vis_one_image_opencv(im, config, boxes, classes, segms=None, hier=None, dataset=None):
    """Constructs a numpy array with the detections visualized."""
    timers = defaultdict(Timer)
    timers['bbox_prproc'].tic()

    global cfg
    cfg = config

    if cfg.VIS.SHOW_HIER.ENABLED and hier is not None:
        classes = np.array(classes)
        boxes = boxes[classes == 1]
        classes = classes[classes == 1]

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < cfg.VIS.VIS_TH:
        return im

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    # get color map
    ins_colormap, parss_colormap = get_instance_parsing_colormap()

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    timers['bbox_prproc'].toc()

    instance_id = 1
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < cfg.VIS.VIS_TH:
            continue

        # get instance color (box, class_bg)
        if cfg.VIS.SHOW_BOX.COLOR_SCHEME == 'category':
            ins_color = ins_colormap[classes[i]]
        elif cfg.VIS.SHOW_BOX.COLOR_SCHEME == 'instance':
            instance_id = instance_id % len(ins_colormap.keys())
            ins_color = ins_colormap[instance_id]
        else:
            ins_color = _GREEN
        instance_id += 1

        # show box (off by default)
        if cfg.VIS.SHOW_BOX.ENABLED:
            timers['show_box'].tic()
            im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), ins_color)
            timers['show_box'].toc()

        # show class (off by default)
        if cfg.VIS.SHOW_CLASS.ENABLED:
            timers['show_class'].tic()
            class_str = get_class_string(classes[i], score, dataset)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str, ins_color)
            timers['show_class'].toc()

        show_segms = True if cfg.VIS.SHOW_SEGMS.ENABLED and segms is not None and len(segms) > i else False
        show_hier = True if cfg.VIS.SHOW_HIER.ENABLED and hier is not None and len(hier) > i else False
        # show mask
        if show_segms:
            timers['show_segms'].tic()
            color_list = colormap_utils.colormap()
            im = vis_mask(im, masks[..., i], ins_color, show_parss=show_parss)
            timers['show_segms'].toc()

        # show hier
        if show_hier:
            timers['show_hier'].tic()
            im = vis_hier(im, hier[i], ins_color)
            timers['show_hier'].toc()

    # for k, v in timers.items():
    #     print(' | {}: {:.3f}s'.format(k, v.total_time))

    return im
