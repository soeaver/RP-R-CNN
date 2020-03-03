import os
import json
import pickle
import tempfile
import shutil
import numpy as np
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.boxlist_ops import boxlist_iou
from utils.data.evaluation.parsing_eval import parsing_png, evaluate_parsing
from utils.misc import logging_rank
from rcnn.datasets import build_dataset, dataset_catalog
from rcnn.modeling.mask_rcnn.inference import mask_results
from rcnn.modeling.parsing_rcnn.inference import parsing_results
from rcnn.core.config import cfg


def post_processing(results, image_ids, dataset):
    cpu_device = torch.device("cpu")
    results = [o.to(cpu_device) for o in results]
    num_im = len(image_ids)

    box_results, ims_dets, ims_labels = prepare_box_results(results, image_ids, dataset)

    if cfg.MODEL.MASK_ON:
        seg_results, ims_segs = prepare_segmentation_results(results, image_ids, dataset)
    else:
        seg_results = []
        ims_segs = [None for _ in range(num_im)]

    if cfg.MODEL.PARSING_ON:
        par_results, par_score = prepare_parsing_results(results, image_ids, dataset)
        ims_pars = par_results
    else:
        par_results = []
        par_score = []
        ims_pars = [None for _ in range(num_im)]

    eval_results = [box_results, seg_results, par_results, par_score]
    ims_results = [ims_dets, ims_labels, ims_segs, ims_pars]
    return eval_results, ims_results


def evaluation(dataset, all_boxes, all_segms, all_parss, all_pscores, clean_up=True):
    output_folder = os.path.join(cfg.CKPT, 'test')
    expected_results = ()
    expected_results_sigma_tol = 4

    coco_results = {}
    iou_types = ("bbox",)
    coco_results["bbox"] = all_boxes
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
        coco_results["segm"] = all_segms
    if cfg.MODEL.PARSING_ON:
        iou_types = iou_types + ("parsing",)
        coco_results['parsing'] = [all_parss, all_pscores]

    results = COCOResults(*iou_types)
    logging_rank("Evaluating predictions", local_rank=0)
    for iou_type in iou_types:
        if iou_type == "parsing":
            eval_ap = cfg.PRCNN.EVAL_AP
            num_parsing = cfg.PRCNN.NUM_PARSING
            assert len(cfg.TEST.DATASETS) == 1, 'Parsing only support one dataset now'
            im_dir = dataset_catalog.get_im_dir(cfg.TEST.DATASETS[0])
            ann_fn = dataset_catalog.get_ann_fn(cfg.TEST.DATASETS[0])
            res = evaluate_parsing(
                coco_results[iou_type], eval_ap, cfg.PRCNN.SCORE_THRESH, num_parsing, im_dir, ann_fn, output_folder
            )
            results.update_parsing(res)
        else:
            with tempfile.NamedTemporaryFile() as f:
                file_path = f.name
                if output_folder:
                    file_path = os.path.join(output_folder, iou_type + ".json")
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type
                )
                results.update(res)
    logging_rank(results, local_rank=0)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
        if clean_up:
            shutil.rmtree(output_folder)
    return results, coco_results


def prepare_box_results(results, image_ids, dataset):
    box_results = []
    ims_dets = []
    ims_labels = []
    for i, result in enumerate(results):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(result) == 0:
            ims_dets.append(None)
            ims_labels.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        result = result.resize((image_width, image_height))
        boxes = result.bbox
        scores = result.get_field("scores")
        labels = result.get_field("labels")
        ims_dets.append(np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False))
        result = result.convert("xywh")
        boxes = result.bbox.tolist()
        scores = scores.tolist()
        labels = labels.tolist()
        ims_labels.append(labels)
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        box_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    return box_results, ims_dets, ims_labels


def prepare_segmentation_results(results, image_ids, dataset):
    seg_results = []
    ims_segs = []
    for i, result in enumerate(results):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(result) == 0:
            ims_segs.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        input_w, input_h = result.size
        result = result.resize((image_width, image_height))
        masks = result.get_field("mask")
        if cfg.MODEL.EMBED_MASK_ON:
            import pycocotools.mask as mask_util
            # resize masks
            stride_mask = result.get_field('stride')
            h = (masks.shape[1] * stride_mask.float() * image_height / input_h).ceil().long()
            w = (masks.shape[2] * stride_mask.float() * image_width / input_w).ceil().long()
            mask_th = result.get_field('mask_th').cuda()
            masks = masks.cuda()
            masks = torch.nn.functional.interpolate(input=masks.unsqueeze(1).float(), size=(h, w), mode="bilinear",
                                                    align_corners=False).gt(mask_th)
            masks = masks[:, :, :image_height, :image_width]
            masks = masks.cpu()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
        else:
            rles = mask_results(masks, result)
        # boxes = prediction.bbox.tolist()
        scores = result.get_field("mask_scores").tolist()
        labels = result.get_field("labels").tolist()
        ims_segs.append(rles)
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        seg_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return seg_results, ims_segs


def prepare_parsing_results(results, image_ids, dataset):
    all_parsing = []
    all_scores = []
    output_folder = os.path.join(cfg.CKPT, 'test')
    for i, result in enumerate(results):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(result) == 0:
            all_parsing.append([])
            all_scores.append(0)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        result = result.resize((image_width, image_height))
        semseg = result.get_field("semseg") if cfg.MODEL.SEMSEG_ON else None
        parsing = result.get_field("parsing")
        parsing = parsing_results(parsing, result, semseg=semseg)
        scores = result.get_field("parsing_scores")
        parsing_png(
            parsing, scores, cfg.PRCNN.SEMSEG_SCORE_THRESH, img_info, output_folder, semseg=semseg
        )
        all_parsing.append(parsing)
        all_scores.append(scores)
    return all_parsing, all_scores


def evaluate_predictions_on_coco(coco_gt, coco_results, json_result_file, iou_type="bbox"):
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    if iou_type == "bbox":
        _print_detection_eval_metrics(coco_gt, coco_eval)
    coco_eval.summarize()
    return coco_eval


def _print_detection_eval_metrics(coco_gt, coco_eval):
    # mAP = 0.0
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    category_ids = coco_gt.getCatIds()
    categories = [c['name'] for c in coco_gt.loadCats(category_ids)]
    classes = tuple(['__background__'] + categories)
    for cls_ind, cls in enumerate(classes):
        if cls == '__background__':
            continue
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{} = {:.1f}'.format(cls, 100 * ap))


def get_box_result():
    box_results = []
    with open(dataset_catalog.get_ann_fn(cfg.TEST.DATASETS[0])) as f:
        anns = json.load(f)['annotations']
        for ann in anns:
            box_results.append({
                "image_id": ann['image_id'],
                "category_id": ann['category_id'],
                "bbox": ann['bbox'],
                "score": 1.0,
            })
            hier = ann['hier']
            N = len(hier) // 5
            for i in range(N):
                if hier[i * 5 + 4] > 0:
                    x1, y1, x2, y2 = hier[i * 5: i * 5 + 4]
                    bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
                    box_results.append({
                        "image_id": ann['image_id'],
                        "category_id": i + 2,
                        "bbox": bbox,
                        "score": 1.0,
                    })
    return box_results


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "parsing": ["mIoU", "APp50", "APpvol", "PCP"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "parsing")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def update_parsing(self, eval):
        if eval is None:
            return

        res = self.results['parsing']
        for k, v in eval.items():
            res[k] = v
            
    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
