import os
import cv2

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.segmentation_mask import SegmentationMask
from utils.data.structures.semantic_segmentation import SemanticSegmentation, get_semseg
from utils.data.structures.hier import Hier

min_hier_per_image = 1


def _count_visible_hier(anno):
    return sum(sum(1 for v in ann["hier"][4::5] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno, ann_types, filter_crowd=True):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    if filter_crowd:
        # if image only has crowd annotation, it should be filtered
        if 'iscrowd' in anno[0]:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    if 'hier' in ann_types:
        hier_vis = _count_visible_hier(anno) >= min_hier_per_image
    else:
        hier_vis = True

    if hier_vis:
        return True

    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations, ann_types, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno, ann_types):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        category_ids = self.coco.getCatIds()
        categories = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.classes = ['__background__'] + categories
        self.ann_types = ann_types
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if len(anno) > 0:
            if 'iscrowd' in anno[0]:
                anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if 'segm' in self.ann_types:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if 'semseg' in self.ann_types:
            if 'parsing' in self.ann_types:
                semsegs_anno = get_semseg(self.root, self.coco.loadImgs(self.ids[idx])[0]['file_name'])
                semsegs = SemanticSegmentation(semsegs_anno, classes, img.size, mode='pic')
            else:
                semsegs_anno = [obj["segmentation"] for obj in anno]
                semsegs = SemanticSegmentation(semsegs_anno, classes, img.size, mode='poly')
            target.add_field("semsegs", semsegs)

        if 'hier' in self.ann_types:
            if anno and "hier" in anno[0]:
                hier = [obj["hier"] for obj in anno]
                hier = Hier(hier, img.size)
                target.add_field("hier", hier)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def pull_image(self, index):
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img
        """
        img_id = self.id_to_img_map[index]

        path = self.coco.loadImgs(img_id)[0]['file_name']

        return cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
