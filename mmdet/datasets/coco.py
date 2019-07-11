import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from icecream import ic
import copy

class CocoDataset(CustomDataset):
    #
    CLASSES = ('tennis-court', 'container-crane', 'storage-tank', 'baseball-diamond', 'plane','ground-track-field',
               'helicopter', 'airport', 'harbor', 'ship', 'large-vehicle', 'swimming-pool', 'soccer-ball-field',
               'roundabout', 'basketball-court', 'bridge', 'small-vehicle', 'helipad')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(copy.deepcopy(self.img_infos[idx]), ann_info, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def random_crop(self, img_info, ann_info):
        H = img_info["height"]
        W = img_info["width"]
        h_array = np.zeros(H, dtype=np.int32)
        w_array = np.zeros(W, dtype=np.int32)
        min_crop_side_ratios = [0.15, 0.15]
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            xmin, ymin, w, h = ann['bbox']
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            w_array[xmin:xmax] = 1
            h_array[ymin:ymax] = 1
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            return img_info, 0, W-1, 0, H-1
        for i in range(50):
            xx = np.random.choice(w_axis, size=2)
            yy = np.random.choice(h_axis, size=2)
            xmin = np.min(xx)
            xmax = np.max(xx)
            ymin = np.min(yy)
            ymax = np.max(yy)
            if xmax - xmin < min_crop_side_ratios[0] * W or ymax - ymin < min_crop_side_ratios[1] * H:
                # area too small
                continue
            img_info['height'] = ymax - ymin
            img_info['width'] = xmax - xmin
            return img_info, xmin, xmax, ymin, ymax
        return img_info, 0, W - 1, 0, H - 1



    def _parse_ann_info(self, img_info, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            img_info: img information
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        # img_info, xmin, xmax, ymin, ymax = self.random_crop(img_info, ann_info)
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h
            # if not (x1 > xmin and x2 < xmax and y1 > ymin and y2 < ymax):
            #     continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann
