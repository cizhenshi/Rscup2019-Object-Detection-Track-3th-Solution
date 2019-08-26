import pickle
import json
import mmcv
from tqdm import tqdm
from mmdet.datasets import build_dataloader, build_dataset
from pycocotools.coco import COCO
import pycocotools.coco as cocoapi
import pycocotools.mask as MASK
import os
import poly.polyiou as polyiou
import cv2
from icecream import ic
import matplotlib.pyplot as plt
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, get_dist_info
import numpy as np
from multiprocessing import Pool
from mmdet.ops.nms import nms_wrapper


def get_ann(bboxes, segs, cls, name, locx, locy, scale_factor):
    Rect = []
    Bbox = []
    Vis = []
    bbox_cls = bboxes[cls]
    seg_cls = segs[cls]
    if (len(bbox_cls) > 0):
        for bbox, rle in zip(bbox_cls, seg_cls):
            xmin, ymin, xmax, ymax, score = bbox
            xmin += locx
            ymin += locy
            xmax += locx
            ymax += locy
            bounding_box = np.array(
                [xmin * scale_factor, ymin * scale_factor, xmax * scale_factor, ymax * scale_factor, score])
            mask = MASK.decode(rle)
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(contour) for contour in contours]
            if (len(areas) > 0):
                index = np.argmax(areas)
                contour = contours[index]
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box += np.array([locx, locy])

                center = np.array(rect[0])
                size = np.array(rect[1])
                angel = rect[2]
                size *= scale_factor
                center += np.array([locx, locy])
                center *= scale_factor
                rect = (tuple(center), tuple(size), angel)
            else:
                continue
            Rect.append(rect)
            Bbox.append(bounding_box)
            Vis.append(
                np.array([box[0] * scale_factor, box[1] * scale_factor, box[2] * scale_factor, box[3] * scale_factor]))
    return cls, name, Rect, Bbox, Vis


def get_all_ann(filename, result, img_prefix, size, CLASS_NUM=18):
    items = filename.split("_")
    name = items[0]
    img = cv2.imread(img_prefix + filename)
    h, w = size
    scale_factor = 1 / float(items[1])
    locx = int(float(items[2]))
    locy = int(float(items[3]))
    bboxes = result[0]
    segs = result[1]
    Rect = [[] for cls in range(CLASS_NUM)]
    Bbox = [[] for cls in range(CLASS_NUM)]
    Vis = [[] for cls in range(CLASS_NUM)]
    for cls in range(CLASS_NUM):
        if (cls == 7):
            if not (locx == 0 and locy == 0):
                continue
        bbox_cls = bboxes[cls]
        seg_cls = segs[cls]
        if (len(bbox_cls) > 0):
            for bbox, rle in zip(bbox_cls, seg_cls):
                xmin, ymin, xmax, ymax, score = bbox
                xmin += locx
                ymin += locy
                xmax += locx
                ymax += locy
                bounding_box = np.array(
                    [xmin * scale_factor, ymin * scale_factor, xmax * scale_factor, ymax * scale_factor, score])
                mask = MASK.decode(rle)
                contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                areas = [cv2.contourArea(contour) for contour in contours]
                if (len(areas) > 0):
                    index = np.argmax(areas)
                    contour = contours[index]
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box += np.array([locx, locy])

                    center = np.array(rect[0])
                    size = np.array(rect[1])
                    angel = rect[2]
                    size *= scale_factor
                    center += np.array([locx, locy])
                    center *= scale_factor
                    rect = (tuple(center), tuple(size), angel)
                else:
                    continue
                Rect[cls].append(rect)
                Bbox[cls].append(bounding_box)
                Vis[cls].append(np.array(
                    [box[0] * scale_factor, box[1] * scale_factor, box[2] * scale_factor, box[3] * scale_factor]))
    return name, Rect, Bbox, Vis


def merge_result(config_file, result_file, anno_file, img_prefix, out_file=None, CLASS_NUM=18):
    results = mmcv.load(result_file)
    coco = COCO(anno_file)
    img_ids = coco.getImgIds()
    img_infos = []
    for i in img_ids:
        info = coco.loadImgs([i])[0]
        info['filename'] = info['file_name']
        info['height'] = int(info['height'])
        info['width'] = int(info['width'])
        img_infos.append(info)

    ann = {}
    rets = []
    pbar = mmcv.ProgressBar(len(results))

    def update(*a):
        pbar.update()

    p = Pool(8)
    for i in range(len(results)):
        filename = img_infos[i]['filename']
        h = img_infos[i]['height']
        w = img_infos[i]['width']
        rets.append(p.apply_async(get_all_ann, args=(filename, results[i], img_prefix, (h, w), 18), callback=update))

    for ret in rets:
        name, Rect, Bbox, Vis = ret.get()
        if name not in ann:
            ann[name] = {"bbox": [[] for i in range(CLASS_NUM)], "vis": [[] for i in range(CLASS_NUM)],
                         "rect": [[] for i in range(CLASS_NUM)]}
        for cls in range(CLASS_NUM):
            ann[name]['bbox'][cls] += Bbox[cls]
            ann[name]['rect'][cls] += Rect[cls]
            ann[name]['vis'][cls] += Vis[cls]
    if out_file is not None:
        mmcv.dump(ann, out_file)
    p.close()
    p.join()
    return ann


def poly_nms(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep


def nms(ann, mode, thresh, out_file=None, CLASS_NUM=18):
    for name in ann.keys():
        info = ann[name]
        for cls in range(CLASS_NUM):
            bbox = np.array(info['bbox'][cls], np.float32)
            vis = np.array(info['vis'][cls])
            if (len(bbox) <= 0):
                continue
            if mode == "rec":
                _, inds = nms_wrapper.nms(bbox, thresh)
            elif mode == "poly":
                dets = vis.reshape(-1, 8)
                dets = np.array(dets, np.int32)
                scores = bbox[:, 4]
                dets = np.c_[dets, scores]
                # print(bbox.shape)
                inds = poly_nms(dets, thresh)
            # print(len(inds))
            ann[name]['bbox'][cls] = bbox[inds]
            ann[name]['vis'][cls] = vis[inds]
    if out_file is not None:
        mmcv.dump(ann, out_file)
    return ann


def generate_submit(ann, out_path, CLASSES, CLASS_NUM=18):
    names = list(ann.keys())
    res = {CLASSES[cls]: [] for cls in range(CLASS_NUM)}
    #pbar = mmcv.ProgressBar(len(names))
    for name in tqdm(names):
        result = ann[name]
        bboxes = result["bbox"]
        segs = result['vis']
        for cls in range(CLASS_NUM):
            curr_class = CLASSES[cls]
            bbox_cls = bboxes[cls]
            seg_cls = segs[cls]
            if (len(bbox_cls) > 0):
                for bbox, rle in zip(bbox_cls, seg_cls):
                    rle = np.array(rle)
                    xmin, ymin, xmax, ymax, score = bbox
                    location = list(rle.flatten())
                    location = [str(int(x)) for x in location]
                    if (score < 0.05):
                        continue
                    out = name + " " + str(score) + " " + " ".join(location)
                    res[curr_class].append(out)
    for key in res.keys():
        print(key)
        fp = open("./result/{}/".format(out_path) + key + ".txt", 'w')
        print(len(res[key]))
        for line in res[key]:
            fp.write(line + "\n")
        fp.close()


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
      Compute VOC AP given precision and recall.
      If use_07_metric is true, uses the
      VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        # first appicend sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_warpper(classname, detpath, ovthresh, coco):
    CLASS = ['tennis-court', 'container-crane', 'storage-tank', 'baseball-diamond', 'plane', 'ground-track-field',
             'helicopter', 'airport', 'harbor', 'ship', 'large-vehicle', 'swimming-pool', 'soccer-ball-field',
             'roundabout', 'basketball-court', 'bridge', 'small-vehicle', 'helipad']
    # CLASS={'tennis-court', 'container-crane', 'storage-tank', 'baseball-diamond', 'plane', 'ground-track-field', 'helicopter', 'airport', 'harbor', 'ship', 'large-vehicle', 'swimming-pool', 'soccer-ball-field', 'roundabout', 'basketball-court', 'bridge', 'small-vehicle', 'helipad'}
    class_to_ind = dict(zip(CLASS, range(len(CLASS))))
    imgIds = coco.getImgIds()
    recs = {}
    use_07_metric = False
    for imgid in imgIds:
        img = coco.loadImgs(imgid)[0]
        file_name = img['file_name']
        file_name = file_name.split(".")[0]
        annIds = coco.getAnnIds(imgIds=[imgid], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objects = []
        for ann in anns:
            obj = {}
            obj['name'] = CLASS[ann['category_id']]
            obj['bbox'] = ann['segmentation']
            objects.append(obj)
        recs[file_name] = objects
    class_recs = {}
    npos = 0
    for filename in list(recs.keys()):
        R = [obj for obj in recs[filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        bbox = np.reshape(np.squeeze(bbox), (-1, 4, 2))
        det = [False] * len(R)
        npos = npos + len(bbox)
        class_recs[filename] = {'bbox': bbox,
                                'det': det}
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    BB = np.reshape(BB, (-1, 4, 2))
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float).flatten()
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float).reshape(-1, 8)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
                BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
                BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
                BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
                bb_xmin = np.min(bb[0::2])
                bb_ymin = np.min(bb[1::2])
                bb_xmax = np.max(bb[0::2])
                bb_ymax = np.max(bb[1::2])

                ixmin = np.maximum(BBGT_xmin, bb_xmin)
                iymin = np.maximum(BBGT_ymin, bb_ymin)
                ixmax = np.minimum(BBGT_xmax, bb_xmax)
                iymax = np.minimum(BBGT_ymax, bb_ymax)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                       (BBGT_xmax - BBGT_xmin + 1.) *
                       (BBGT_ymax - BBGT_ymin + 1.) - inters)

                overlaps = inters / uni

                BBGT_keep_mask = overlaps > 0
                BBGT_keep = BBGT[BBGT_keep_mask, :]
                BBGT_keep_index = np.where(overlaps > 0)[0]

                def calcoverlaps(BBGT_keep, bb):
                    overlaps = []
                    for index, GT in enumerate(BBGT_keep):
                        overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                        overlaps.append(overlap)
                    return overlaps

                if len(BBGT_keep) > 0:
                    overlaps = calcoverlaps(BBGT_keep, bb)

                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                    # pdb.set_trace()
                    jmax = BBGT_keep_index[jmax]

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    print("ap of {} is {}".format(classname, ap))
    return ap


def evaluate(iou_thresh):
    p = Pool(8)
    detpath = "./result/detection/{}.txt"
    CLASS = ['tennis-court', 'container-crane', 'storage-tank', 'baseball-diamond', 'plane', 'ground-track-field',
             'helicopter', 'airport', 'harbor', 'ship', 'large-vehicle', 'swimming-pool', 'soccer-ball-field',
             'roundabout', 'basketball-court', 'bridge', 'small-vehicle', 'helipad']
    iou_thresh = 0.5
    aps = []
    coco = COCO("./gt_val.json")
    for classname in CLASS:
        aps.append(p.apply_async(eval_warpper, args=(classname, detpath, iou_thresh, coco)))
    ret = []
    for ap in aps:
        ret.append(ap.get())
    p.close()
    p.join()
    print("map is {}".format(np.mean(np.array(ret))))

if __name__ == "__main__":
    CLASSES = ['tennis-court', 'container-crane', 'storage-tank', 'baseball-diamond', 'plane', 'ground-track-field',
               'helicopter', 'airport', 'harbor', 'ship', 'large-vehicle', 'swimming-pool', 'soccer-ball-field',
               'roundabout', 'basketball-court', 'bridge', 'small-vehicle', 'helipad']
    config_file = None
    result_file = "./batch_3s.pkl"
    anno_file = "./data/rscup/annotation/annos_rscup_val.json"
    out_file = "./result/eval_temp.pkl"
    img_prefix = "./data/rscup/val/"
    ann = merge_result(config_file, result_file, anno_file, img_prefix, out_file)
    ann = nms(ann, "poly", 0.5)
    mmcv.dump(ann, "./result/post_nms.pkl")
    generate_submit(ann, "detection", CLASSES)
    #evaluate(0.5)