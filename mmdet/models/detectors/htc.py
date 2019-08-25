import torch
import torch.nn.functional as F

from .cascade_rcnn import CascadeRCNN
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (bbox2roi, bbox2result, build_assigner, build_sampler,
                        merge_aug_masks)
from icecream import ic
import time
from torch.multiprocessing import Pool
from mmdet.core import (delta2bbox, multiclass_nms, nms, bbox_target, force_fp32,
                        auto_fp16)


def get_det_bboxes(j, rois,
                   scores,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
    target_means = [0., 0., 0., 0.],
    target_stds = [0.033, 0.033, 0.067, 0.067]

    if bbox_pred is not None:
        bboxes = delta2bbox(rois[:, 1:], bbox_pred, target_means,
                            target_stds, img_shape)
    else:
        bboxes = rois[:, 1:].clone()
        if img_shape is not None:
            bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
            bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

    if rescale:
        bboxes /= scale_factor

    if cfg is None:
        return bboxes, scores
    else:
        det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)

        return [j, det_bboxes, det_labels]


@DETECTORS.register_module
class HybridTaskCascade(CascadeRCNN):

    def __init__(self,
                 num_stages,
                 backbone,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 **kwargs):
        super(HybridTaskCascade, self).__init__(num_stages, backbone, **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head  # shared head not supported
        if semantic_head is not None:
            self.semantic_roi_extractor = builder.build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = builder.build_head(semantic_head)

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.p = Pool(4)

    @property
    def with_semantic(self):
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat

        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                            gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        return loss_bbox, rois, bbox_targets, bbox_pred

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        ic(len(sampling_results))

        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)
        ic(mask_feats.shape)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](
                    mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats)

        mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                            rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        return loss_mask

    def _bbox_forward_test(self, stage, x, rois, semantic_feat=None):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)
        return cls_score, bbox_pred

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN part, the same as normal two-stage detectors
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                    gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                loss_mask = self._mask_forward_train(i, x, sampling_results,
                                                     gt_masks, rcnn_train_cfg,
                                                     semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        return losses

    def transform(self, det_bboxes, scale_factor, rescale):
        if det_bboxes.shape[0] == 0:
            return det_bboxes
        else:
            # _bboxes = (
            #     det_bboxes[:, :4] * scale_factor
            #     if rescale else det_bboxes)
            if isinstance(scale_factor, float):
                _bboxes = (det_bboxes[:, :4] * scale_factor
                           if rescale else det_bboxes)
            else:
                _bboxes = (det_bboxes[:, :4] * torch.from_numpy(
                    scale_factor).to(det_bboxes.device)
                           if rescale else det_bboxes)
            return _bboxes

    def get_bboxes(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_meta,
                    rescale=False,
                    cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        target_means = [0., 0., 0., 0.],
        target_stds = [0.033, 0.033, 0.067, 0.067]
        scale_factors = torch.Tensor([img_meta[j]['scale_factor'] for j in range(len(img_meta))])
        index = rois[:, 0].clone().long()
        scale_factor = scale_factors[index]
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        bboxes = delta2bbox(rois[:, 1:], bbox_pred, target_means, target_stds)
        if rescale:
            bboxes /= scale_factor.unsqueeze(1).to(bboxes.device)

        return bboxes, scores

    # def multi_batch_test(self, img, img_meta, proposals=None, rescale=False):
    #     t1 = time.time()
    #     x = self.extract_feat(img)
    #     max_roi = self.test_cfg.rcnn.max_per_img
    #     proposal_list = self.simple_test_rpn(
    #         x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
    #     batch_size = len(proposal_list)
    #     semantic_feat = None
    #
    #     # "ms" in variable names means multi-stage
    #     ms_bbox_result = {}
    #     ms_segm_result = {}
    #     bbox_results = []
    #     segm_results = []
    #     ms_scores = []
    #     rcnn_test_cfg = self.test_cfg.rcnn
    #     rois = bbox2roi(proposal_list)
    #     t4 = time.time()
    #     for i in range(self.num_stages):
    #         bbox_head = self.bbox_head[i]
    #         cls_score, bbox_pred = self._bbox_forward_test(
    #             i, x, rois, semantic_feat=semantic_feat)
    #         ms_scores.append(cls_score)
    #         if i < self.num_stages - 1:
    #             bbox_label = cls_score.argmax(dim=1)
    #             rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
    #                                               img_meta[0])
    #     cls_score = sum(ms_scores) / float(len(ms_scores))
    #     det_bboxes_list = []
    #     det_labels_list = []
    #
    #     results = []
    #     tn = time.time()
    #     for j in range(batch_size):
    #         curr_cls_score = cls_score[j*max_roi:(j+1)*max_roi]
    #         curr_bbox_pred = bbox_pred[j*max_roi:(j+1)*max_roi]
    #         curr_rois = rois[j*max_roi:(j+1)*max_roi]
    #         img_shape = img_meta[j]['img_shape']
    #         ori_shape = img_meta[j]['ori_shape']
    #         scale_factor = img_meta[j]['scale_factor']
    #         results.append(self.p.apply_async(get_det_bboxes, args=(j, curr_rois.cpu(), F.softmax(curr_cls_score,
    #                                                                                         dim=1).cpu(),
    #                                                            curr_bbox_pred.cpu(),
    #                                                            img_shape, scale_factor, rescale, rcnn_test_cfg)))
    #         # det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
    #         #     curr_rois,
    #         #     curr_cls_score,
    #         #     curr_bbox_pred,
    #         #     img_shape,
    #         #     scale_factor,
    #         #     rescale=rescale,
    #         #     cfg=rcnn_test_cfg)
    #     boxes_list = [[] for i in range(batch_size)]
    #     labels_list = [[] for i in range(batch_size)]
    #     for res in results:
    #         index, det_bboxes, det_labels = res.get()
    #         labels_list[index] = det_labels
    #         boxes_list[index] = det_bboxes
    #     tw = time.time()
    #     for j in range(batch_size):
    #         scale_factor = img_meta[j]['scale_factor']
    #         det_bboxes = boxes_list[j]
    #         det_labels = labels_list[j]
    #         det_bboxes_list.append(self.transform(det_bboxes, scale_factor, rescale).to(bbox_pred.device))
    #         det_labels_list.append(det_labels.to(bbox_pred.device))
    #         bbox_result = bbox2result(det_bboxes, det_labels,
    #                                   self.bbox_head[-1].num_classes)
    #         bbox_results.append(bbox_result)
    #     tm = time.time()
    #     ic(tm-tn)
    #     # if det_bboxes.shape[0] == 0:
    #     #     segm_result = [
    #     #         [] for _ in range(self.mask_head[-1].num_classes - 1)
    #     #     ]
    #     # else:
    #     #     # _bboxes = (
    #     #     #     det_bboxes[:, :4] * scale_factor
    #     #     #     if rescale else det_bboxes)
    #     #     if isinstance(scale_factor, float):
    #     #         _bboxes = (det_bboxes[:, :4] * scale_factor
    #     #                    if rescale else det_bboxes)
    #     #     else:
    #     #         _bboxes = (det_bboxes[:, :4] * torch.from_numpy(
    #     #             scale_factor).to(det_bboxes.device)
    #     #                    if rescale else det_bboxes)
    #
    #     # mask_rois = bbox2roi([_bboxes])
    #     ret = []
    #     mask_rois = bbox2roi(det_bboxes_list)
    #     if mask_rois.shape[0] == 0:
    #         for i in range(len(det_labels_list)):
    #             segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
    #             ms_segm_result['ensemble'] = segm_result
    #             ms_bbox_result['ensemble'] = bbox_results[i]
    #             results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
    #             ret.append(results)
    #         return ret
    #     aug_masks = []
    #     mask_roi_extractor = self.mask_roi_extractor[-1]
    #     mask_feats = mask_roi_extractor(
    #         x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
    #     # if self.with_semantic and 'mask' in self.semantic_fusion:
    #     #     mask_semantic_feat = self.semantic_roi_extractor(
    #     #         [semantic_feat], mask_rois)
    #     #     mask_feats += mask_semantic_feat
    #     last_feat = None
    #     for i in range(self.num_stages):
    #         mask_head = self.mask_head[i]
    #         if self.mask_info_flow:
    #             mask_pred, last_feat = mask_head(mask_feats, last_feat)
    #         else:
    #             mask_pred = mask_head(mask_feats)
    #         aug_masks.append(mask_pred.sigmoid().cpu().numpy())
    #     merged_masks = merge_aug_masks(aug_masks,
    #                                    [img_meta] * self.num_stages,
    #                                    self.test_cfg.rcnn)
    #     offset = 0
    #     for i, _bboxes in enumerate(det_bboxes_list):
    #         ori_shape = img_meta[i]['ori_shape']
    #         scale_factor = img_meta[i]['scale_factor']
    #         if _bboxes.shape[0] == 0:
    #             segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
    #         else:
    #             length = len(_bboxes)
    #             segm_result = self.mask_head[-1].get_seg_masks(
    #                 merged_masks[offset:offset + length], _bboxes, det_labels_list[i], rcnn_test_cfg,
    #                 ori_shape, scale_factor, rescale)
    #             offset += length
    #         segm_results.append(segm_result)
    #     t9 = time.time()
    #     # ic(t9 - t1)
    #     for i in range(len(segm_results)):
    #         ms_segm_result['ensemble'] = segm_results[i]
    #         ms_bbox_result['ensemble'] = bbox_results[i]
    #         results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
    #         ret.append(results)
    #     return ret
    def multi_batch_test(self, img, img_meta, proposals=None, rescale=False):
        t1 = time.time()
        x = self.extract_feat(img)
        max_roi = self.test_cfg.rcnn.max_per_img
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        batch_size = len(proposal_list)
        semantic_feat = None

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        bbox_results = []
        segm_results = []
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn
        rois = bbox2roi(proposal_list)
        t4 = time.time()
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])
        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes_list = []
        det_labels_list = []
        tn = time.time()
        for j in range(batch_size):
            curr_cls_score = cls_score[j*max_roi:(j+1)*max_roi]
            curr_bbox_pred = bbox_pred[j*max_roi:(j+1)*max_roi]
            curr_rois = rois[j*max_roi:(j+1)*max_roi]
            img_shape = img_meta[j]['img_shape']
            ori_shape = img_meta[j]['ori_shape']
            scale_factor = img_meta[j]['scale_factor']
            det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
                curr_rois,
                curr_cls_score,
                curr_bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes_list.append(self.transform(det_bboxes, scale_factor, rescale))
            det_labels_list.append(det_labels)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.bbox_head[-1].num_classes)
            bbox_results.append(bbox_result)
        tm = time.time()
        # if det_bboxes.shape[0] == 0:
        #     segm_result = [
        #         [] for _ in range(self.mask_head[-1].num_classes - 1)
        #     ]
        # else:
        #     # _bboxes = (
        #     #     det_bboxes[:, :4] * scale_factor
        #     #     if rescale else det_bboxes)
        #     if isinstance(scale_factor, float):
        #         _bboxes = (det_bboxes[:, :4] * scale_factor
        #                    if rescale else det_bboxes)
        #     else:
        #         _bboxes = (det_bboxes[:, :4] * torch.from_numpy(
        #             scale_factor).to(det_bboxes.device)
        #                    if rescale else det_bboxes)

        # mask_rois = bbox2roi([_bboxes])
        ret = []
        mask_rois = bbox2roi(det_bboxes_list)
        if mask_rois.shape[0] == 0:
            for i in range(len(det_labels_list)):
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
                ms_segm_result['ensemble'] = segm_result
                ms_bbox_result['ensemble'] = bbox_results[i]
                results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
                ret.append(results)
            return ret
        aug_masks = []
        mask_roi_extractor = self.mask_roi_extractor[-1]
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        # if self.with_semantic and 'mask' in self.semantic_fusion:
        #     mask_semantic_feat = self.semantic_roi_extractor(
        #         [semantic_feat], mask_rois)
        #     mask_feats += mask_semantic_feat
        last_feat = None
        for i in range(self.num_stages):
            mask_head = self.mask_head[i]
            if self.mask_info_flow:
                mask_pred, last_feat = mask_head(mask_feats, last_feat)
            else:
                mask_pred = mask_head(mask_feats)
            aug_masks.append(mask_pred.sigmoid().cpu().numpy())
        merged_masks = merge_aug_masks(aug_masks,
                                       [img_meta] * self.num_stages,
                                       self.test_cfg.rcnn)
        offset = 0
        for i, _bboxes in enumerate(det_bboxes_list):
            ori_shape = img_meta[i]['ori_shape']
            scale_factor = img_meta[i]['scale_factor']
            if _bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
            else:
                length = len(_bboxes)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks[offset:offset + length], _bboxes, det_labels_list[i], rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
                offset += length
            segm_results.append(segm_result)
        t9 = time.time()
        # ic(t9 - t1)
        for i in range(len(segm_results)):
            ms_segm_result['ensemble'] = segm_results[i]
            ms_bbox_result['ensemble'] = bbox_results[i]
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
            ret.append(results)
        return ret

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    nms_cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        segm_result = [
                            [] for _ in range(mask_head.num_classes - 1)
                        ]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] * scale_factor
                            if rescale else det_bboxes)
                        mask_pred = self._mask_forward_test(
                            i, x, _bboxes, semantic_feat=semantic_feat)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])
        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result
        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [
                    [] for _ in range(self.mask_head[-1].num_classes - 1)
                ]
            else:
                _bboxes = (
                    det_bboxes[:, :4] * scale_factor
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result
        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage])
                    for stage in ms_bbox_result
                }
            else:
                results = ms_bbox_result
        return results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError
