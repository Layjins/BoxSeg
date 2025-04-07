import logging
import torch
from torch import nn


from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import pairwise_iou

from .condinst import CondInst


__all__ = ["BoxSeg"]
logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class BoxSeg(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.eval_teacher = cfg.MODEL.BOX_TEACHER.TEACHER_EVAL
        self.use_teacher_inference = cfg.MODEL.BOX_TEACHER.USE_TEACHER_INFERENCE
        # iou thr
        self.filter_iou_thr = cfg.MODEL.BOX_TEACHER.IOU_THR
        self.filter_score_thr = cfg.MODEL.BOX_TEACHER.SCORE_THR
        self.teacher_mask_threshold = cfg.MODEL.BOX_TEACHER.TEACHER_MASK_THRESHOLD
        self.teacher_with_nms = cfg.MODEL.BOX_TEACHER.TEACHER_WITH_NMS
        self.dynamic_mask_thr = cfg.MODEL.BOX_TEACHER.DYNAMIC_MASK_THRESH
        self.label_hist = torch.ones((cfg.MODEL.FCOS.NUM_CLASSES,), dtype=torch.float32) * self.teacher_mask_threshold
        self.register_buffer("_iter", torch.zeros([1]))

        self.teacher = CondInst(cfg)
        self.student = CondInst(cfg)
        self.freeze_teacher()

    def pseudo_labeling1(self, batched_inputs, instances, iou_thr=0.5, score_thr=0.0):

        for idx, instances_per_im in enumerate(instances):
            batched_inputs[idx]["teacher_mask_feat"] = instances_per_im["teacher_mask_feat"]
            gt_instances = batched_inputs[idx]["instances"]
            pred_instances = instances_per_im["instances"]
            M = len(pred_instances)
            N = len(gt_instances)
            # print(M, N)
            if N == 0 or not pred_instances.has("pred_masks"):
                h, w = batched_inputs[idx]['image'].shape[1:]
                batched_inputs[idx]["instances"].gt_bitmasks = torch.zeros(
                    (N, h, w), dtype=torch.bool, device=self.device)
                batched_inputs[idx]["instances"].gt_masks_flags = torch.zeros(
                    (N,), dtype=torch.float, device=self.device)
                batched_inputs[idx]["instances"].pred_scores = torch.zeros(
                    (N,), dtype=torch.float, device=self.device)
                continue
            pred_masks = pred_instances.pred_masks.to(self.device)

            gt_boxes = gt_instances.gt_boxes.to(self.device)
            pred_boxes = pred_instances.pred_boxes.to(self.device)
            pred_scores = pred_instances.scores.to(self.device)
            iou = pairwise_iou(pred_boxes, gt_boxes)
            
            sort_index = torch.argsort(pred_scores, descending=True)
            dtm = torch.zeros((M,), dtype=torch.long, device=self.device)
            gtm = torch.zeros((N,), dtype=torch.long, device=self.device) - 1
            biou = torch.zeros((N,), dtype=torch.float, device=self.device)

            for i in sort_index:
                if pred_scores[i] < score_thr:
                    continue
                max_iou = -1
                m = -1
                for j in range(N):
                    iou_ = iou[i, j]
                    if gtm[j] > 0 or iou_ < iou_thr or iou_ < max_iou:
                        continue
                    max_iou = iou_
                    m = j
                if m == -1:
                    continue
                dtm[i] = m
                gtm[m] = i
                biou[m] = max_iou

            new_instances = gt_instances
            new_masks_inds = gtm[gtm > -1]
            gt_masks = torch.zeros((N, pred_masks.shape[-2], pred_masks.shape[-1])).to(pred_masks)
            gt_masks[gtm > -1] = pred_masks[new_masks_inds]
            new_instances.gt_bitmasks = gt_masks
            new_instances.gt_masks_flags = (gtm > -1).float()
            new_instances.gt_masks_flags[gtm > -1] = pred_instances.mask_score[new_masks_inds]
            new_instances.pred_scores = (gtm > -1).float()
            new_instances.pred_scores[gtm > -1] = pred_instances.pred_scores[new_masks_inds]
            batched_inputs[idx]["instances"] = new_instances

    def pseudo_labeling2(self, batched_inputs, instances, iou_thr=0.5, score_thr=0.0):
        if self.training:
            self._iter += 1

        for idx, instances_per_im in enumerate(instances):
            batched_inputs[idx]["teacher_mask_feat"] = instances_per_im["teacher_mask_feat"]
            gt_instances = batched_inputs[idx]["instances"]
            pred_instances = instances_per_im["instances"]
            M = len(pred_instances)
            N = len(gt_instances)
            # print(M, N)
            if N == 0 or not pred_instances.has("pred_masks"):
                h, w = batched_inputs[idx]['image'].shape[1:]
                batched_inputs[idx]["instances"].gt_bitmasks = torch.zeros(
                    (N, h, w), dtype=torch.bool, device=self.device)
                batched_inputs[idx]["instances"].gt_masks_flags = torch.zeros(
                    (N,), dtype=torch.float, device=self.device)
                batched_inputs[idx]["instances"].pred_scores = torch.zeros(
                    (N,), dtype=torch.float, device=self.device)
                continue

            nums_per_gt = pred_instances.nums_per_gt[0].cpu().numpy().tolist()
            pred_labels = pred_instances.pred_classes.to(self.device)
            pred_scores = pred_instances.pred_scores.to(self.device)
            pred_boxes = pred_instances.pred_boxes.tensor.to(self.device)
            pred_masks = pred_instances.pred_masks.to(self.device)
            gt_labels = gt_instances.gt_classes.to(self.device)
            gt_boxes = gt_instances.gt_boxes.tensor.to(self.device)

            pseudo_pred_scores = pred_scores.new_zeros((len(nums_per_gt),))
            pseudo_scores = pred_scores.new_zeros((len(nums_per_gt),))
            pseudo_masks = pred_masks.new_zeros((len(nums_per_gt),)+pred_masks.shape[1:])

            split_pred_labels = pred_labels.split(nums_per_gt, dim=0)
            split_pred_boxes = pred_boxes.split(nums_per_gt, dim=0)
            split_pred_scores = pred_scores.split(nums_per_gt, dim=0)
            split_pred_masks = pred_masks.split(nums_per_gt, dim=0)
            for i in range(len(nums_per_gt)):
                if nums_per_gt[i] == 0:
                    continue

                if self.dynamic_mask_thr and self._iter.item() > 10000:
                #if self.dynamic_mask_thr:
                    # update the threshhold of every label by std of top 10 masks of each gt
                    label_rate = (split_pred_labels[i]==gt_labels[i]).sum() / nums_per_gt[i]
                    delta_thr = -0.1 * label_rate + 0.2   # label_rate is range from 0 to 1
                    range_thr = torch.arange(self.teacher_mask_threshold - delta_thr,
                        self.teacher_mask_threshold + delta_thr, 0.01)
                    bin_pred_masks = split_pred_masks[i][None,:,:,:] > range_thr[:,None,None,None].to(gt_labels.device)
                    std = bin_pred_masks.float().std(dim=1).mean(dim=(-2,-1))
                    #mean_bin_pred_masks = bin_pred_masks.float().mean(dim=1, keepdim=True).gt(0.5)
                    #bin_mious = (bin_pred_masks & mean_bin_pred_masks).sum(dim=(-2,-1)) / (bin_pred_masks | mean_bin_pred_masks).sum(dim=(-2,-1)).clamp(min=1.0)
                    #std = bin_mious.std(dim=1)

                    std_ind = std.argmin(dim=0)
                    #self.label_hist[gt_labels[i]] = self.label_hist[gt_labels[i]] * 0.999 + range_thr[std_ind] * 0.001   # ema update
                    self.label_hist[gt_labels[i]] = range_thr[std_ind]   # realtime update

                # mask aware confidence score
                #per_pred_masks = split_pred_masks[i].gt(self.label_hist[gt_labels[i]])
                #mean_pred_masks = per_pred_masks.float().mean(dim=0).gt(0.5)
                #aware_aves = (split_pred_masks[i]*per_pred_masks.float()).sum(dim=(1,2)) / per_pred_masks.float().sum(dim=(1,2)).clamp(min=1.0)
                ##aware_scores = (split_pred_scores[i].sqrt() * aware_aves).sqrt()
                #aware_scores = ((split_pred_scores[i]+1)/2 * aware_aves).sqrt()


                # new mask aware
                per_pred_masks = split_pred_masks[i].gt(self.label_hist[gt_labels[i]])
                mean_pred_masks = per_pred_masks.float().mean(dim=0).gt(0.5)
                mean_mious = (per_pred_masks & mean_pred_masks[None,:,:]).sum(dim=(-2,-1)) / (per_pred_masks | mean_pred_masks[None,:,:]).sum(dim=(-2,-1)).clamp(min=1.0)
                aware_aves = (split_pred_masks[i]*per_pred_masks.float()).sum(dim=(1,2))*mean_mious / mean_pred_masks[None,:,:].float().sum(dim=(1,2)).clamp(min=1.0)
                aware_scores = ((split_pred_scores[i]+1)/2 * aware_aves).sqrt()
                #weighted_mean_pred_masks = (per_pred_masks.float() * aware_scores[:,None,None]).sum(dim=0) / aware_scores.sum().clamp(min=1.0) > 0.5


                pseudo_pred_scores[i] = split_pred_scores[i].mean()
                pseudo_scores[i] = aware_scores.mean()
                pseudo_masks[i] = mean_pred_masks

            new_instances = gt_instances
            new_instances.gt_bitmasks = pseudo_masks
            new_instances.gt_masks_flags = pseudo_scores
            new_instances.pred_scores = pseudo_pred_scores
            batched_inputs[idx]["instances"] = new_instances


    def freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, batched_inputs):
        #from ipdb import set_trace; set_trace()
        if self.training:
            if self.eval_teacher:
                self.teacher.eval()
            else:
                self.teacher.train()

            with torch.no_grad():
                instances = self.teacher.forward_teacher(batched_inputs)

            if self.teacher_with_nms:
                self.pseudo_labeling1(batched_inputs, instances, iou_thr=self.filter_iou_thr, score_thr=self.filter_score_thr)
            else:
                self.pseudo_labeling2(batched_inputs, instances, iou_thr=self.filter_iou_thr, score_thr=self.filter_score_thr)

            del instances
            return self.student(batched_inputs)
        else:
            if self.use_teacher_inference:
                return self.teacher(batched_inputs)
            return self.student(batched_inputs)
