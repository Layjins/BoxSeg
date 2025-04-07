import cv2
import math
import copy
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

def draw(img, bboxes, masks, idx):
    out_dir = '/youtu/fuxi-team2-2/persons/ezrealwu/workspace/research/segmentation/BoxSeg/taiji/tmp'
    draw = img.permute(1,2,0).cpu().numpy().copy()
    bboxes = bboxes.cpu().numpy()
    masks = masks.cpu().numpy()
    for i in range(len(bboxes)):
        cv2.rectangle(draw, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (0,0,255), 1)
        cv2.imwrite(f'{out_dir}/{idx}_mask{i}.png', masks[i]*255)
    cv2.imwrite(f'{out_dir}/{idx}_img.png', draw)


class BoxSegCopyPaste():
    def __init__(self, copypaste_type='instance', device='cpu'):
        self.iou_thr = 0.1
        self.overlap_radio = 0.1
        self.score_thr = 0.5
        self.bbox_occluded_thr = 10
        self.mask_occluded_thr = 300
        self.mem_bank_max_num = 80
        self.mem_bank = OrderedDict()
        self.rate = 0.7
        self.copypaste_type = copypaste_type   # ['simple', 'segmentation', 'instance'] 
        self.device = device
        self._copypaste_iter = 0
        self._mem_update_iter = 0

    def transform(self, item, _iter):
        cur_img, instances = item
        img = cur_img.to(torch.float32)
        labels = instances.gt_classes.to(self.device)
        bboxes = instances.gt_boxes.tensor.to(self.device)
        scores = instances.pred_scores.to(torch.float32).to(self.device)
        masks = instances.gt_bitmasks.to(torch.float32).to(self.device)
        mask_scores = instances.gt_masks_flags.to(self.device)
        bbox_masks = self._get_masks_from_boxes(bboxes, masks.shape[1:])
        masks = masks * bbox_masks

        warmup_factor = min(_iter / 10000, 1.0)
        #warmup_factor = math.sqrt(_iter / 90000)
        if np.random.rand() < self.rate * warmup_factor:
            self._update_bank(img, labels, bboxes, masks, mask_scores, scores)
            return item

        src_img, src_label, src_bbox, src_mask, src_mask_score = self._get_selected_inds(labels, bboxes, mask_scores, img.shape[1:])
        self._update_bank(img, labels, bboxes, masks, mask_scores, scores)

        if src_img is None:
            return item

        result_img, result_bboxes, result_masks, valid_inds = self._copy_paste(src_img, src_bbox, src_mask, img, bboxes, masks)
        result_labels = torch.cat([labels[valid_inds], src_label])
        result_mask_scores = torch.cat([mask_scores[valid_inds], src_mask_score])

        #draw(src_img, src_bbox, src_mask, 'src', _iter)
        #draw(img, bboxes, masks, 'dst', _iter)
        #draw(result_img, result_bboxes, result_masks, 'copypaste', _iter)

        result_img = result_img.to(torch.uint8)
        result_instances = Instances(result_img.shape[1:])
        result_instances.gt_classes = result_labels
        result_instances.gt_boxes = Boxes(result_bboxes)
        result_instances.gt_bitmasks = result_masks
        result_instances.gt_masks_flags = result_mask_scores

        self._copypaste_iter += 1
        get_event_storage().put_scalar("iter", _iter)
        get_event_storage().put_scalar("copypaste_iter", self._copypaste_iter)
        get_event_storage().put_scalar("num_mem_bank", len(self.mem_bank))
        get_event_storage().put_scalar("mem_update_iter", self._mem_update_iter)

        return (result_img, result_instances)



    def _update_bank(self, img, labels, bboxes, masks, mask_scores, scores):
        ious = self._pairwise_iou(bboxes, bboxes)
        ious = ious - torch.eye(len(bboxes), dtype=torch.float32, device=self.device)
        no_occluded_inds = (ious < self.iou_thr).all(dim=0)

        high_score_inds = (mask_scores > self.score_thr) & (scores > self.score_thr)

        binary_masks = (masks > 0.5).float()
        mask_area_inds = binary_masks.sum(dim=(1, 2)) > self.mask_occluded_thr
        selected_inds = no_occluded_inds & high_score_inds & mask_area_inds

        selected_labels = labels[selected_inds]
        selected_bboxes = bboxes[selected_inds]
        selected_masks = binary_masks[selected_inds]
        selected_mask_scores = mask_scores[selected_inds]
        assert len(selected_labels) == len(selected_bboxes) == len(selected_masks) == len(selected_mask_scores)

        key_labels = tuple(selected_labels.cpu().numpy().tolist())
        if key_labels and key_labels not in self.mem_bank:
            self.mem_bank[key_labels] = (img.cpu(), selected_labels.cpu(), selected_bboxes.cpu(), selected_masks.cpu(), selected_mask_scores.cpu())
            self._mem_update_iter += 1

        if len(self.mem_bank) > self.mem_bank_max_num:
            drop_labels = random.choice(list(self.mem_bank.keys()))
            self.mem_bank.pop(drop_labels)
            #self.mem_bank.popitem(last=False)

        #print(f'current gpu : {torch.cuda.current_device()}, memory bank num : {len(self.mem_bank)}, memory bank keys : {self.mem_bank.keys()}')


    def _get_selected_inds(self, labels, bboxes, mask_scores, img_shape):
        if not self.mem_bank:
            return None, None, None, None, None

        if self.copypaste_type == 'simple':
            #from ipdb import set_trace; set_trace()
            key_labels = random.choice(list(self.mem_bank.keys()))
            selected_img, selected_label, selected_bbox, selected_mask, selected_mask_score = self.mem_bank[key_labels]
            selected_img = selected_img.to(self.device)
            selected_label = selected_label.to(self.device)
            selected_bbox = selected_bbox.to(self.device)
            selected_mask = selected_mask.to(self.device)
            selected_mask_score = selected_mask_score.to(self.device)

            h, w = img_shape
            s_h, s_w = selected_img.shape[1:]
            pad_bottom = max(0, h - s_h)
            pad_right = max(0, w - s_w)
            selected_img = F.pad(selected_img, (0, pad_right, 0, pad_bottom), 'constant', 0.)
            selected_mask = F.pad(selected_mask, (0, pad_right, 0, pad_bottom), 'constant', 0.)
            selected_img = selected_img[:,:h,:w]
            selected_mask = selected_mask[:,:h,:w]

            get_bbox = self._get_bounding_boxes(selected_mask)
            valid_inds = selected_mask.sum(dim=(1, 2)) > self.mask_occluded_thr
            if not valid_inds.any():
                return None, None, None, None, None

            selected_label = selected_label[valid_inds]
            selected_bbox = get_bbox[valid_inds]
            selected_mask = selected_mask[valid_inds]
            selected_mask_score = selected_mask_score[valid_inds]
            assert len(selected_label) == len(selected_bbox) == len(selected_mask) == len(selected_mask_score)

            self.mem_bank.pop(key_labels)
            return selected_img, selected_label, selected_bbox, selected_mask, selected_mask_score

        elif self.copypaste_type == 'segmentation':
            ious = self._pairwise_iou(bboxes, bboxes)
            ious = ious - torch.eye(len(bboxes), dtype=torch.float32, device=self.device)
            #no_occluded_inds = (ious < self.iou_thr).all(dim=0)
            no_occluded_inds = torch.ones((len(labels),), dtype=torch.bool, device=self.device)
            if not no_occluded_inds.any():
                return None, None, None, None, None

            #from ipdb import set_trace; set_trace()
            for key_labels in self.mem_bank.keys():
                key_labels = random.choice(list(self.mem_bank.keys()))
                bank_img, bank_labels, bank_bboxes, bank_masks, bank_mask_scores = self.mem_bank[key_labels]
                bank_labels = bank_labels.to(self.device)
                bank_img = bank_img.to(self.device)
                bank_bboxes = bank_bboxes.to(self.device)
                bank_masks = bank_masks.to(self.device)
                bank_mask_scores = bank_mask_scores.to(self.device)

                bank_selected_ind = torch.zeros(bank_mask_scores.shape, dtype=torch.bool, device=self.device)
                bank_selected_ind[bank_mask_scores.argmax()] = True

                #bank_selected_ind = bank_mask_scores == bank_mask_scores.max()
                final_selected_ind = np.random.choice(len(labels[no_occluded_inds]))

                h, w = img_shape
                s_h, s_w = bank_img.shape[1:]
                src_x1, src_y1, src_x2, src_y2 = bank_bboxes[bank_selected_ind].squeeze().cpu().numpy().tolist()
                src_w, src_h = src_x2 - src_x1, src_y2 - src_y1
                dst_x1, dst_y1, dst_x2, dst_y2 = bboxes[no_occluded_inds][final_selected_ind].squeeze().cpu().numpy().tolist()

                shift_x1 = np.random.uniform(dst_x1 - src_w * (1-self.overlap_radio), dst_x2 - src_w * self.overlap_radio)
                shift_y1 = np.random.uniform(dst_y1 - src_h * (1-self.overlap_radio), dst_y2 - src_h * self.overlap_radio)

                delta_x = round(shift_x1 - src_x1)
                delta_y = round(shift_y1 - src_y1)
                pad_left = max(0, delta_x)
                pad_right = max(0, w - s_w - delta_x)
                pad_top = max(0, delta_y)
                pad_bottom = max(0, h - s_h - delta_y)
                shift_img = F.pad(bank_img, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0.)
                shift_mask = F.pad(bank_masks[bank_selected_ind], (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0.)
                shift_img = shift_img[:, max(0,-delta_y):h+max(0,-delta_y), max(0,-delta_x):w+max(0,-delta_x)]
                shift_mask = shift_mask[:, max(0,-delta_y):h+max(0,-delta_y), max(0,-delta_x):w+max(0,-delta_x)]

                get_bbox = self._get_bounding_boxes(shift_mask)
                valid_inds = shift_mask.sum(dim=(1, 2)) > self.mask_occluded_thr
                if not valid_inds.any():
                    return None, None, None, None, None

                shift_label = bank_labels[bank_selected_ind][valid_inds]
                shift_bbox = get_bbox[valid_inds]
                shift_mask = shift_mask[valid_inds]
                shift_mask_score = bank_mask_scores[bank_selected_ind][valid_inds]
                assert len(shift_label) == len(shift_bbox) == len(shift_mask) == len(shift_mask_score)

                self.mem_bank.pop(key_labels)
                return shift_img, shift_label, shift_bbox, shift_mask, shift_mask_score
            return None, None, None, None, None

        elif self.copypaste_type == 'instance':
            ious = self._pairwise_iou(bboxes, bboxes)
            ious = ious - torch.eye(len(bboxes), dtype=torch.float32, device=self.device)
            #no_occluded_inds = (ious < self.iou_thr).all(dim=0)
            no_occluded_inds = torch.ones((len(labels),), dtype=torch.bool, device=self.device)

            for key_labels in self.mem_bank.keys():
                key_labels = random.choice(list(self.mem_bank.keys()))
                bank_img, bank_labels, bank_bboxes, bank_masks, bank_mask_scores = self.mem_bank[key_labels]
                bank_labels = bank_labels.to(self.device)
                equal_inds = bank_labels[:,None] == labels[no_occluded_inds][None,:]
                if not equal_inds.any():
                    continue

                #from ipdb import set_trace; set_trace()
                bank_img = bank_img.to(self.device)
                bank_bboxes = bank_bboxes.to(self.device)
                bank_masks = bank_masks.to(self.device)
                bank_mask_scores = bank_mask_scores.to(self.device)

                bank_equal_inds = equal_inds.any(dim=1)
                #bank_selected_bboxes = bank_bboxes[bank_equal_inds]
                #bank_selected_bboxes_areas = (bank_selected_bboxes[:,2]-bank_selected_bboxes[:,0]) * (bank_selected_bboxes[:,3]-bank_selected_bboxes[:,1])
                #bank_selected_mask_scores = bank_mask_scores[bank_equal_inds]
                #bank_selected_ind = (bank_selected_bboxes_areas/bank_selected_bboxes_areas.max()).sqrt() * (bank_selected_mask_scores/bank_selected_mask_scores.max())
                #bank_selected_ind = bank_selected_ind == bank_selected_ind.max()
                bank_selected_ind = bank_mask_scores[bank_equal_inds] == bank_mask_scores[bank_equal_inds].max()

                selected_label = bank_labels[bank_equal_inds][bank_selected_ind][0]
                selected_inds = labels[no_occluded_inds] == selected_label
                #selected_bboxes = bboxes[no_occluded_inds][selected_inds]
                #selected_bbox_areas = (selected_bboxes[:,2]-selected_bboxes[:,0]) * (selected_bboxes[:,3]-selected_bboxes[:,1])
                #selected_mask_scores = mask_scores[no_occluded_inds][selected_inds]
                #final_selected_ind = (selected_bbox_areas/selected_bbox_areas.max()).sqrt() * (selected_mask_scores/selected_mask_scores.max())
                #final_selected_ind = final_selected_ind.argmax()
                final_selected_ind = np.random.choice(len(labels[no_occluded_inds][selected_inds]))


                #valid_selected_inds = labels[no_occluded_inds][selected_inds]
                #if np.random.rand() > min(len(valid_selected_inds) / 5, 1.0):
                #    return None, None, None, None, None


                h, w = img_shape
                s_h, s_w = bank_img.shape[1:]
                src_x1, src_y1, src_x2, src_y2 = bank_bboxes[bank_equal_inds][bank_selected_ind].squeeze().cpu().numpy().tolist()
                src_w, src_h = src_x2 - src_x1, src_y2 - src_y1
                dst_x1, dst_y1, dst_x2, dst_y2 = bboxes[no_occluded_inds][selected_inds][final_selected_ind].squeeze().cpu().numpy().tolist()

                shift_x1 = np.random.uniform(dst_x1 - src_w * (1-self.overlap_radio), dst_x2 - src_w * self.overlap_radio)
                shift_y1 = np.random.uniform(dst_y1 - src_h * (1-self.overlap_radio), dst_y2 - src_h * self.overlap_radio)

                delta_x = round(shift_x1 - src_x1)
                delta_y = round(shift_y1 - src_y1)
                pad_left = max(0, delta_x)
                pad_right = max(0, w - s_w - delta_x)
                pad_top = max(0, delta_y)
                pad_bottom = max(0, h - s_h - delta_y)
                shift_img = F.pad(bank_img, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0.)
                shift_mask = F.pad(bank_masks[bank_equal_inds][bank_selected_ind], (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0.)
                shift_img = shift_img[:, max(0,-delta_y):h+max(0,-delta_y), max(0,-delta_x):w+max(0,-delta_x)]
                shift_mask = shift_mask[:, max(0,-delta_y):h+max(0,-delta_y), max(0,-delta_x):w+max(0,-delta_x)]

                get_bbox = self._get_bounding_boxes(shift_mask)
                valid_inds = shift_mask.sum(dim=(1, 2)) > self.mask_occluded_thr
                if not valid_inds.any():
                    return None, None, None, None, None

                shift_label = bank_labels[bank_equal_inds][bank_selected_ind][valid_inds]
                shift_bbox = get_bbox[valid_inds]
                shift_mask = shift_mask[valid_inds]
                shift_mask_score = bank_mask_scores[bank_equal_inds][bank_selected_ind][valid_inds]
                assert len(shift_label) == len(shift_bbox) == len(shift_mask) == len(shift_mask_score)

                self.mem_bank.pop(key_labels)
                return shift_img, shift_label, shift_bbox, shift_mask, shift_mask_score

            return None, None, None, None, None
        else:
            raise ValueError(f'Unknown copy-paste type : {self.copypaste_type}')

    def _copy_paste(self, src_img, src_bboxes, src_masks, dst_img, dst_bboxes, dst_masks):
        if len(src_bboxes) == 0:
            return dst_img, dst_bboxes, dst_masks, torch.ones((len(dst_bboxes),), dtype=torch.bool, device=dst_img.device)

        # update masks and generate bboxes from updated masks
        composed_mask = torch.where(torch.any(src_masks, dim=0), torch.ones(src_masks.shape[1:], dtype=torch.float32, device=self.device), torch.zeros(src_masks.shape[1:], dtype=torch.float32, device=self.device))
        updated_dst_masks = torch.where(composed_mask[None,...]>0.5, torch.zeros(dst_masks.shape, dtype=torch.float32, device=self.device), dst_masks)
        #updated_dst_bboxes = self._get_bounding_boxes(updated_dst_masks)

        #from ipdb import set_trace; set_trace()
        dst_bbox_masks = self._get_masks_from_boxes(dst_bboxes, dst_masks.shape[1:])
        updated_dst_bbox_masks = torch.where(composed_mask[None,...]>0.5, torch.zeros(dst_bbox_masks.shape, dtype=torch.float32, device=self.device), dst_bbox_masks)
        updated_dst_bboxes = self._get_bounding_boxes(updated_dst_bbox_masks)

        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes - dst_bboxes).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(dim=-1)
        masks_inds = updated_dst_masks.sum(dim=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[None, ...]) + src_img * composed_mask[None, ...]
        bboxes = torch.cat([updated_dst_bboxes[valid_inds], src_bboxes])
        masks = torch.cat([updated_dst_masks[valid_inds], src_masks])

        return img, bboxes, masks, valid_inds


    def _pairwise_iou(self, bboxes1, bboxes2):
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        #lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])
        #rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])
        #wh = np.maximum(rb - lt, 0)
        lt = torch.max(bboxes1[:, None, :2], bboxes2[None, :, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[None, :, 2:])  # [rows, cols, 2]
        wh = (rb - lt).clamp(min=0)

        overlap = wh[:, :, 0] * wh[:, :, 1]
        ious = overlap / (area1[:, None] + area2[None, :] - overlap)

        return ious

    def _get_bounding_boxes(self, masks):
        num_masks = len(masks)
        boxes = torch.zeros((num_masks, 4), dtype=torch.float32, device=masks.device)
        x_any = masks.any(dim=1)
        y_any = masks.any(dim=2)
        for idx in range(num_masks):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                # use +1 for x_max and y_max so that the right and bottom
                # boundary of instance masks are fully included by the box
                boxes[idx, :] = torch.as_tensor([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32, device=masks.device)
        return boxes

    def _get_masks_from_boxes(self, boxes, mask_shape):
        num_boxes = len(boxes)
        masks = torch.zeros((num_boxes,)+mask_shape, dtype=torch.float32, device=boxes.device)
        for idx, box in enumerate(boxes):
            masks[idx, int(box[1]):int(box[3] + 1), int(box[0]):int(box[2] + 1)] = 1.0
        return masks



