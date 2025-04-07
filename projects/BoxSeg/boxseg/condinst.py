import logging
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.structures import ImageList, pairwise_iou
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks, PolygonMasks, polygons_to_bitmask
from detectron2.utils.events import get_event_storage

from adet.utils.comm import aligned_bilinear, compute_locations, reduce_mean
from adet.utils.contrast import l2_normalize, momentum_update
from adet.utils.sinkhorn import distributed_sinkhorn
from adet.modeling.condinst.mask_branch import MaskBranch
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from torch_scatter import scatter_mean
# from .mask_branch import MaskBranch
from .utils import rgb2lab, parse_dynamic_params
from .loss import compute_avg_projection, compute_project_term, dice_coefficient
from .copy_paste import BoxSegCopyPaste, draw

logger = logging.getLogger(__name__)


def get_warmup_fn(warmup_method, max_iters, **kwargs):

    def _linear_warmup(cur_iter):
        if cur_iter > max_iters:
            return 1.0
        return cur_iter / max_iters

    def _static_warmup(cur_iter):
        if cur_iter > max_iters:
            return 1.0
        else:
            return 0.0

    if warmup_method == "linear":
        return _linear_warmup
    elif warmup_method == "static":
        return _static_warmup
    else:
        raise NotImplementedError()



def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    # from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_)
        + torch.exp(log_same_bg_prob - max_)
    ) + max_
    # loss = -log(prob)
    return -log_same_prob[:, 0]

def mask_focal_loss(x, targets, weights, alpha: float = 0.25, gamma: float = 2):
    ce_loss = F.binary_cross_entropy_with_logits(x, targets, weight=weights, reduction="none")
    p_t = x * targets + (1 - x) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()/(weights.sum()+1) 

class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        # boxinst configs
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        
        self.mask_weight = cfg.MODEL.BOX_TEACHER.MASK_WEIGHT
        self.with_warmup = cfg.MODEL.BOX_TEACHER.WITH_WARMUP
        self.mask_warmup_iters = cfg.MODEL.BOX_TEACHER.WARMUP_ITERS
        self.mask_warmup_fn = get_warmup_fn('linear', self.mask_warmup_iters)

        # add avg loss here
        self.with_avg_loss = cfg.MODEL.BOX_TEACHER.WITH_AVG_LOSS
        self.avg_weight = cfg.MODEL.BOX_TEACHER.AVG_LOSS_WEIGHT
        self.avg_loss = compute_avg_projection

        # add prototype loss and not resize images
        self.with_prototype_loss = cfg.MODEL.WITH_PROTOTYPE_LOSS
        self.using_resize = cfg.MODEL.BOX_TEACHER.RESIZE_IMAGES
        self.MSELoss = nn.MSELoss()

        # use pseudo mask affinity
        self.pseudo_mask_affinity_thresh = cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_THRESH
        self.pseudo_mask_affinity_weight = cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_WEIGHT
        
        self.fix_reduction = cfg.MODEL.BOX_TEACHER.FIX_REDUCTION

        self.proto_on = cfg.MODEL.PROTO_ON
        self.prototypes = nn.Parameter(torch.zeros(cfg.MODEL.FCOS.NUM_CLASSES, 10, self.in_channels), requires_grad=False)
        trunc_normal_(self.prototypes, std=0.02)

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

        self.mask_affinity_binary = cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_BINARY
        if self.proto_on:
            self.attention = nn.MultiheadAttention(self.in_channels, 4)


    def prototype_learning(self, mask_feat, masks, mask_scores, gt_classes, pred_labels, im_inds):
        for b in range(len(mask_feat)):
            gt_classes_unique = torch.unique(gt_classes[b])

            protos = self.prototypes.data.clone()
            for k in gt_classes_unique:
                inds = (im_inds==b) & (pred_labels==k)

                pseudo = mask_scores[inds]
                init_q = masks[inds]
                c_q = mask_feat[b].unsqueeze(0).expand(len(init_q), -1,-1,-1)

                init_q = rearrange(init_q, 'n p h w -> (n h w) p')
                pseudo = rearrange(pseudo, 'n p h w -> (n p h w)')
           
                c_q = rearrange(c_q, 'n h w p -> (n h w) p')

                init_q = init_q[pseudo==1]
                c_q = c_q[pseudo==1]

                if init_q.shape[0] == 0:
                    continue
                q, indexs = distributed_sinkhorn(init_q)

                f = q.transpose(0, 1) @ c_q
                f = F.normalize(f, p=2, dim=-1)
                n = torch.sum(q, dim=0)
                new_value = momentum_update(old_value=protos[k, n!=0, :], new_value=f[n!=0, :], momentum=0.999, debug=False)
                protos[k, n!=0, :] = new_value
            self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
 
        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

    def mask_heads_forward(self, features, weights, biases, num_insts):

        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def forward_test(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.proto_on:
            n,c,h,w = mask_feats.shape
            prototypes = self.prototypes.repeat(len(mask_feats), 1, 1, 1)
            q = mask_feats.flatten(2).permute(2,0,1)
            k = prototypes.flatten(1,2).permute(1,0,2)
            out = self.attention(q,k,k)[0]
            mask_feats = out.permute(1,2,0).reshape(n,c,h,w)

        if len(pred_instances) > 0:
            mask_logits = self.mask_heads_forward_with_coords(
                mask_feats, mask_feat_stride, pred_instances
            )
            pred_instances.pred_global_masks = mask_logits.sigmoid()
        return pred_instances

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None, teacher_mask_feats=None):
        #from ipdb import set_trace; set_trace()
        if self.proto_on:
            n,c,h,w = mask_feats.shape
            prototypes = self.prototypes.repeat(len(mask_feats), 1, 1, 1)
            q = mask_feats.flatten(2).permute(2,0,1)
            k = prototypes.flatten(1,2).permute(1,0,2)
            out = self.attention(q,k,k)[0]
            mask_feats = out.permute(1,2,0).reshape(n,c,h,w)

        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            # pseudo masks
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            # pseudo mask scores
            mask_flags = torch.cat([per_im.gt_masks_flags for per_im in gt_instances])
            mask_flags = mask_flags[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            # box masks
            box_gt_bitmasks = torch.cat([per_im.boxinst_gt_bitmasks for per_im in gt_instances])
            box_gt_bitmasks = box_gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                losses["loss_mask"] = dummy_loss
                losses["loss_prj"] = dummy_loss
                losses["loss_pairwise"] = dummy_loss
                losses["loss_mask_affinity"] = dummy_loss
                if self.with_avg_loss:
                    losses["loss_mask_avg"] = dummy_loss
                if self.with_prototype_loss and not self.using_resize:
                    losses["loss_prototype"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()
                # num_masks across devices
                num_instances_local = torch.tensor([1.0]).to(
                    mask_logits.device) * mask_logits.size(0)
                num_instances = max(reduce_mean(num_instances_local).item(), 1.0)

                # box-supervised BoxInst losses, color similarity
                image_color_similarity = torch.cat(
                    [x.boxinst_image_color_similarity for x in gt_instances])
                image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                loss_prj_term = compute_project_term(mask_scores, box_gt_bitmasks)

                loss_prj_term = loss_prj_term.mean()

                pairwise_losses = compute_pairwise_term(
                    mask_logits, self.pairwise_size,
                    self.pairwise_dilation
                )
                # color weights
                color_weights = (image_color_similarity >= self.pairwise_color_thresh).float() * \
                    box_gt_bitmasks.float()

                loss_pairwise = (pairwise_losses * color_weights).sum() / color_weights.sum().clamp(min=1.0)
                warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                loss_pairwise = loss_pairwise * warmup_factor

                losses.update({
                    "loss_prj": loss_prj_term,
                    "loss_pairwise": loss_pairwise,
                })


                if self.proto_on:
                    mask_feats = F.interpolate(mask_feats, scale_factor=2, mode="bilinear", align_corners=False)
                    proto_masks_list = []
                    self.prototypes.data.copy_(l2_normalize(self.prototypes))
                    mask_feats = l2_normalize(mask_feats).permute(0,2,3,1)

                    for i, x in enumerate(gt_instances):
                        masks = torch.einsum('hwd,npd->nphw', mask_feats[i], self.prototypes[x.gt_classes]).detach()
                        proto_masks_list.append(masks)
                    proto_masks = torch.cat([x for x in proto_masks_list])
                    proto_masks = proto_masks[gt_inds]

                    mask_scores_mean = scatter_mean(mask_scores, gt_inds.unsqueeze(1), dim=0)[gt_inds]
                    masks = (mask_scores_mean > 0.7) * box_gt_bitmasks.float()
                    #if self._iter > 10000:
                    #    n_pos = torch.sum(masks, dim=(2,3))
                    #    pseudo_seg = torch.amax(proto_masks, dim=1).unsqueeze(1).sigmoid()
                    #    pseudo_seg = 0.5 * pseudo_seg + 0.5 * mask_scores_mean
                    #    pseudo_scores = (pseudo_seg * gt_bitmasks).reshape(len(gt_inds), -1) 
                    #    pseudo_scores_rank = torch.sort(pseudo_scores, descending=True)[0]
                    #    n_pos = torch.clamp(n_pos, min=0, max=pseudo_scores_rank.shape[1]-1).long()
                    #      
                    #    assert n_pos.max() < pseudo_scores_rank.shape[1]
                    #    thr = torch.gather(pseudo_scores_rank, dim=1, index=n_pos)
                    #    thr = scatter_mean(thr.squeeze(1), gt_inds)[gt_inds][:,None,None,None]
                    #    thr = torch.clamp(thr, min=0.7, max=0.8)
                    #    pseudo_seg_final = (pseudo_seg > thr) * box_gt_bitmasks.float()

                    #    warmup_factor_2 = min(self._iter / float(270000), 1)
                    #    weights = ((pseudo_seg > thr) | (pseudo_seg < 0.4)) * box_gt_bitmasks
                    #    loss_pseudo = (mask_focal_loss(mask_scores, pseudo_seg_final.detach(), weights)) * warmup_factor_2 * 0.3
                    #    losses["loss_pseudo"] = loss_pseudo

                    # update the prototypes
                    gt_classes = [x.gt_classes for x in gt_instances]
                    self.prototype_learning(mask_feats, proto_masks, masks, gt_classes, pred_instances.labels, pred_instances.im_inds)


                # mask for pseudo masks
                # pseudo weight
                pseudo_mask_weight = mask_flags.reshape(-1).float()
                # num pseudo masks
                if self.fix_reduction:
                    num_valid_masks = (mask_flags > 0).float().sum().clamp(min=1.0)
                else:
                    num_valid_masks = pseudo_mask_weight.sum() + 1e-6

                mask_affinity = torch.cat([x.gt_mask_affinity for x in gt_instances])
                mask_affinity = mask_affinity[gt_inds].to(dtype=mask_feats.dtype)
                if not self.mask_affinity_binary:
                    mask_affinity_weights = mask_affinity * box_gt_bitmasks.float()
                else:
                    mask_affinity_weights = (mask_affinity >= self.pseudo_mask_affinity_thresh).float() * box_gt_bitmasks.float()

                loss_mask_affinity = (pairwise_losses * mask_affinity_weights).sum([1, 2, 3]) / mask_affinity_weights.sum([1, 2, 3]).clamp(min=1.0)
                # NOTE: weight???
                loss_mask_affinity = (loss_mask_affinity * pseudo_mask_weight).sum() / num_valid_masks

                # pseudo losses
                pseudo_mask_losses = dice_coefficient(mask_scores, gt_bitmasks)

                # fix: from 1e-6 to 1
                reduced_num_masks = reduce_mean(num_valid_masks).item()
                get_event_storage().put_scalar("num_pseudo_masks", reduced_num_masks)
                # pseudo loss
                loss_mask = (pseudo_mask_losses * pseudo_mask_weight).sum() / num_valid_masks

                if self.with_warmup:
                    warmup_factor = self.mask_warmup_fn(self._iter)
                else:
                    warmup_factor = 1.0
                losses["loss_mask"] = loss_mask * self.mask_weight * warmup_factor

                # add avg loss here
                if self.with_avg_loss:
                    loss_avg = self.avg_loss(mask_scores, gt_bitmasks)
                    loss_avg = (loss_avg * pseudo_mask_weight).sum() / num_valid_masks
                    losses["loss_avg"] = loss_avg * self.avg_weight * warmup_factor

                losses["loss_mask_affinity"] = loss_mask_affinity * warmup_factor * self.pseudo_mask_affinity_weight

                # add prototype loss
                if self.with_prototype_loss and not self.using_resize:
                    teacher_ind_mask = gt_bitmasks
                    student_ind_mask = (mask_scores > 0.5).float()
                    teacher_feat = F.interpolate(teacher_mask_feats, size=teacher_ind_mask.shape[-2:], mode="bilinear", align_corners=False)
                    teacher_feat = teacher_feat[pred_instances.im_inds]
                    student_feat = F.interpolate(mask_feats, size=student_ind_mask.shape[-2:], mode="bilinear", align_corners=False)
                    student_feat = student_feat[pred_instances.im_inds]

                    avg_teacher_feat = (teacher_ind_mask * teacher_feat).sum(dim=(-2,-1)) / teacher_ind_mask.sum(dim=(-2,-1)).clamp(min=1.0)
                    avg_student_feat = (student_ind_mask * student_feat).sum(dim=(-2,-1)) / student_ind_mask.sum(dim=(-2,-1)).clamp(min=1.0)

                    loss_prototype = self.MSELoss(avg_teacher_feat, avg_student_feat)
                    losses["loss_prototype"] = loss_prototype * warmup_factor * 0.5

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )
    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation, only_color=False):
    assert images.dim() == 4
    assert images.size(0) == 1

    if only_color:
        color_inds = [1, 2]
    else:
        color_inds = [0, 1, 2]

    unfolded_images = unfold_wo_center(
        images[:, color_inds], kernel_size=kernel_size, dilation=dilation
    )
    # print(unfolded_images.shape, images.shape)
    diff = images[:, color_inds, None] - unfolded_images

    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights

def get_instance_masks_affinity(pseudo_masks, image_masks, kernel_size, dilation):
    # Kx1xHxW
    pseudo_masks = pseudo_masks.unsqueeze(1)
    # add center pool
    pseudo_masks_pool = F.avg_pool2d(pseudo_masks, kernel_size=3, padding=1, stride=1)
    pseudo_masks = (pseudo_masks_pool + pseudo_masks) / 2.0

    # Kx(1)xNxHxW, N=8
    unfolded_masks = unfold_wo_center(
        pseudo_masks, kernel_size=kernel_size, dilation=dilation
    )
    # Kx1xNxHxW
    similarity = unfolded_masks * pseudo_masks[:, :, None] + \
        (1 - unfolded_masks) * (1 - pseudo_masks[:, :, None])
    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]
    similarity = (unfolded_weights * similarity).squeeze(1)
    return similarity

def get_instance_masks_affinity_new(pseudo_masks, image_masks, kernel_size, dilation):
    # Kx1xHxW
    pseudo_masks = pseudo_masks.unsqueeze(1)

    # Kx(1)xNxHxW, N=8
    unfolded_masks = unfold_wo_center(
        pseudo_masks, kernel_size=kernel_size, dilation=dilation
    )
    # Kx1xNxHxW
    diff = pseudo_masks[:, :, None] - unfolded_masks

    similarity = torch.exp(-torch.norm(diff, dim=1) * 1)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]
    similarity = unfolded_weights * similarity
    return similarity

def resize_images(images, image_size):
    max_size, min_size = image_size
    original_sizes = []
    resized_images = []
    for idx, img in enumerate(images):
        h, w = img.shape[-2:]
        scale = min_size / min(h, w)
        if h < w:
            newh, neww = min_size, scale * w
        else:
            newh, neww = scale * h, min_size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        resized_images.append(
            F.interpolate(img.unsqueeze(0), (newh, neww),
                            mode='bilinear', align_corners=False).squeeze(0)
        )
        original_sizes.append((h, w))
    return resized_images, original_sizes


class CondInst(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.im_format = cfg.INPUT.FORMAT
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = DynamicMaskHead(cfg)
        self.mask_branch = MaskBranch(cfg, self.backbone.output_shape())

        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM

        # boxinst configs
        self.boxinst_enabled = True
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        
        # pairwise loss from boxinst
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH

        # resize images
        self.using_resize = cfg.MODEL.BOX_TEACHER.RESIZE_IMAGES
        self.resize_size = cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST
        self.return_float_masks = cfg.MODEL.BOX_TEACHER.RETURN_FLOAT_MASK

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module
        
        self.use_aug = cfg.INPUT.AUG_TYPE != "none"

        self.filter_iou_thr = cfg.MODEL.BOX_TEACHER.IOU_THR
        
        # teacher mask threshold
        self.teacher_mask_threshold = cfg.MODEL.BOX_TEACHER.TEACHER_MASK_THRESHOLD

        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        # self.exist_classses = torch.tensor([0, ], dtype=torch.long).to(self.device)

        # copy paste
        self.copypaste_on = cfg.MODEL.COPYPASTE_ON
        self.copypaste = BoxSegCopyPaste(cfg.MODEL.COPYPASTE_TYPE, self.device)

        self.mask_affinity_binary = cfg.MODEL.BOX_TEACHER.MASK_AFFINITY_BINARY
        self.teacher_with_nms = cfg.MODEL.BOX_TEACHER.TEACHER_WITH_NMS
        self.use_vfl = cfg.MODEL.USE_VFL
        self.dynamic_mask_thr = cfg.MODEL.BOX_TEACHER.DYNAMIC_MASK_THRESH

    @torch.no_grad()
    def forward_teacher(self, batched_inputs):
        #from ipdb import set_trace; set_trace()
        original_images = [x["image"].to(self.device) for x in batched_inputs]
        images_norm = [self.normalizer(x) for x in original_images]
        if self.using_resize:
        # resize images to (maxsize, minsize)
            images_norm, image_sizes = resize_images(images_norm, self.resize_size)

        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)
        if not self.using_resize:
            image_sizes = images_norm.image_sizes

        features = self.backbone(images_norm.tensor)

        gt_instances = None
        mask_feats, _ = self.mask_branch(features, gt_instances)

        proposals, _ = self.proposal_generator.forward_test(
            images_norm, features, gt_instances, self.controller, teacher_with_nms=self.teacher_with_nms)
        if not self.teacher_with_nms:
            new_proposal_list = []
            for im_id, (input_per_image, proposal, image_size) in enumerate(zip(batched_inputs, proposals, image_sizes)):
                scale_x, scale_y = proposal.image_size[1]/image_size[1], proposal.image_size[0]/image_size[0]
                gt_labels = input_per_image['instances'].gt_classes.clone().to(self.device)
                gt_bboxes = input_per_image['instances'].gt_boxes.clone().to(self.device)
                gt_bboxes.scale(scale_x, scale_y)
                gt_bboxes.clip(proposal.image_size)
                pred_bboxes = proposal.pred_boxes
                pred_labels = proposal.pred_classes
                pred_scores = proposal.scores

                ious = pairwise_iou(gt_bboxes, pred_bboxes)

                # filter by ious with gt, get top 10 each gt
                topk = min(20, len(pred_bboxes))
                topk_ious, topk_inds = torch.topk(ious, topk, dim=-1)
                selected_ious = (topk_ious>0.5) & (pred_labels[topk_inds]==gt_labels[:,None]) & (pred_scores[topk_inds]>0.1)
                new_proposal = proposal[topk_inds[selected_ious]]
                nums_per_gt = selected_ious.int().sum(dim=-1)
                new_proposal.nums_per_gt = nums_per_gt.repeat(len(topk_inds[selected_ious]), 1)
                new_proposal_list.append(new_proposal)
            proposals = new_proposal_list

        padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
        processed_results = []
        # if not self.using_resize:
        #     image_sizes = images_norm.image_sizes
        for im_id, (input_per_image, proposal, image_size) in enumerate(zip(batched_inputs, proposals, image_sizes)):

            height = image_size[0]
            width = image_size[1]
            instances_per_im = self._forward_mask_heads_test(
                [proposal], mask_feats[im_id].unsqueeze(0))

            teacher_threshold = self.teacher_mask_threshold
            if self.return_float_masks:
                teacher_threshold = -1.0
            
            instances_per_im = self.postprocess(
                instances_per_im, height, width,
                padded_im_h, padded_im_w, teacher_threshold
            )
            processed_results.append({
                "instances": instances_per_im,
                "teacher_mask_feat": mask_feats[im_id],
            })

        return processed_results

    def forward(self, batched_inputs):
        #from ipdb import set_trace; set_trace()
        if self.training and self.use_aug:
            original_images = [x["image"].to(self.device) for x in batched_inputs]
            aug_images = [x["aug_image"].to(self.device) for x in batched_inputs]
            # auxiliary augmentation, e.g., random erasing
            #images_norm = [self.normalizer(x) for x in aug_images]
            # images_norm = [self.aux_aug(self.normalizer(x)) for x in aug_images]

            if self.copypaste_on:
                result_images, result_instances = [], []
                for i, (img, x) in enumerate(zip(aug_images, batched_inputs)):
                    result_img, result_instance = self.copypaste.transform((img, x["instances"]), self.mask_head._iter)
                    result_images.append(result_img)
                    result_instances.append(result_instance)
                aug_images = result_images
                for i, x in enumerate(batched_inputs):
                    x["instances"] = result_instances[i]
        else:
            original_images = [x["image"].to(self.device) for x in batched_inputs]
            aug_images = original_images

        images_norm = [self.normalizer(x) for x in aug_images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

        features = self.backbone(images_norm.tensor)

        teacher_mask_feats = None
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # add pseudo masks to gt
            self.add_pseudo_bitmasks(gt_instances, images_norm.tensor.size(-2),
                                     images_norm.tensor.size(-1), images_norm.image_sizes)

            original_image_masks = [torch.ones_like(
                x[0], dtype=torch.float32) for x in original_images]

            # mask out the bottom area where the COCO dataset probably has wrong annotations
            for i in range(len(original_image_masks)):
                im_h = batched_inputs[i]["height"]
                pixels_removed = int(
                    self.bottom_pixels_removed
                    * float(original_images[i].size(1)) / float(im_h)
                )
                if pixels_removed > 0:
                    original_image_masks[i][-pixels_removed:, :] = 0

            original_images = ImageList.from_tensors(
                original_images, self.backbone.size_divisibility)
            original_image_masks = ImageList.from_tensors(
                original_image_masks, self.backbone.size_divisibility, pad_value=0.0
            )
            self.add_bitmasks_from_boxes(
                gt_instances, original_images.tensor, original_image_masks.tensor,
                original_images.tensor.size(-2), original_images.tensor.size(-1),
            )
            # constraint mask with boxs
            self.constrain_mask_with_box(gt_instances)
            # compute pseudo pairwise affnity
            self.pseudo_mask_pairwise_affinity(gt_instances, original_image_masks.tensor)
            # collect teacher mask_feats
            teacher_mask_feats = torch.stack([x["teacher_mask_feat"] for x in batched_inputs])
        else:
            gt_instances = None

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, self.controller, self.use_vfl
        )

        if self.training:
            mask_losses = self._forward_mask_heads_train(proposals, mask_feats, gt_instances, teacher_mask_feats)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(mask_losses)
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)

            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                if instances_per_im.has("pred_masks"):
                    instances_per_im.pred_masks = instances_per_im.pred_masks > self.teacher_mask_threshold
                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances, teacher_mask_feats):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)
            # print([x.image_size for x in kept_instances])
            pred_instances = Instances.cat(kept_instances)

        pred_instances.mask_head_params = pred_instances.top_feats

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances, teacher_mask_feats
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        # print([x.image_size for x in proposals])
        # from IPython import embed; embed()
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat
        if self.training:
            pred_instances_w_masks = self.mask_head.forward_test(
                mask_feats, self.mask_branch.out_stride, pred_instances
            )
        else:
            pred_instances_w_masks = self.mask_head(
                mask_feats, self.mask_branch.out_stride, pred_instances
            )

        return pred_instances_w_masks

    def add_pseudo_bitmasks(self, instances, im_h, im_w, ori_sizes):
        for per_im_gt_inst, ori_size in zip(instances, ori_sizes):
            start = int(self.mask_out_stride // 2)
            if not per_im_gt_inst.has("gt_bitmasks"):
                gt_masks_flags = per_im_gt_inst.gt_masks_flags
                if gt_masks_flags.size(0) > 0:
                    bitmasks_full = torch.zeros(
                        (gt_masks_flags.size(0), im_h, im_w)).to(self.device)
                    # if self.add_full_masks:
                    #     per_im_gt_inst.gt_bitmasks_full = bitmasks_full
                    bitmasks = bitmasks_full[:, start::self.mask_out_stride,
                                             start::self.mask_out_stride]
                    per_im_gt_inst.gt_bitmasks = bitmasks
                continue

            bitmasks = per_im_gt_inst.get("gt_bitmasks")

            h, w = bitmasks.size()[1:]
            # print(h, w, ", image: ", im_h, im_w, ori_size)
            # pad to new size
            bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
            bitmasks = bitmasks_full[:, start::self.mask_out_stride,
                                     start::self.mask_out_stride]
            per_im_gt_inst.gt_bitmasks = bitmasks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else:  # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride,
                                         start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def constrain_mask_with_box(self, instances):
        for per_im_gt_inst in instances:
            box_bitmasks = per_im_gt_inst.boxinst_gt_bitmasks
            gt_bitmasks = per_im_gt_inst.gt_bitmasks.float()
            per_im_gt_inst.gt_bitmasks = gt_bitmasks * box_bitmasks

    def pseudo_mask_pairwise_affinity(self, instances, image_masks):
        stride = self.mask_out_stride
        start = int(stride // 2)
        image_masks = image_masks[:, start::stride, start::stride]
        for im_i, per_im_gt_inst in enumerate(instances):
            if self.mask_affinity_binary:
                valid_gt_bitmasks = (per_im_gt_inst.gt_bitmasks > self.teacher_mask_threshold).float()
                mask_affinity = get_instance_masks_affinity(
                    valid_gt_bitmasks, image_masks[im_i],
                    self.pairwise_size, self.pairwise_dilation)
            else:
                valid_gt_bitmasks = per_im_gt_inst.gt_bitmasks
                mask_affinity = get_instance_masks_affinity_new(
                    valid_gt_bitmasks, image_masks[im_i],
                    self.pairwise_size, self.pairwise_dilation)
            per_im_gt_inst.gt_mask_affinity = mask_affinity
            # binary masks
            per_im_gt_inst.gt_bitmasks = per_im_gt_inst.gt_bitmasks > self.teacher_mask_threshold

    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
        stride = self.mask_out_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )
        # RGB to BGR
        downsampled_images = downsampled_images[:, [2, 1, 0]]

        image_masks = image_masks[:, start::stride, start::stride]
        # float to byte to float
        downsampled_images_lab = rgb2lab(downsampled_images.byte().float())
        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = downsampled_images_lab[im_i][None]
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],
                self.pairwise_size, self.pairwise_dilation,
            )
            per_im_boxes = per_im_gt_inst.gt_boxes.tensor

            bitmasks_full = torch.zeros((len(per_im_boxes), im_h, im_w)).to(self.device).float()
            for idx, per_box in enumerate(per_im_boxes):
                bitmasks_full[idx, int(per_box[1]):int(per_box[3] + 1),
                              int(per_box[0]):int(per_box[2] + 1)] = 1.0

            bit_masks = bitmasks_full[:, start::stride, start::stride]
            per_im_gt_inst.boxinst_gt_bitmasks = bit_masks
            per_im_gt_inst.boxinst_gt_bitmasks_full = bitmasks_full
            per_im_gt_inst.boxinst_image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst))
            ], dim=0)

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.50):

        scale_x, scale_y = (
            output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            
            pred_scores = results.scores

            pred_masks_binary = (pred_global_masks > mask_threshold).float()

            mask_score = (pred_global_masks * pred_masks_binary).sum([1,2]) / \
                (pred_masks_binary.sum([1,2]) + 1e-6)
            pred_scores_sqrt = pred_scores.sqrt()
            results.mask_score =  (pred_scores_sqrt * mask_score).sqrt()
            results.scores = (pred_scores * mask_score).sqrt()
            results.pred_scores = pred_scores
            #results.mask_score = pred_scores_sqrt
            #results.scores = pred_scores
            if mask_threshold == -1:
                # return float masks
                results.pred_masks = pred_global_masks
            else:
                #results.pred_masks = pred_masks_binary.bool()
                results.pred_masks = pred_global_masks
            
        return results
