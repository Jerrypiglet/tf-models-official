import numpy as np
import utils.utils as uts
from collections import OrderedDict
import pdb


def eval_instance_depth(gt_instance, pred_instance, num_samples):
    """ For all image evaluate the depth precision and average IOU of each instance
        Instance ID must match between gt and prediction
    """

    num_images = len(pred_instance)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    IOUs = np.zeros(num_samples, np.float32)
    i = 0

    for img_id in range(num_images):
        print("eval %s" % img_id)
        depth_gt = gt_instance[img_id][1]
        instance_gt = gt_instance[img_id][0]
        depth_pred = pred_instance[img_id][1]
        instance_pred = pred_instance[img_id][0]

        ids = np.unique(instance_gt[instance_gt > 0])
        all_mask = np.logical_and(depth_gt > 0, depth_pred > 0)
        # uts.plot_images({'mask': instance_gt})

        for idx in ids:
            gt_seg = instance_gt == idx
            pred_seg = instance_pred == idx
            IOUs[i] = IOU(gt_seg, pred_seg)
            # uts.plot_images({'gt': gt_seg, 'pred': pred_seg})
            if IOUs[i] == 0:
                continue

            mask = np.logical_and(
                    all_mask, np.logical_and(gt_seg, pred_seg))
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
                    compute_errors(depth_gt[mask], depth_pred[mask])
            i = i + 1

    num_samples = np.float32(num_samples)
    cond1 = np.logical_and(IOUs > 0.5, a1 > 0.6)
    cond2 = np.logical_and(IOUs > 0.7, a1 > 0.8)
    cond3 = np.logical_and(IOUs > 0.85, a1 > 0.9)

    delta1 = np.sum(cond1) / num_samples
    delta2 = np.sum(cond2) / num_samples
    delta3 = np.sum(cond3) / num_samples

    valid = ~np.isnan(abs_rel)
    res = [abs_rel[valid].mean(), sq_rel[valid].mean(), rms[valid].mean(), \
           log_rms[valid].mean(), a1[valid].mean(), a2[valid].mean(), a3[valid].mean()]
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(*res))

    print("{:>10}, {:>10}, {:>10}, {:>10}".format('mean_IOU', 'rc % 0.5', '% iou > 0.75', '% iou > 0.9'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}".format(IOUs.mean(), delta1, delta2, delta3))


def IOU(mask1, mask2):
    """ two logical inputs
    """

    inter = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    if np.sum(inter) == 0:
        return 0.

    return np.float32(np.sum(inter)) / np.float32(np.sum(union))


def compute_errors(gt, pred, thresholds=None):
    if len(gt) == 0:
        res = [np.nan for i in range(7)]
        return res

    assert len(gt) == len(pred)
    if thresholds is None:
        thresholds = [1.25, 1.25 ** 2, 1.25 ** 3]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < thresholds[0]).mean()
    a2 = (thresh < thresholds[1]).mean()
    a3 = (thresh < thresholds[2]).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / (gt))
    sq_rel = np.mean(((gt - pred)**2) / (gt))

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_reward(mask, depth, pred_mask, pred_depth, thr):
    """compute compatibility
       mask: proposed mask
       depth: estimated depth
       pred_mask: rendered mask
       pred_depth: rendered depth
       thr: occlusion threshold
    """

    def rm_occ_part(depth_stereo, depth_render, mask_render):

        occ_mask = (depth_render - depth_stereo) < thr
        occ_mask = np.logical_or(occ_mask, depth_stereo <= 0)
        mask_nocc = np.logical_and(occ_mask, mask_render)
        # uts.plot_images({'non_occ_mask': occ_mask, 'valid_mask': mask_nocc})
        return mask_nocc

    min_reward = 1e-10
    non_occ_mask = rm_occ_part(depth, pred_depth, pred_mask)
    iou = IOU(mask, non_occ_mask)
    # inter_mask = np.logical_and(pred_depth > 0, \
    #              np.logical_and(depth > 0, \
    #              np.logical_and(mask, pred_mask)))

    # the depth error at rendered depth
    inter_mask = np.logical_and(pred_depth > 0, \
                 np.logical_and(depth > 0, pred_mask > 0),)

    if np.sum(inter_mask) == 0:
        return min_reward, min_reward

    gt = depth[inter_mask]
    pred = pred_depth[inter_mask]
    thresh = np.maximum((gt / pred), (pred / gt))
    # delta_1 = np.exp(1.0 - thresh.mean())
    delta = (thresh < 1.05).mean()
    alpha = 0.3
    reward = 10.0 * ((1.0 - alpha) * IOU + alpha * delta)

    return max(iou, 1e-10), max(delta, 1e-10), reward


def mcmc_reward_v2(mask, disp, layer_disp, inst_disp, pred_mask, pred_disp):
    """ notice here disp must be a full depth with bkground depth.
        Otherwise there will be bad
    """
    # eval within given mask
    cur_mask = np.logical_and(mask > 0, disp > 0)
    cur_disp = disp[cur_mask]

    pred_disp[pred_disp <= 0] = -1 * np.inf
    cur_render_disp = pred_disp[cur_mask]

    d_num = cur_render_disp.size
    d_err = np.minimum(np.abs(cur_disp - cur_render_disp), 3) / 3.0
    err = 10. * np.sum(d_err) / d_num

    # eval within rendered mask
    valid_render_disp = np.logical_and(pred_mask, disp > 0)
    # handle instance area
    appear_mask = layer_disp <= inst_disp
    # handle background area
    non_occ_mask = np.logical_and(appear_mask, (disp - 3.) < pred_disp)
    non_occ_mask = np.logical_and(non_occ_mask, valid_render_disp)

    inter_mask = np.logical_and(non_occ_mask, cur_mask)
    if (np.float32(np.sum(inter_mask)) / np.float32(np.sum(cur_mask))) < 0.2:
        err = 1e4

    # no occlusion happend outside mask region
    valid_none_mask = np.logical_and(disp > 0, mask == 0)
    occ_err = np.logical_and((disp + 5.) < pred_disp, valid_none_mask)
    err = err + 0.005 * np.sum(occ_err)
    # uts.plot_images({'disp': disp, 'layer_disp': layer_disp, 'appear_mask': appear_mask, \
    #         'pred_mask': pred_mask, 'non_occ_mask': non_occ_mask,
    #         'occ_mask': occ_err}, layout=[3, 3])

    # measure the depth error at inter section between mask & pred_mask
    iou = IOU(mask, non_occ_mask)
    if np.sum(inter_mask) > 0:
        gt = disp[inter_mask]
        pred = pred_disp[inter_mask]
        thresh = np.maximum((gt / pred), (pred / gt))
        delta = (thresh < 1.05).mean()
    else:
        delta = 1e-10

    return max(iou, 1e-10), delta, np.exp(-1 *(err / 10 + 1.0 - iou))


def mcmc_reward(mask, disp, pred_mask, pred_disp):
    """ notice here disp must be a full depth with bkground depth.
        Otherwise there will be bad
    """
    # eval within given mask
    cur_mask = np.logical_and(mask > 0, disp > 0)
    cur_disp = disp[cur_mask]

    pred_disp[pred_disp <= 0] = -1 * np.inf
    cur_render_disp = pred_disp[cur_mask]

    d_num = cur_render_disp.size
    d_err = np.minimum(np.abs(cur_disp - cur_render_disp), 3) / 3.0
    err = 10. * np.sum(d_err) / d_num

    # eval within rendered mask
    valid_render_disp = np.logical_and(pred_mask, disp > 0)
    non_occ_mask = np.logical_and((disp - 3.) < pred_disp,
                                     valid_render_disp)
    # uts.plot_images({'mask': disp}) #, 'pred_mask': pred_mask, 'given_mask': mask})
    inter_mask = np.logical_and(non_occ_mask, mask)
    if (np.float32(np.sum(inter_mask)) / np.float32(np.sum(mask))) < 0.2:
        err = 1e4

    occ_err = np.logical_and((disp + 8.) < pred_disp, valid_render_disp)
    err = err + 0.005 * np.sum(occ_err)

    # measure the depth error at inter section between mask & pred_mask
    iou = IOU(mask, non_occ_mask)
    if np.sum(inter_mask) > 0:
        gt = disp[inter_mask]
        pred = pred_disp[inter_mask]
        thresh = np.maximum((gt / pred), (pred / gt))
        delta = (thresh < 1.05).mean()
    else:
        delta = 1e-10

    return max(iou, 1e-10), delta, np.exp(-1 * err / 10)


def merge_inst(res,
               inst_id,
               total_mask,
               total_depth,
               boxes=None,
               thresh=0.3):
    """ merge the prediction of each car instance to a full image
    """

    render_depth = res['depth'].copy()
    render_depth[render_depth <= 0] = np.inf
    depth_arr = np.concatenate([render_depth[None, :, :],
        total_depth[None, :, :]], axis=0)
    idx = np.argmin(depth_arr, axis=0)
    visible = np.sum(idx == 0) / (np.sum(res['mask']) + np.spacing(1))

    if visible < thresh:
        return total_mask, total_depth, boxes, False

    total_depth = np.amin(depth_arr, axis=0)
    total_mask[idx == 0] = inst_id
    if not (boxes is None):
        assert isinstance(boxes, list)
        boxes.append(uts.get_mask_bounding_box(res['mask']))

    return total_mask, total_depth, boxes, True



def vis_res(total_mask, total_depth, boxes, state, colors):
    # current fitting images stereo depth

    image = state['image']
    image = image[:, :, ::-1]
    depth_cnn = state['depth']
    masks = state['masks']
    gt_mask = uts.mask2label(masks)

    alpha = 0.7
    label_c = uts.label2color(np.uint32(total_mask), colors, [255,255,255])
    image = np.uint8(alpha * image + (1-alpha) * label_c);
    uts.drawboxes(image, boxes)

    uts.plot_images(OrderedDict([('image', image),
                                 ('pred_instance', total_mask),
                                 ('pred_depth', total_depth),
                                 ('gt_instance', gt_mask),
                                 ('depth_cnn', depth_cnn)]),
                                 layout=[2, 3])


def save_res(total_mask, total_depth, boxes):
    pass


def get_max_reward(reward, inspect,
        max_values=None):
    max_reward = np.max(reward['reward'])
    idx = np.argmax(reward['reward'])
    max_IOU = inspect['IOU'][idx]
    max_delta = inspect['delta'][idx]

    if max_values == None:
        return max_reward, max_IOU, max_delta
    else:
        return [max(x, max_values[i]) for i, x in enumerate([max_reward, max_IOU, max_delta])]

