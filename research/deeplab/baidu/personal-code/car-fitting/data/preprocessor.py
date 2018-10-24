import numpy as np
import cv2
import pdb
import utils.utils as uts
import utils.utils_3d as uts_3d


def get_floor_mask(depth_in, intr, floor_height=1.2, rescale=1.0):
    """results remove points too far away from its median points
    """
    depth = np.float32(depth_in.copy())
    intrinsic = np.float32(intr.copy())
    h, w = depth.shape
    intrinsic[[0, 2]] /= np.float32(w)
    intrinsic[[1, 3]] /= np.float32(h)
    xyz = uts_3d.depth2xyz(depth, intrinsic, homo=False, flat=False)

    if rescale < 1.0:
        depth = cv2.resize(depth, (int(w * rescale), int(h * rescale)),
                interpolation=cv2.INTER_LINEAR)
    normal = uts_3d.depth2normal(depth, intrinsic)
    if rescale < 1.0:
        normal = cv2.resize(normal, (w, h), interpolation=cv2.INTER_LINEAR)

    is_floor = np.abs(normal[:, :, 1]) > 0.75
    is_low_height = np.logical_and(is_floor, xyz[:, :, 1] > floor_height)

    is_floor = np.logical_and(is_floor, is_low_height)

    # uts.plot_images({'normal': (normal + 1.0) / 2.0, 'xyz': xyz[:, :, 1],
    #     'depth': depth, 'xyz_2': xyz[:, :, 2]})
    return is_floor


def denoise_mask(masks, depth, floor_mask, thresh=3.5):
    """results remove points too far away from its median points
    """

    masks_out = []
    non_floor_mask = np.logical_not(floor_mask)
    for i, mask in enumerate(masks):
        pixs = np.where(mask)
        pixs = np.vstack(pixs)
        depths= depth[pixs[0, :], pixs[1, :]]
        median_idx = np.argsort(depths)[len(depths) // 2]
        pix = pixs[:, median_idx]
        med_z = depth[pix[0], pix[1]]
        valid = np.abs(depths - med_z) <= thresh
        pixs = pixs[:, valid]
        mask = np.zeros(mask.shape, dtype=np.bool)
        mask[pixs[0, :], pixs[1, :]] = True
        masks_out.append(np.logical_and(mask, non_floor_mask))

    return masks_out



if __name__ == '__main__':

    import data.kitti as kitti
    import cv2
    params = kitti.set_params_disp(disp='psm')

    for i in range(194):
        image_name = '%06d_10' % i
        print image_name
        masks = kitti.get_instance_masks(params, image_name, 'car')
        if len(masks) == 0:
            continue
        depth = kitti.depth_read(params['cnn_disp_path'] \
                    + image_name + '.png')
        intr_file = params['calib_path'] + image_name[:-3] + '.txt'
        intrinsic = kitti.load_intrinsic(intr_file)

        floor_mask = get_floor_mask(depth, intrinsic, floor_height=1.5)
        masks_out = denoise_mask(masks, depth, floor_mask)
        car_inst_label = uts.mask2label(masks_out)
        # uts.plot_images({'mask': car_inst_label, 'floor_mask':floor_mask})
        cv2.imwrite(params['car_inst_path'] + image_name + '.png',
                    np.uint8(car_inst_label))
        # uts.plot_images({'mask': masks[0],
        #                  'depth': depth,
        #                  'mask_out': masks_out[0]})


