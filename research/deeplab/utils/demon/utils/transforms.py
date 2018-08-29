import numpy as np
import cv2
import pdb


def image_transform(img_in, method='norm', center_crop=None):
    """ transform image to network input
    """
    img = img_in.copy()
    if 'norm' == method:
        img = np.float32(img) / 255.0
        img -= 0.5

    if center_crop:
        height, width = img.shape[:2]
        up = height / 2 - center_crop / 2
        left = width / 2 - center_crop / 2
        img = img[up:up+center_crop, left:left+center_crop]

    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)  # (1, c, h, w)
    return img


def depth_transform(depth_in, mean_depth=None, forward=True):
    depth = depth_in.copy()
    if not (mean_depth is None):
        depth -= mean_depth
    else:
        mean_depth = 0.0

    # depth[depth <= -1 * mean_depth] = -1.0
    depth = np.float32(depth) / 30.0

    if forward:
        depth = np.float32(depth[None, None, :, :])  # (1, 1, h, w)
    else:
        depth = np.float32(depth[:, 0, :, :])

    return depth


def mask_transform(mask, forward=True):
    return np.float32(mask[None, None, :, :]) if forward \
            else np.float32(mask[:, 0, :, :])


def pose_transform(pose_in, forward=True):
    """ rot 0-3, trans 4-6
    """
    pose = pose_in.copy()
    pose = np.float32(pose)
    sign = (np.float32(pose[:, 3:] >= 0.0) - 0.5) * 2
    if forward:
        pose[:, :3] = pose[:, :3] / (2 * np.pi) - 0.5
        pose[:, 3:] = np.log(np.abs(pose[:, 3:]) + 1.0) * sign

    else:
        pose[:, :3] = (np.float32(pose[:, :3]) + 0.5) * 2 * np.pi
        pose[:, 3:] = (np.exp(np.abs(pose[:, 3:])) - 1.0) * sign

    return np.float32(pose)


def crop_transform(crop, size):
    """ t l b r
    """
    crop = np.float32(crop)
    return np.hstack([crop[[0, 2]] / size[0],
        crop[[1, 3]] / size[1]])[None, :]


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    pose_test = np.float32([[1, 2, 3, -1, 2, -30000]])
    logging.info("orgin %s" % (pose_test))
    pose_1 = pose_transform(pose_test)
    logging.info("forward %s" % (pose_1))
    pose_2 = pose_transform(pose_1, False)
    logging.info('backward %s' % (pose_2))

