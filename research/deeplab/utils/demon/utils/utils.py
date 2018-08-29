import os
import cv2
import numpy as np
from PIL import Image

import matplotlib as mpl
# mpl.use('Agg')
# mpl.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import cm
import time
import argparse
import pdb


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def exists(file):
    """
    Check whether a file is existing.
    """
    return os.path.exists(file)


def get_partition_list(lst, n, i):
    return lst[i::n]


def rm_b_from_a(la, lb):
    """
    Remove any elements in b from a
    img_path: image path.
    is_color: is color image or not.
    """
    i = 0
    while i < len(la):
        name = la[i]
        if name in lb:
            la.remove(name)
        else:
            i += 1
    return la


def load_image(img_path, is_color=True):
    """
    Load image and return.
    img_path: image path.
    is_color: is color image or not.
    """
    img = Image.open(img_path)
    img.load()

    return img


def exclude_pattern(f):
    """
    Return whether f is in the exlucde pattern.
    Exclude the files that starts with . or ends with ~.
    """
    return f.startswith(".") or f.endswith("~")


def list_images(path, exts=set(["jpg", "png", "bmp", "jpeg"])):
    """
    Return a list of images in path.
    path: the base directory to search over.
    exts: the extensions of the images to find.
    """
    return [os.path.join(path, d) for d in  os.listdir(path) \
            if os.path.isfile(os.path.join(path, d)) and not exclude_pattern(d)\
            and os.path.splitext(d)[-1][1:] in exts]


def save_video_to_images(in_path, out_path, max_frame=100000):
    print("saving videos to frames at {}".format(out_path))
    cap = cv2.VideoCapture(in_path)
    frame_id = 0
    mkdir_if_need(out_path)

    cv2.namedWindow("video")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        filename = out_path + 'frame {}.jpg'.format(str(frame_id))
        print(filename)
        cv2.imshow('video',frame)
        cv2.imwrite(filename, frame)
        frame_id += 1
        if frame_id > max_frame:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("finished")


def color2label_slow(label_color, color_map):
    height, width = label_color.shape[0:2]
    label = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            rgb = tuple(label_color[i, j, :])
            try:
                label[i, j] = color_map[rgb]
            except:
                continue

    return label


def color2label(label_color, color_map):
    # default bkg 255
    label_color = np.int32(label_color)
    height, width = label_color.shape[0:2]
    label = label_color[:, :, 0] * (255 ** 2) + \
            label_color[:, :, 1] * 255 + \
            label_color[:, :, 2]

    label_id = np.unique(label)
    for rgb, i in color_map.items():
        cur_num = rgb[0] * (255 ** 2) + rgb[1] * 255 + rgb[2]
        if cur_num in label_id:
            mask = (label - cur_num) != 0
            label = label * mask  + i * (1 - mask)

    return label


def one_hot(label_map, class_num):
    assert np.ndim(label_map) == 2
    height, width = label_map.shape
    label_one_hot = np.zeros((height * width, class_num))
    label_map = label_map.flatten()
    label_one_hot[range(height * width), label_map] = 1
    label_one_hot = label_one_hot.reshape((height, width, class_num))
    return label_one_hot


def mask2label(masks, idx=None):
    """ masks must be h, w, c or a list of masks
    """
    assert isinstance(masks, list)
    if idx is None:
        idx = np.arange(len(masks)) + 1
    gt_mask = np.zeros(masks[0].shape, dtype=np.uint32)
    for i, mask in enumerate(masks):
        gt_mask[mask > 0] = idx[i]

    return gt_mask


def label2mask(label):
    """ convert a label map to a list of masks
        0 is background
    """

    assert len(label.shape) == 2
    masks = []
    label_ids = np.unique(label)
    for i in label_ids:
        if i == 0:
            continue
        masks.append(label == i)

    return masks


def resize_and_pad(image, size, interpolation, crop=None, get_pad_sz=None):
    """ resize to possible max image
    Input:
        crop: t l b r
        get_pad_sz: [t_pad, l_pad, b_pad, r_pad, scale] the coordinate of image
                inside the output padded image, and the relative scale

    """

    ndim = np.ndim(image)
    assert ndim >= 2

    h, w = image.shape[0], image.shape[1]
    scale = min(np.float32(size[0]) / h, np.float32(size[1]) / w)
    size_new = np.uint32(np.around(np.array([h, w]) * scale))
    crop_new = crop.copy() if not (crop is None) else None

    if size[0] == size_new[0]:
        h_range = np.uint32([0, size[0]])
        pad = np.floor((size[1] - size_new[1]) / 2.0)
        c_range = np.uint32([pad, min(pad + size_new[1], size[1])])
        if not (crop is None):
            s = (size[1] * h) / (size[0] * w)
            change = (crop[3] - crop[1]) * (s - 1.) / 2.
            crop_new[[1, 3]] = crop_new[[1, 3]] + \
                    np.array([-1 * change, change])

    else:
        pad = np.floor((size[0] - size_new[0]) / 2.0)
        h_range = np.uint32([pad, min(pad + size_new[0], size[0])])
        c_range = np.uint32([0, size[1]])
        if not (crop is None):
            s = (size[0] * w) / (size[1] * h)
            change = (crop[2] - crop[0]) * (s - 1.) / 2.
            crop_new[[0, 2]] = crop_new[[0, 2]] + \
                    np.array([-1 * change, change])

    if ndim == 2:
        image_out = np.zeros(size)
        image_out[h_range[0]:h_range[1], c_range[0]:c_range[1]] =  \
            cv2.resize(image, (size_new[1], size_new[0]), interpolation=interpolation)

    elif ndim > 2:
        image_out = np.zeros([size[0], size[1], image.shape[2]])
        image_out[h_range[0]:h_range[1], c_range[0]:c_range[1], :] =  \
            cv2.resize(image, (size_new[1], size_new[0]), interpolation=interpolation)

    if crop is None:
        output = image_out
    else:
        output = [image_out, crop_new]

    if get_pad_sz:
        output = [output] if not isinstance(output, list) else output
        output = output + [np.array([h_range[0], c_range[0], h_range[1],
            c_range[1], scale])]

    return output


def prob2label(label_prob):
    """Convert probability to a descrete label map
    """
    assert label_prob.ndim == 3
    return np.argmax(label_prob, axis=2)


def vec2img(inputs, height, width):
    """Convert a vector to image based on height and width
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
        height = [height]
        width = [width]

    for i in range(len(inputs)):
        inputs[i] = inputs[i].reshape((-1, height[i], width[i]))
        inputs[i] = inputs[i].transpose((1, 2, 0))
        inputs[i] = inputs[i].squeeze()

    return inputs if len(inputs) > 1 else inputs[0]


def prob2color(label_prob, color_map, bkg_color=[0,0,0]):
    height, width, dim = label_prob.shape

    color_map_mat = np.matrix([bkg_color] + color_map)
    label_prob_mat = np.matrix(label_prob.reshape((height * width, dim)))
    label_color = np.array(label_prob_mat * color_map_mat)
    label_color = label_color.reshape((height, width, -1))

    return np.uint8(label_color)


def label2color(label, color_map, bkg_color=[0, 0, 0]):
    height, width = label.shape[0:2]
    class_num = len(color_map) + 1
    label_one_hot = one_hot(label, class_num)
    label_color = prob2color(label_one_hot, color_map, bkg_color)

    return label_color


def read_depth(depth_name):
    bit_16 = np.array(load_image(depth_name)).astype(np.uint16)
    depth = np.bitwise_or(np.right_shift(bit_16, 3), np.left_shift(bit_16, 13))
    depth = np.float32(depth)/1000.0
    return depth


def transform(image, height, width):
    """
    image: a numpy image with shape (h x w x c)
    """
    image = cv2.resize(image, (width, height),
            interpolation=cv2.INTER_LINEAR)
    image = np.float32(image)
    image = image / 255.0 - 0.5
    image = image.transpose((2, 0, 1)).flatten()
    return image


def mkdir_if_need(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def read_file(filename):
    with open(filename, 'r') as f:
        names = [x.strip('\n') for x in f]
    return names


def show_grey(image):
    image = image.squeeze()
    assert len(image.shape) == 2
    plt.imshow(image, cmap='gray')


def show_grey_rev(image):
    assert len(image.shape.squeeze()) == 2
    plt.imshow(1.0 - image, cmap='gray')


def flow2color(flow):
    assert flow.shape[2] == 2
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3),
        dtype=np.float32)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)
    return hsv, rgb


def show_flow(flow):
    hsv, rgb = flow2color(flow)
    plt.imshow(np.uint8(rgb))


vis_func = {'gray': show_grey,
            'rev_gray': show_grey,
            'flow': show_flow,
            'color': plt.imshow,
            'cv_color': cv2.imshow}


def plot_images(images,
                layout=[2,2],
                fig_size=10,
                attr=None,
                save_fig=False,
                is_close=False,
                fig=None,
                fig_name='tmp.jpg'):

    is_show = True if fig is None else False
    if fig is None:
        fig = plt.figure(figsize=(10,5))

    pylab.rcParams['figure.figsize'] = fig_size, fig_size/2
    Keys = images.keys()
    attr_all = {}
    for iimg, name in enumerate(Keys):
        # not support the one dim data
        assert len(images[name].shape) >= 2

        if len(images[name].shape) == 2:
            attr_all[name] = 'color'
        else:
            if images[name].shape[2] == 2:
                attr_all[name] = 'flow'
            else:
                attr_all[name] = 'color'

    if attr:
        attr_all.update(attr)

    for iimg, name in enumerate(Keys):
        # print(name)
        s = plt.subplot(layout[0], layout[1], iimg+1)
        vis_func[attr_all[name]](images[name])

        s.set_xticklabels([])
        s.set_yticklabels([])
        s.set_title(name)
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')

    plt.tight_layout()

    if save_fig:
        pylab.savefig(fig_name)
    else:
        if is_show:
            plt.show()
        else:
            fig.canvas.draw()


def plot_figure(x, y, x_label='x', y_label='y', name='fig'):
    plt.plot(x, y,
        color=cm.jet(1.0),
        label=y_label)
    plt.title(name)
    plt.xlabel(x_label)
    plt.legend(loc='upper right')
    plt.show()


def plot_mcmc_log(log_file, metric_name=None, ylim=None):
    import re
    if not isinstance(metric_name, list):
        metric_name = [metric_name]
    VA_RE = {}
    for name in metric_name:
        if name in ['reward', 'IOU', 'delta']:
            VA_RE[name] = re.compile('.*?Ave-'+name+'\s([\d\.]+)')
        elif name in ['mean_xyz', 'median_xyz', 'mean_theta', 'median_theta']:
            VA_RE[name] = re.compile('.*?]\sValidation-' + name + '=([\d\.]+)')
    log = open(log_file).read()

    fig = plt.figure(figsize=(8, 6))
    plt.xlabel("Epoch")
    color = ['g', 'r', 'b', 'k', 'c']
    for i, name in enumerate(metric_name):
        log_num = [float(x) for x in VA_RE[name].findall(log)]
        idx = np.arange(len(log_num))
        plt.plot(idx, log_num, 'o', linestyle='-', color=color[i],
                 label=name )

    plt.legend(loc="best")
    plt.show()

    # INFO:root:Epoch[155] Validation-mean_xyz=0.476085
    # INFO:root:Epoch[155] Validation-median_xyz=0.412631
    # INFO:root:Epoch[155] Validation-mean_theta=39.221869
    # INFO:root:Epoch[155] Validation-median_theta=14.541846

    # log_tr = [float(x) for x in TR_RE.findall(log)]



def plot_mxnet_log(log_file, metric_name='accuracy', ylim=None, fig=None):
    import re
    TR_RE = re.compile('.*?]\sTrain-' + metric_name + '=([\d\.]+)')
    VA_RE = re.compile('.*?]\sValidation-' + metric_name + '=([\d\.]+)')

    log = open(log_file).read()
    log_tr = [float(x) for x in TR_RE.findall(log)]
    log_va = [float(x) for x in VA_RE.findall(log)]
    idx_tr = np.arange(len(log_tr))
    idx_va = np.arange(len(log_va))
    assert len(log_tr) > 0
    if ylim is None:
        ylim = [np.min(np.array(log_tr)), np.max(np.array(log_tr))]

    # log out maximum val results
    if len(log_va) > 0:
        max_idx = np.argmax(np.array(log_va))
        min_idx = np.argmin(np.array(log_va))
        print "maxval{} epoch{}".format(log_va[max_idx], max_idx)
        print "minval{} epoch{}".format(log_va[min_idx], min_idx)

    is_show = False if fig is None else True
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(idx_tr, log_tr, 'o', linestyle='-', color="r",
             label="Train " + metric_name )
    if len(log_va) > 0:
        plt.plot(idx_va, log_va, 'o', linestyle='-', color="b",
                 label="Validation " + metric_name)

    plt.legend(loc="best")
    plt.xticks(np.arange(min(idx_tr), max(idx_tr)+1, 5))
    y_step = (ylim[1] - ylim[0]) / 10
    plt.yticks(np.arange(ylim[0], ylim[1], y_step))
    plt.ylim(ylim)

    if not is_show:
        plt.show()
    else:
        fig.canvas.draw()


def plot_my_log(log_file, names, max_len=np.inf):

    def parse_line(line):
        pose = [line.find(name) for name in names]
        if np.all(np.array(pose) == -1):
            return None
        pose.append(len(line))
        idx = np.argsort(pose)
        res = np.zeros(len(names))
        res = [float(line[pose[idx[i]] + 2 + len(names[idx[i]]): pose[idx[i + 1]]]) for i in range(len(names))]
        num = len(names)
        res = np.array(res)[idx[:num]]
        return res

    log = [line for line in open(log_file)]
    res = []
    for i, line in enumerate(log):
        cur_res = parse_line(line)
        if cur_res is None:
            continue
        res.append(cur_res)
        if i > max_len:
            break

    res = np.array(res)
    line_styles = ['-', '.-']
    colors = ['r', 'g', 'b']
    idx = np.arange(res.shape[0])
    for i, name in enumerate(names):
        plt.plot(idx, np.log(res[:, i]), '.', linestyle='-', color=colors[i],
                 label="Train " + name)
    plt.legend(loc="best")
    plt.xticks(np.arange(min(idx), max(idx)+1, 100))
    plt.show()


color_system = ['red', 'tan', 'lime']
def plot_pose_map(pose_list, leftup, scale,
                is_angle=False,
                background=None,
                save_file=None):

    assert (not is_angle)
    assert isinstance(pose_list, dict)
    color = ['g', 'r', 'b']
    offset = 100
    fig = plt.figure(figsize=(60,50))
    if background:
        img = plt.imread(background)
        plt.imshow(img)

    for i, item in enumerate(pose_list.items()):
        key, pose = item
        x = (pose[:, 0] - leftup[0]) * scale
        y = (pose[:, 1] - leftup[1]) * scale
        direct = pose[:, 3:] * 100
        Q = plt.quiver(x, y, direct[:, 0], direct[:, 1],
                scale_units='xy',
                angles='xy',
                color=color[i],
                width=0.001,
                scale=1,
                alpha=0.5)
        plt.quiverkey(Q, 5600, 6000+i*offset, 100, label =key, labelpos='W',
                coordinates='data')

    if save_file:
        fig.savefig(save_file)
    else:
        plt.show()


def plot_histogram(values, n_bins=20, labels=None):
    assert isinstance(values, list)
    ndim = len(values)
    cmap = plt.get_cmap('jet')
    color_values = [0.5, 0.25, 0.8]
    colors = [cmap(val) for val in color_values]
    if labels is None:
        labels = [str(i) for i in range(ndim)]

    for i, (value, label) in enumerate(zip(values, labels)):
        N, bins, patches = plt.hist(value, n_bins, alpha=0.5,
                density=True, histtype='bar', color=colors[i],
                label=label)
    plt.legend()

    plt.ylabel('counts')
    plt.show()


def frame_to_video(image_path,
                   label_path,
                   frame_list,
                   label_ext='',
                   is_color=False,
                   color_map=None,
                   sz=None,
                   fps=10,
                   alpha=0.5,
                   video_name='video.avi'):
    """Combine frames to video
    """
    # import pdb
    # pdb.set_trace()

    if sz is None:
        label = cv2.imread("%s%s.png" % (label_path, frame_list[0]))
        sz = label.shape

    fourcc = cv2.cv.CV_FOURCC(*'DIV3')
    video = cv2.VideoWriter(video_name, fourcc, fps, (sz[1], sz[0]))
    for i, image_name in enumerate(frame_list):
        print "compress %04d" % i
        image = cv2.resize(cv2.imread("%s%s.jpg" % (image_path, image_name),
            cv2.IMREAD_UNCHANGED),
            (sz[1], sz[0]))
        label_name = image_name + label_ext
        label = cv2.resize(cv2.imread("%s%s.png" % (label_path, label_name),
            cv2.IMREAD_UNCHANGED),
            (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)

        if not is_color:
            bkg = [255, 255, 255]
            label[label > len(color_map)] = 0
            label = label2color(label, color_map, bkg)
            label = label[:, :, ::-1]

        frame = np.uint8(image * alpha + label * (1 - alpha))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def images_to_video(image_path, frame_list, sz=None,
        fps=10, video_name='video.avi'):
    """Combine frames to video
    """
    if sz is None:
        image = cv2.imread("%s%s" % (image_path, frame_list[0]),
                           cv2.IMREAD_UNCHANGED)
        sz = image.shape[:2]

    fourcc = cv2.cv.CV_FOURCC(*'DIV3')
    video = cv2.VideoWriter(video_name, fourcc, fps, (sz[1], sz[0]))
    for i, image_name in enumerate(frame_list):
        print "compress %04d" % i
        image = cv2.imread("%s%s" % (image_path, image_name),
                           cv2.IMREAD_UNCHANGED)

        if sz is not None:
            image = cv2.resize(image, (sz[1], sz[0]),
                    interpolation=cv2.INTER_NEAREST)

        frame = np.uint8(image)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def get_mask_center(mask):
    assert len(mask.shape) == 2
    idx = np.where(mask)
    pixs = np.vstack(idx)
    center = np.mean(pixs, axis=1)

    return center


def get_mask_bounding_box(mask, context=1.0):
    """ t, l, b, r
    """
    assert len(mask.shape) == 2

    h, w = mask.shape
    pixs = np.where(mask > 0)
    box = np.zeros(4)
    box[[0, 2]] = [max(0, np.min(pixs[0]) - 1), min(h - 1, np.max(pixs[0]) + 1)]
    box[[1, 3]] = [max(0, np.min(pixs[1]) - 1), min(w - 1, np.max(pixs[1]) + 1)]

    if context > 0.0:
        sz = np.array([box[2] - box[0], box[3] - box[1]]) * context
        box[0] = max(0, box[0] - sz[0] - 1)
        box[1] = max(0, box[1] - sz[1] - 1)
        box[2] = min(h - 1, box[2] + sz[0] + 1)
        box[3] = min(w - 1, box[3] + sz[1] + 1)

    return np.uint32(box)


def crop_image(image, box, forward=True, image_size=None):
    """ image must be height, width, dim or height width
    """
    box = np.uint32(box)
    if forward:
        if len(image.shape) == 2:
            return image[box[0]:box[2], box[1]:box[3]]
        elif len(image.shape) == 3:
            return image[box[0]:box[2], box[1]:box[3], :]
    else:
        out_image = np.zeros(image_size)
        if len(image.shape) == 2:
            out_image[box[0]:box[2], box[1]:box[3]] = image
        else:
            out_image[box[0]:box[2], box[1]:box[3], :] = image
        return out_image


def drawboxes(image, boxes, color=None):
    """ in box shape, t, l, b, r
    """
    img = image.copy()
    color = [255, 0, 0] if color is None else color
    lw = 4
    for box in boxes:
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]),
                tuple(color), lw)

    return img


def iou(mask1, mask2):
    """ the intersection over union between two masks
    """
    inter = np.float32(np.sum(mask1 == mask2))
    union = np.float32(np.sum(np.logical_or(mask1 > 0, mask2 > 0)))
    return inter / union


def find_bin_idx(vals, bin_vals, is_equal=False):
    """find the ground truth bin index
    """
    def get_idx(val, bins):
        if is_equal:
            return np.round((val - bins[0]) / (bins[1] - bins[0]))
        else:
            return np.argwhere(bin_vals > val)[0][0]
    gt_idx = np.zeros(len(vals))
    for i, (val, bins) in enumerate(zip(vals, bin_vals)):
        gt_idx[i] = get_idx(val, bins)

    return gt_idx


def padding_image(image_in, crop, image_size,
        interpolation=cv2.INTER_NEAREST, pad_val=0.):
    """Pad image to target image_size based on given crop
    """
    image = image_in.copy()
    if np.ndim(image) == 2:
        image = image[:, :, None]
    dim = image.shape[2]
    image_pad = pad_val * np.ones(image_size + [dim], dtype=image_in.dtype)
    h, w = image_size
    crop_cur = np.uint32([crop[0] * h, crop[1] * w, crop[2] * h, crop[3] * w])
    image = cv2.resize(
            image, (crop_cur[3] - crop_cur[1], crop_cur[2] - crop_cur[0]),
            interpolation=interpolation)
    image = image[:, :, None] if np.ndim(image) == 2 else image
    image_pad[crop_cur[0]:crop_cur[2], crop_cur[1]:crop_cur[3], :] = image

    if np.ndim(image) == 3:
        image_pad = np.squeeze(image_pad)

    return image_pad


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../')
    import time
    import data.zpark as zpark
    # find_bin_idx(vals, bins_vals)
    # log_file = '/home/peng/baidu/VideoSegRec/car_fitting/log/mp_policy_mcmc_kitti.log'
    # log_file = '/home/peng/baidu/VideoSegRec/car_fitting/log/mp_policy_mcmc_kitti_w_prev.log'
    # plot_mcmc_log(log_file, metric_name=['reward', 'IOU', 'delta'])

    # temp = [np.random.uniform(size=10), np.random.uniform(size=20)]
    # plot_histogram(temp, labels=['1', '2'])

    # test_image = '/home/peng/Data/zpark/SemanticLabel/Record001/Camera_1/170427_223140575_Camera_1.png'
    # test_image = '/home/peng/Data/zpark/SemanticLabel/Record001/Camera_1/170427_222952577_Camera_1.png'
    # params = zpark.set_params()
    # color_params = zpark.gen_color_list(params['data_path'] + 'color_v2.lst')
    # label_color = cv2.imread(test_image)

    # label_color = label_color[:, :, ::-1]
    # label_color = cv2.resize(label_color, (512, 512),
    #         interpolation=cv2.INTER_NEAREST)
    # # label_color = label_color[100:130, 100:130, :]
    # s = time.time()
    # label_id_0 = color2label_slow(label_color, color_params['color_map'])
    # label_id = color2label(label_color, color_params['color_map'])
    # label_id[label_id >= color_params['color_num']] = 0
    # print np.unique(label_id_0)
    # print np.unique(label_id)
    # plot_images({'diff':(label_id - label_id_0)})
    # # assert np.sum(label_id_0 - label_id) == 0
    # print time.time() - s

