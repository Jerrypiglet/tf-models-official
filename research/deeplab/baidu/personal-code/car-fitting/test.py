
import cv2
import numpy as np
import utils.utils as uts
import pdb
import Networks.util_layers as utl
import Networks.mx_losses as mx_loss
import matplotlib.pyplot as plt
import data.kitti as kitti
import scipy.io as io
# import evaluation.eval_network as eval_net
import mxnet as mx

from multiprocessing import Process, current_process

def test():
    print("process id is {:s}".format(current_process().name))
    a = mx.nd.array(np.zeros((10, 10)), mx.gpu(0))
    print a.asnumpy()

def test_mx_index():
    temp = mx.nd.ones((10, 10))
    indx = mx.nd.floor(mx.nd.array(np.array([[0.5, 1, 2], [0, 1.2, 2]])))
    out = mx.nd.gather_nd(temp, indx)
    print out.asnumpy()

def test1():
    temp = mx.nd.ones((10, 6))
    ids = mx.nd.split(temp, axis=1, num_outputs=6, squeeze_axis=1)
    ids_onehot = [mx.nd.one_hot(t, 3) for t in ids]
    print ids_onehot[0]
    # print [t.shape for t in ids_onehot]


def image_to_label(image_in, label, colors=None, sz=None, is_inst=False):

    image = cv2.resize(image_in.copy(), (sz[1], sz[0]),
            interpolation=cv2.INTER_NEAREST)
    label = cv2.resize(label.copy(), (sz[1], sz[0]),
            interpolation=cv2.INTER_NEAREST)

    if colors is None:
        colors = np.random.random((200, 3)) * 255

    label_id = np.unique(label)
    image = 0.5 * image
    print len(label_id)
    for counter, i in enumerate(label_id):
        if i == 0 or i == 255:
            continue
        print i
        if is_inst:
            color_id = counter
        else:
            color_id = i

        frame = np.float32(label == i)
        frame = np.tile(frame[:, :, None], (1, 1, 3))
        image = image + frame * 0.5 * colors[color_id, :]

    return np.uint8(image)


def show_example():
    import data.labels as label
    import data.zpark as zpark
    data_path = '/home/peng/Data/zpark/'
    color_params = zpark.gen_color_list(data_path + 'color_v2.lst')

    home = '/media/peng/DATA/Data/apolloscape/test_vis/'
    image_name = '170927_063932013_Camera_5'
    # image_name = '170908_062145788_Camera_6'
    # image_name = '171206_025742296_Camera_5'
    np_id2trainid = np.zeros(256, dtype = np.uint8)
    for idx, trainid in label.id2trainId.items():
        np_id2trainid[idx] = trainid

    image = cv2.imread('%s/%s.jpg' % (home, image_name),
            cv2.IMREAD_UNCHANGED)
    h, w = image.shape[0] / 2, image.shape[1] / 2
    class_map = cv2.imread('%s/%s_bin.png' % (home, image_name),
                           cv2.IMREAD_UNCHANGED)
    class_map = np_id2trainid[class_map]
    instance_map = cv2.imread('%s/%s_instanceIds.png' % (home, image_name),
                           cv2.IMREAD_UNCHANGED)
    # uts.plot_images({'inst': instance_map},
    #                  layout=[1, 3])
    # class_ids = instance / 1000
    class_map = image_to_label(image, class_map,
            np.array(color_params['color_map_list']), [h, w])
    uts.plot_images({'image_vis': np.uint8(image[:, :, ::-1]),
                     'class': class_map[:, :, ::-1]},
                     layout=[1, 3])
    cv2.imwrite('class.jpg',  class_map)
    inst_map = image_to_label(image, instance_map, sz=[h, w], is_inst=True)
    # uts.plot_images({'image_vis': np.uint8(image[:, :, ::-1]),
    #                  # 'class': class_map[:, :, ::-1],
    #                  'inst': inst_map[:, :, ::-1]},
    #                  layout=[1, 3])
    cv2.imwrite('inst.jpg',  inst_map)



if __name__ == '__main__':
    # test_mx_index()
    show_example()
    # test1()
    # a = mx.nd.array(np.zeros((10, 10)), mx.gpu(3))
    # print a.asnumpy()
    # runs = [Process(target=test) for i in range(2)]  # 1 or 2 or N process is the same error
    # for p in runs:
    #   p.start()
    # for p in runs:
    #   p.join()
    # print("done!")


# x = mx.sym.Variable('x')
# y = mx.sym.Variable('y')
# x_np = [[-0.901,  0.01,   0.001,  0.003, -0.001,  0.004]]
# y_np = [[ 0.,     0.,     0.,     0.837, -2.697,  8.478]]

# h, w = 3, 3
# pose_f = mx.sym.tile(x, reps=(1, h*w))
# pose_f = mx.sym.reshape(pose_f, (-1, h, w, 6))
# loss = mx.sym.transpose(pose_f, (0, 3, 1, 2))

# # loss = mx_loss.my_pose_loss(x, y, 1, balance=1)
# exe = loss.bind(mx.cpu(), {'x': mx.nd.array(x_np)})
# y = exe.forward()
# print y[0].asnumpy()[:, :, 1, 1]

# log_file = './log/policy_net.log'
# names = ['IOU', 'delta']
# uts.plot_my_log(log_file, names)
# params = kitti.set_params_disp()
# img_file = params['image_path'] + '%06d_10.png' % 0
# print img_file
# img = cv2.imread(img_file,
#         cv2.IMREAD_UNCHANGED)
# print img.shape
# x1, y1, x2, y2 = 1, 1, 100, 100
# cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
# plt.imshow(img)
# plt.show()

# x = np.linspace(0, 6*np.pi, 100)
# y = np.sin(x)

# plt.ion()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

# for phase in np.linspace(0, 10*np.pi, 500):
#     line1.set_ydata(np.sin(x + phase))
#     fig.canvas.draw()
#     fig.canvas.flush_events()

# def test():
#     x = mx.nd.array([[  1.,   2.,   3.,   4.],
#                      [  5.,   6.,   7.,   8.],
#                      [  9.,  10.,  11.,  12.]])

#     x2 = mx.nd.reshape(x, (1, 1, 3, 4))
#     pdb.set_trace()
#     a = mx.sym.Variable('a', dtype=np.float64)
#     conv_bn_layer = utl.ConvBNLayer()
#     b = conv_bn_layer(a, 4, 3, 1, name='test')
#     exe = b.bind(mx.cpu(), {'a': x2})
#     y = exe.forward()
#     print y[0].asnumpy()


# a = mx.sym.Variable('a')
# b = mx.sym.Variable('b')
# c = mx.sym.take(a, b, 0)
# d = mx.sym.tile(a, (1, 4))
# e = mx.sym.reshape(d, (3, 2, 2, -1))
# e = mx.sym.transpose(e, (0, 3, 1, 2))


# print x.shape
# # exe = e.bind(mx.cpu(), {'a':x, 'b':mx.nd.array([2])})
# exe = e.bind(mx.cpu(), {'a':x})
# y = exe.forward()

# print y[0].asnumpy()[0, :, 0, 0]
