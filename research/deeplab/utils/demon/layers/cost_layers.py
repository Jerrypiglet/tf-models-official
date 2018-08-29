import paddle.trainer.config_parser as cp
import paddle.v2.layer as pd
import utils.utils as uts
import util_layers

cost_func = {'l1': pd.smoothl1_cost,
			 'l2': pd.mse_cost}

image_resize_func = {'nearest': pd.nearest_interp,
                     'bilinear': pd.bilinear_interp}


def sum_weighted_loss(loss, weight, size=1):
    """Loss has input batch_size x image_size, weight has input batch_size x weight
        ( i * w ) / sum(W)
       The output is normalized weighted loss
    """
    weighted_loss = pd.mixed(size=size,
                     input=[pd.dotmul_operator(a=loss, b=weight,scale=1.0)])
    weight_fac = pd.sum_cost(input=weight)
    weight_fac = util_layers.math_op(input=weight_fac,act=pd.activation.Inv())
    weighted_loss = pd.scaling(input=loss, weight=weight_fac)
    weighted_loss = pd.sum_cost(input=weighted_loss)

    return weighted_loss


def ele_norm_cost(input, label, weight,
                  height=None,
                  width=None,
                  num_channel=None,
                  cost_type='l1'):
    if height > 1 and width > 1:
        input = pd.bilinear_interp(input=input, out_size_x=width,
            out_size_y=height)
        label = pd.bilinear_interp(input=label,out_size_x=width,
            out_size_y=height)
        if weight:
            weight = pd.nearest_interp(input=weight,out_size_x=width,
                out_size_y=height)

    size = height * width * num_channel
    if weight:
        input = pd.mixed(size=size,
                input=[pd.dotmul_operator(a=input, b=weight, scale=1.0)])
        label = pd.mixed(size=size,
                input=[pd.dotmul_operator(a=label, b=weight, scale=1.0)])
        cost = cost_func[cost_type](input=input, label=label)
        fac = pd.sum_cost(input=weight)
        fac = util_layers.math_op(input=fac, act=pd.activation.Inv())
        cost = pd.scaling(input=cost, weight=fac)
        cost = pd.sum_cost(input=cost)
    else:
        cost = cost_func[cost_type](input=input, label=label)
        fac = 1.0 / float(height * width)
        cost = pd.slope_intercept(input=cost, slope=fac, intercept=0.0)
        cost = pd.sum_cost(input=cost)

    return cost


def ns_ele_l2_cost(input, label, weight,
                   height, width, num_channel=None,
                   interp='nearest'):
    assert interp in image_resize_func.keys()
    # make sure all the input label and weight have the same size
    input = pd.bilinear_interp(input=input, out_size_x=width,
        out_size_y=height)
    label = image_resize_func[interp](input=label, out_size_x=width,
        out_size_y=height)
    weight = image_resize_func[interp](input=weight, out_size_x=width,
        out_size_y=height)

    # reshape the orignal layer
    # input has shape  c x h x w change to h x w x c
    input_ts = pd.transpose(input=input,
                            trans_order=[1, 2, 0],
                            height=height,
                            width=width)
    input_rs = pd.resize(input=input_ts, size=num_channel, height=1, width=1)

    label_ts = pd.transpose(input=label,
                            trans_order=[1, 2, 0],
                            height=height,
                            width=width)
    label_rs = pd.resize(input=label_ts, size=num_channel, height=1, width=1)
    weight_rs = pd.resize(input=weight, size=1, height=1, width=1)

    cost_rs = pd.mse_cost(input=input_rs, label=label_rs)
    sqrt_l2_cost = util_layers.math_op(input=cost_rs, act=pd.activation.Sqrt())
    sqrt_l2_cost = pd.mixed(size=1,input=[pd.dotmul_operator(
               a=sqrt_l2_cost,b=weight_rs,scale=1.0)])
    sqrt_l2_cost = pd.resize(input=sqrt_l2_cost, size=height * width,
                             height=height, width=width)

    weight_fac = pd.sum_cost(input=weight)
    weight_fac = util_layers.math_op(input=weight_fac, act=pd.activation.Inv())
    sqrt_l2_cost = pd.scaling(input=sqrt_l2_cost, weight=weight_fac)
    cost = pd.sum_cost(input=sqrt_l2_cost)

    return cost


def gradient_cost(input, label, weight,
                  height, width, num_channel, scales=[1,2]):
    grad_diff_input = pd.gradient_diff(input=input, scales=scales)
    grad_diff_label = pd.gradient_diff(input=label, scales=scales)

    weight = image_resize_func['nearest'](input=weight,
                                          out_size_x=width,
                                          out_size_y=height)

    weight_diff = pd.gradient_diff(input=weight, scales=scales)
    weight_diff = util_layers.reduce(input=weight_diff,
                                     shape=[len(scales) * 2, height, width],
                                     op='sum')
    weight_diff = util_layers.math_op(input=weight_diff,
                                      act=pd.activation.IsZero())
    weight_diff = util_layers.math_op(input=[weight_diff, weight], op='dot')

    out_ch = num_channel * len(scales) * 2
    cost = ns_ele_l2_cost(grad_diff_input, grad_diff_label,
                          weight_diff, height, width, num_channel=out_ch)

    return cost


def relative_l1(input, label, weight,
                height, width,
                interp='nearest',
                is_inverse=False):
    """Relative l1 loss for depth
    """

    assert interp in image_resize_func.keys()

    # make sure all the input label and weight have the same size
    if height > 1 and width > 1:
        input = pd.bilinear_interp(input=input, out_size_x=width,
            out_size_y=height)
        label = pd.bilinear_interp(input=label, out_size_x=width,
            out_size_y=height)
        if weight:
            weight = image_resize_func[interp](input=weight, out_size_x=width,
                out_size_y=height)

    label_inv = util_layers.math_op(input=label, act=pd.activation.Inv())
    label_neg = pd.slope_intercept(input=label, slope=-1)
    diff = pd.addto(input=[input, label_neg],act=pd.activation.Abs(),
                    bias_attr=False)

    rel_error = pd.mixed(size=1,input=[pd.dotmul_operator(
               a=diff,b=label_inv,scale=1.0)])

    if weight:
        rel_error = sum_weighted_loss(rel_error, weight, size=height * width)
    else:
        fac = 1.0 / float(height * width)
        inner = pd.slope_intercept(input=inner, slope=fac, intercept=0.0)
        inner_error = pd.sum_cost(input=inner)

    return rel_error



def inner_product_cost(input, label, weight,
                       height, width, num_channel, interp='nearest',
                       is_angle=False):
    """If is_angle, we can not back propagate through the angle, only back
       through the inner product, the loss is not consistent with the evaluation.
    """
    # make sure all the input label and weight have the same size
    if height > 1 and width > 1:
        input = pd.bilinear_interp(input=input, out_size_x=width,
            out_size_y=height)
        label = pd.bilinear_interp(input=label, out_size_x=width,
            out_size_y=height)
        if weight:
            weight = image_resize_func[interp](input=weight, out_size_x=width,
                out_size_y=height)

    size = height * width * num_channel

    input = util_layers.norm(input, height, width, num_channel,
        trans_back=False)
    label = util_layers.norm(label, height, width, num_channel,
        trans_back=False)

    inner = pd.mixed(size=size,
                     input=[pd.dotmul_operator(
                            a=input,b=label,scale=1.0)])
    inner = pd.resize(input=pd.sum_cost(input=inner),
                      size=height*width,
                      height=height, width=width)
    if is_angle:
        inner = util_layers.math_op(input=inner, act=pd.activation.Acos())
    else:
        inner = pd.slope_intercept(input=inner, slope=-1, intercept=1.0)

    if weight:
        inner_error = sum_weighted_loss(inner, weight, size=height * width)
    else:
        fac = 1.0 / float(height * width)
        inner = pd.slope_intercept(input=inner, slope=fac, intercept=0.0)
        inner_error = pd.sum_cost(input=inner)

    return inner_error


def pixel_accuracy(input, label, weight,
                   height, width):
    label = pd.nearest_interp(input=label, out_size_x=width,
                              out_size_y=height)
    weight = pd.nearest_interp(input=weight, out_size_x=width,
                              out_size_y=height)

    label_neg = pd.slope_intercept(input=label, slope=-1)
    diff = pd.addto(input=[input, label_neg],
                    act=pd.activation.Linear(), bias_attr=False)
    correct = util_layers.math_op(
        input=diff, act=pd.activation.IsZero())

    correct = util_layers.math_op(input=[correct, weight], op='dot')

    correct = pd.sum_cost(input=correct)
    eval_pixels_num = pd.sum_cost(input=weight)
    divider = util_layers.math_op(
        input=eval_pixels_num, act=pd.activation.Inv())
    acc = util_layers.math_op(input=[correct, divider], op='dot')

    return acc


def iou_score(input, label, weight,
             height, width, class_num, is_cost=True):
    """ class num is semantic classes plus background,
        this score can also serve as iou cost for training
    """
    # input = pd.resize(input=input, size=height * width)
    # label = pd.resize(input=label, size=height * width)

    weight = pd.nearest_interp(input=weight, out_size_x=width,
                               out_size_y=height)
    if not is_cost:
        # if not is cost, then it is eval, we can do
        # one hot for label. Otherwise
        input = util_layers.math_op(input=[input, weight], op='dot')
        input_one_hot = util_layers.ele_one_hot(input, class_num,
            height, width)
    else:
        input_one_hot = input
        input_one_hot = pd.bilinear_interp(input=input_one_hot,
            out_size_x=width, out_size_y=height)

    label = pd.nearest_interp(input=label, out_size_x=width,
                              out_size_y=height)
    label = util_layers.math_op(input=[label, weight], op='dot')

    label_one_hot = util_layers.ele_one_hot(
            label, class_num, height, width)
    inter = util_layers.math_op(input=[input_one_hot, label_one_hot],
          op='dot')
    union = pd.addto(input=[input_one_hot, label_one_hot],
                     act=pd.activation.Linear(), bias_attr=False)
    inter_neg = pd.slope_intercept(input=inter, slope=-1)

    union = pd.addto(input=[union, inter_neg],
                     act=pd.activation.Linear(), bias_attr=False)

    inter = pd.resize(input=inter, size=height * width)
    inter = pd.sum_cost(input=inter)
    union = pd.resize(input=union, size=height * width)
    union = pd.sum_cost(input=union)

    union_inv = util_layers.math_op(input=union, act=pd.activation.Inv())
    iou = pd.mixed(size=1,
                   input=[pd.dotmul_operator(
                        a=inter,
                        b=union_inv,
                        scale=1.0)])
    iou = pd.resize(input=iou, size=class_num)

    if is_cost:
        iou = pd.sum_cost(iou)

    return iou


