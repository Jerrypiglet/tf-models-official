
import paddle.v2.layer as pd


def ele_one_hot(input, class_num, height, width):
    input_rs = pd.resize(input=input, size=1)
    one_hot = pd.one_hot(input=input_rs, class_num=class_num)
    one_hot = pd.resize(input=one_hot, size=height * width * class_num)

    one_hot = pd.transpose(input=one_hot,
                           trans_order=[2, 0, 1],
                           height=width,
                           width=class_num,
                           channels=height)
    return one_hot


def math_op(input, act=pd.activation.Linear(), op='dot', size=0):
    if not isinstance(input, list):
        input = [input]

    if len(input) == 1:
        # unary operation
        result = pd.mixed(
                input=[pd.identity_projection(input=input[0])], act=act)

    elif len(input) == 2:
        # binary operation
        if op == 'dot':
            result = pd.mixed(size=size,
                              input=pd.dotmul_operator(
                                        a=input[0],
                                        b=input[1],
                                        scale=1.0),
                              act=act)
    else:
        raise ValueError('not supporting math op with more than two\
                          input')

    return result


def reduce(input, shape, op, axis=1):
    """reduce with op in axis dimension

       shape: [channel, height, width]
    """
    if op == 'sum':
        if axis == 1:
            input= pd.transpose(input=input,
                                trans_order=[1, 2, 0],
                                height=shape[1],
                                width=shape[2])
        input = pd.resize(input=input, size=shape[0],
                          height=1, width=1)
        input = pd.sum_cost(input=input)
        input = pd.resize(input=input, size=shape[1] * shape[2],
                          height=shape[1], width=shape[2])

    return input


def get_cnn_input(name, size, channel):
    input = pd.data(
        name=name, type=pd.data_type.dense_vector(
            channel * size[0] * size[1]),
        height=size[0], width=size[1])

    return {name: input}


def power(input, power=0.5):
    """power layer for input
    """
    output = math_op(input=input, act=pd.activation.Log())
    output = pd.slope_intercept(input=output, slope=power)
    output = math_op(input=output, act=pd.activation.Exp())
    return output


def add(input, other):
    output = pd.slope_intercept(input=input, intercept=other)
    return output


def mul(input, other):
    output = pd.slope_intercept(input=input, slope=other)
    return output


def norm(input, height, width, channel, type='l2', trans_back=True):
    """Channel wise normalize each layer
    """
    size = height * width * channel
    if height > 1 or width > 1:
        input= pd.transpose(input=input,
                            trans_order=[1, 2, 0],
                            height=height,
                            width=width)
        input = pd.resize(input=input, size=channel)

    if type == 'l2':
        norm = pd.mixed(size=size,
                        input=[pd.dotmul_operator(a=input,
                                                  b=input,
                                                  scale=1.0)])
        norm = pd.sum_cost(input=norm)
        norm = math_op(norm, pd.activation.Sqrt())

    if type == 'l1':
        norm = math_op(input, pd.activation.Abs())
        norm = pd.sum_cost(input=norm)

    norm_inv = math_op(norm, pd.activation.Inv())
    norm_inv = pd.repeat(input=norm_inv, num_repeats=3)
    input = math_op(input=[input, norm_inv],
                    act=None, op='dot', size=size)

    if trans_back:
        input = pd.resize(input=input, size=size)
        input = pd.transpose(input=input,
                             trans_order=[2, 0, 1],
                             height=width,
                             width=channel,
                             channels=height)
    return input
