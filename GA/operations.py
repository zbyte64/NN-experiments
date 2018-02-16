import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError
import numpy as np
from functools import reduce
import traceback

#populated from: https://www.tensorflow.org/api_docs/python/nn/
OPERATIONS = dict()

ArgMax = lambda x: np.iinfo(x.dtype).max if str(x.dtype).startswith('int') else np.finfo(x.dtype).max
ArgNormalize = lambda x: np.float64(np.abs(x)) / ArgMax(x)

_OPERATION_KEYS = None
def opkey_from_arg(arg):
    global _OPERATION_KEYS
    if _OPERATION_KEYS and len(_OPERATION_KEYS) != len(OPERATIONS):
        print("OPERATION KEYS were stale!")
        _OPERATION_KEYS = None
    if not _OPERATION_KEYS:
        _OPERATION_KEYS = list(OPERATIONS.keys())
        #move passthrough to the begining, 0 = no-op
        _p_index = _OPERATION_KEYS.index('passthrough')
        _OPERATION_KEYS.insert(0, _OPERATION_KEYS.pop(_p_index))
    o_index = int((len(_OPERATION_KEYS)-1) * ArgNormalize(arg))
    return _OPERATION_KEYS[o_index]
    
    
def OperationsFromMatrix(input_layer, matrix):
    if matrix.ndim == 1:
        key = opkey_from_arg(matrix[0])
        try:
            next_layer = OPERATIONS[key][0](input_layer, *matrix[1:])
        except (InvalidArgumentError, ZeroDivisionError, TypeError, ValueError, AttributeError) as error:
            #print(error)
            if str(error).startswith('Shape must be rank'): #not worth mentioning
                return input_layer
            if str(error).startswith('Rank of inputs is'):
                return input_layer
            print("Skipping failed layer:", type(error), key, matrix[1:])
            print(error)
            return input_layer
        except Exception as error:
            print("Operation construction failed:", type(error), key, matrix[1:])
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print("skipping layer")
            return input_layer
        else:
            next_shape = next_layer.get_shape()
            if (len(next_shape) > 1 and not next_shape[-1].value):
                print("Bad return shape:", key, next_shape)
                return input_layer
            return next_layer
    if matrix.ndim == 2:
        layer = input_layer
        for layer_matrix in matrix:
            layer = OperationsFromMatrix(layer, layer_matrix)
        return layer
    #TODO 3dim?
    raise TypeError("Too many dimensions: "+str(matrix.ndim))

    
def ReprFromMatrix(matrix, layer=None, include_all=False):
    repr_layers = list()
    if layer is None:
        layer = tf.placeholder(tf.float32, shape=(None, 20))
    for row in matrix:
        key = opkey_from_arg(row[0])
        l_repr = OPERATIONS[key][1](layer, *row[1:])
        next_layer = None
        try:
            next_layer = OPERATIONS[key][0](layer, *row[1:])
        except Exception as error:
            if include_all:
                print(error)
                traceback.print_exc()
        if next_layer is None:
            if include_all:
                repr_layers.append(l_repr)
        elif next_layer is not layer:
            next_shape = next_layer.get_shape()
            if (len(next_shape) > 1 and not next_shape[-1].value):
                #print("Bad return shape:", key, next_shape)
                if include_all:
                    repr_layers.append(l_repr)
            else:
                repr_layers.append(l_repr)
                layer = next_layer
        else:
            if include_all:
                repr_layers.append(l_repr)
    return repr_layers
    
    
def Args(f, *d_args, **d_kwargs):
    def get_args(layer, g_args):
        kwargs = {}
        args = []
        for g_arg, d_arg in zip(g_args, d_args):
            g_arg = ArgNormalize(g_arg)
            args.append(d_arg(layer, g_arg))
        for g_arg, (d_name, d_arg) in zip(g_args[len(args):], d_kwargs.items()):
            g_arg = ArgNormalize(g_arg)
            kwargs[d_name] = d_arg(layer, g_arg)
        return tuple(args), kwargs
        
    def builder(layer, *g_args):
        args, kwargs = get_args(layer, g_args)
        return f(layer, *args, **kwargs)
    
    def rdoc(layer, *g_args):
        args, kwargs = get_args(layer, g_args)
        return (f.__name__, args, kwargs)
    
    return builder, rdoc


def Choices(values):
    def f(layer, arg_value):
        index = int((len(values)-1) * arg_value)
        return values[index]
    return f

def Scaled(arg_max):
    def f(layer, arg_value):
        value = arg_max * arg_value
        #coerce to arg_max dtype
        if type(value) != type(arg_max):
            value = type(arg_max)(value)
        return value 
    return f

def Range(arg_min, arg_max):
    def f(layer, arg_value):
        value = (arg_max - arg_min) * arg_value + arg_min
        #coerce to arg_max dtype
        if type(value) != type(arg_max):
            value = type(arg_max)(value)
        return value 
    return f

def DimSelect(layer, arg_value):
    arg_max = len(layer.get_shape()) - 1
    value = int(arg_max * arg_value)
    return value

def KernelSize(dims):
    def f(layer, arg_value):
        size = int(5 * arg_value)+1
        channels = int(layer.get_shape()[-1])
        k = []
        for i in range(dims-1):
            k.append(size)
        k.append(channels)
        return k
    return f

def Strides(scalar):
    def f(layer, arg_value):
        rate = int(scalar * arg_value)+1
        dims = len(layer.get_shape())
        k = [1]
        for i in range(dims-2):
            k.append(rate)
        k.append(1)
        return k
    return f

def filter_size(layer, arg_value):
    size = int(5 * arg_value+1)
    channels = int(layer.get_shape()[-1])
    #TODO cast to layer dtype
    return [size, size, channels, channels]

def kernel_size(layer, arg_value):
    size = int(5 * arg_value+1)
    channels = int(layer.get_shape()[-1])
    dims = len(layer.get_shape())
    
    k = list()
    for i in range(dims-1):
        k.append(size)
    k.append(channels)
    return k

def rate_stride(layer, arg_value):
    rate = arg_value+1
    #width, height = layer.get_shape()[1:2]
    return [1, rate, rate, 1]

def strides(layer, arg_value):
    rate = int(10*arg_value+1)
    return [1, rate, rate, 1]

def dynamic_strides(layer, arg_value):
    rate = arg_value+1
    dims = len(layer.get_shape())
    k = [1]
    for i in range(dims-3):
        k.append(rate)
    return k

def filters_and_strides(layer, scalar):
    shape = layer.get_shape()
    max_stride = int(min(shape[1:-1]))
    stride_step = int(scalar*max_stride)
    strides = [stride_step] * (len(shape)-2)
    strides.insert(0, 1)
    strides.append(1)
    
    #layer.size == filters.size
    #filter_height, filter_width, in_channels, out_channels]
    filters = np.array([*(int(i) for i in shape[1:]), stride_step])
    filters = filters.astype(layer.dtype.name)
    #filters = strides #TODO must be same size but can be different values
    return filters, strides


def op(key, fn):
    def wrap(fn_args):
        def execute(layer, *args):
            args = [ArgNormalize(a) for a in args]
            fargs, fwargs = fn_args(layer, args)
            return fn(*fargs, **fwargs)
        def document(layer, *args):
            args = [ArgNormalize(a) for a in args]
            fargs, fwargs = fn_args(layer, args)
            return (key, tuple(filter(lambda x: x is not layer, fargs)), fwargs)
        OPERATIONS[key] = (execute, document)
    return wrap

        
OPERATIONS['passthrough'] = Args(lambda layer: layer)

#activators
OPERATIONS['relu'] = Args(tf.nn.relu)
OPERATIONS['relu6'] = Args(tf.nn.relu6)
OPERATIONS['crelu'] = Args(tf.nn.crelu)
OPERATIONS['elu'] = Args(tf.nn.elu)
OPERATIONS['softplus'] = Args(tf.nn.softplus)
OPERATIONS['softsign'] = Args(tf.nn.softsign)
OPERATIONS['dropout'] = Args(tf.nn.dropout, Range(0.01, 1.0)) 
OPERATIONS['bias_add'] = Args(tf.nn.bias_add, Range(0.01, 1.0))
OPERATIONS['sigmoid'] = Args(tf.sigmoid)
OPERATIONS['tanh'] = Args(tf.tanh)

PADDING = Choices(['SAME', 'VALID'])

#convolutions
#want binary strides?
#OPERATIONS['convolution'] = Args(tf.nn.convolution, padding=PADDING, strides=dynamic_strides, filter=filter_size)

#OPERATIONS['conv2d'] = Args(tf.nn.conv2d, padding=PADDING, strides=Strides(5), filter=filter_size)
'''
@op('conv2d', tf.contrib.layers.convolution2d#tf.nn.conv2d)
def conv2d(layer, args):
    filters, strides = filters_and_strides(layer, args[0])
    padding = PADDING(layer, args[1])
    return (layer,), dict(filter=filters, strides=strides, padding=padding)


OPERATIONS['depthwise_conv2d'] = Args(tf.nn.depthwise_conv2d, padding=PADDING, filter=filter_size, strides=strides)
OPERATIONS['conv1d'] = Args(tf.nn.conv1d, filters=filter_size, stride=Range(1, 5), padding=PADDING)

#pooling (ksize, strides settled)
OPERATIONS['avg_pool'] = Args(tf.nn.avg_pool, ksize=KernelSize(4), strides=strides, padding=PADDING)
OPERATIONS['max_pool'] = Args(tf.nn.max_pool, ksize=KernelSize(4), strides=strides, padding=PADDING)
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard9/tf.nn.max_pool_with_argmax.md
#OPERATIONS['max_pool_with_argmax'] = Args(tf.nn.max_pool_with_argmax, ksize=KernelSize(4), strides=strides, padding=PADDING)
OPERATIONS['avg_pool3d'] = Args(tf.nn.avg_pool3d, ksize=KernelSize(5), strides=Strides(5), padding=PADDING)
OPERATIONS['max_pool3d'] = Args(tf.nn.max_pool3d, ksize=KernelSize(5), strides=Strides(5), padding=PADDING)

#morphological
OPERATIONS['dilation2d'] = Args(tf.nn.dilation2d, filter=Scaled(5.0), strides=strides, rates=rate_stride, padding=PADDING)
OPERATIONS['erosion2d'] = Args(tf.nn.erosion2d, kernel=kernel_size, strides=Range(1, 5), rates=rate_stride, padding=PADDING)
'''
#normalization
OPERATIONS['l2_normalize'] = Args(tf.nn.l2_normalize, dim=DimSelect)
OPERATIONS['local_response_normalization'] = Args(tf.nn.local_response_normalization, depth_radius=Range(1, 5))

#rnn
#OPERATIONS['dynamic_rnn'] = Args(tf.nn.dynamic_rnn)

#layers: https://www.tensorflow.org/api_docs/python/contrib.layers/
@op('avg_pool2d', tf.contrib.layers.avg_pool2d)
def avg_pool2d(layer, args):
    s = lambda x: int(x*10+1)
    kernel_size = [s(args[0])] * 2 
    stride = [s(args[1])] * 2
    padding = PADDING(layer, args[2])
    return (layer,), dict(kernel_size=kernel_size, stride=stride, padding=padding)

@op('conv2d', tf.contrib.layers.convolution2d)
def conv2d(layer, args):
    #nodes = reduce(lambda c,x: c*int(x), layer.get_shape()[1:], 1)
    #num_outputs = int(args[0]*nodes/2+1)
    #in_channels = int(layer.get_shape()[-1])
    num_outputs = int(1+args[0]*63)
    kernel_size = int(args[1]*10+1)
    stride = int(args[2]*10+1)
    padding = PADDING(layer, args[3])
    return (layer,), dict(num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding)

@op('conv2d_in_plane', tf.contrib.layers.convolution2d_in_plane)
def conv2d_in_plane(layer, args):
    kernel_size = int(args[0]*10+1)
    stride = int(args[1]*10+1)
    padding = PADDING(layer, args[2])
    return (layer,), dict(kernel_size=kernel_size, stride=stride, padding=padding)

@op('conv2d_transpose', tf.contrib.layers.convolution2d_transpose)
def conv2d_transpose(layer, args):
    #nodes = reduce(lambda c,x: c*int(x), layer.get_shape()[1:], 1)
    #in_channels = int(layer.get_shape()[-1])
    num_outputs = int(1+args[0]*63)
    kernel_size = int(args[1]*10+1)
    stride = int(args[2]*10+1)
    padding = PADDING(layer, args[3])
    return (layer,), dict(num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding)

#OPERATIONS['avg_pool2d'] = Args(tf.contrib.layers.avg_pool2d, kernel_size=Range(1, 5), stride=Range(1, 5), padding=PADDING)
#OPERATIONS['batch_norm'] = Args(tf.contrib.layers.batch_norm)
#OPERATIONS['convolution2d'] = Args(tf.contrib.layers.convolution2d, num_outputs=Range(1, 8), kernel_size=Range(1, 5), stride=Range(1, 5), padding=PADDING)
OPERATIONS['flatten'] = Args(tf.contrib.layers.flatten)
#TODO relative to incoming layer size
#OPERATIONS['fully_connected'] = Args(tf.contrib.layers.fully_connected, num_outputs=Range(1, 1024))
#OPERATIONS['layer_norm'] = Args(tf.contrib.layers.layer_norm)
#OPERATIONS['max_pool2d'] = Args(tf.contrib.layers.max_pool2d, kernel_size=Range(1, 5), stride=Range(1, 5), padding=PADDING)
#OPERATIONS['l1_regularizer'] = Args(tf.contrib.layers.l1_regularizer)
@op('fully_connected', tf.contrib.layers.fully_connected)
def fully_connected(layer, args):
    nodes = reduce(lambda c,x: c*int(x), layer.get_shape()[1:], 1)
    reduce_size = args[0] <= .8 or nodes > 1024
    if reduce_size:
        num_outputs = int(args[1]*min(nodes/2, 1024*8))
    else:
        num_outputs = int(args[1]*min(nodes*2, 1024*8))
    return (layer,), dict(num_outputs=num_outputs)

@op('batch_norm', tf.contrib.layers.batch_norm)
def batch_norm(layer, args):
    center = True if args[0] >= .5 else False
    scale = True if args[1] >= .5 else False
    decay = min(args[2] / 10 + .9, .99999999)
    return (layer,), dict(center=center, scale=scale)

@op('layer_norm', tf.contrib.layers.layer_norm)
def layer_norm(layer, args):
    center = True if args[0] >= .5 else False
    scale = True if args[1] >= .5 else False
    return (layer,), dict(center=center, scale=scale)

@op('max_pool2d', tf.contrib.layers.max_pool2d)
def max_pool2d(layer, args):
    s = lambda x: int(x*10+1)
    kernel_size = [s(args[0])] * 2 
    stride = [s(args[1])] * 2
    padding = PADDING(layer, args[2])
    return (layer,), dict(kernel_size=kernel_size, stride=stride, padding=padding)

@op('separable_convolution2d', tf.contrib.layers.separable_convolution2d)
def separable_convolution2d(layer, args):
    in_channels = int(layer.get_shape()[-1])
    depth_multiplier = int(args[0]*10+1)
    kernel_size = int(args[1]*10+1)
    stride = int(args[2]*10+1)
    padding = PADDING(layer, args[3])
    num_outputs = depth_multiplier * in_channels 
    return (layer,), dict(num_outputs=num_outputs, depth_multiplier=depth_multiplier, kernel_size=kernel_size, stride=stride, padding=padding)

@op('unit_norm', tf.contrib.layers.unit_norm)
def unit_norm(layer, args):
    max_dim = len(layer.get_shape())-1
    dim = int(args[0] * max_dim)
    return (layer,), dict(dim=dim)
