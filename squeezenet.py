import keras.backend as K

from keras.models import Model
from keras.layers import Input, Flatten, Dropout, merge, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
import warnings

BASE_URL = 'https://github.com/DeepCognition/keras-squeezenet/releases/download/v0.4/'
TH_WEIGHTS_PATH = BASE_URL + 'squeezenet_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = BASE_URL + 'squeezenet_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = BASE_URL + 'squeezenet_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = BASE_URL + 'squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5'

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire' + str(fire_id) + '/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1

    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
    return x

def SqueezeNet(name="SqueezeNet", include_top=False, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    if include_top:
        x = Dropout(0.5, name='dropout9')(x)

        x = Convolution2D(classes, 1, 1, border_mode='valid', activation="relu", name='conv10')(x)
        x = GlobalAveragePooling2D(name="avgpool10")(x)
        x = Activation("softmax", name='softmax')(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D(name="avgpool10")(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D(name="maxpool10")(x)
            
    model = Model(img_input, x, name=name)

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('squeezenet_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='0c0fd03dfaba6887135e112eb930c586')
            else:
                weights_path = get_file('squeezenet_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='fb2d6b5a715e6076f6c0cca1226f7795')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='aa024f186955a4c1661bd2075dc374a8')
            else:
                weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='80b66c92d79d4310e07fb60a297bf0d1')
            model.load_weights(weights_path)
            if K.backend() == 'theano' or K.backend() == "mxnet":
                convert_all_kernels_in_model(model)

    return model
