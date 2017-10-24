import keras.backend as K

from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Merge, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

BASE_URL = 'https://github.com/DeepCognition/keras-squeezenet/releases/download/v0.3/'
TH_WEIGHTS_PATH = BASE_URL + 'squeezenet_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = BASE_URL + 'squeezenet_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = BASE_URL + 'squeezenet_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = BASE_URL + 'squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5'

def _fire(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, 1, 1, activation='relu', border_mode='valid', name=name + "/squeeze1x1")(x)
    expand1 = Convolution2D(ex1_filters, 1, 1, activation='relu', border_mode='valid', name=name + "/expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, 3, 3, activation='relu', border_mode='same', name=name + "/expand3x3")(squeeze)
    axis = 3
    if K.image_dim_ordering() == 'th':
        axis = 1
    x = Merge(concat_axis=axis, name=name+'concat', mode='concat')([expand1, expand2])
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

    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="valid", activation="relu", name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(x)

    x = _fire(x, (16, 64, 64), name="fire2")
    x = _fire(x, (16, 64, 64), name="fire3")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3')(x)

    x = _fire(x, (32, 128, 128), name="fire4")
    x = _fire(x, (32, 128, 128), name="fire5")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5')(x)

    x = _fire(x, (48, 192, 192), name="fire6")
    x = _fire(x, (48, 192, 192), name="fire7")

    x = _fire(x, (64, 256, 256), name="fire8")
    x = _fire(x, (64, 256, 256), name="fire9")

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
                                        md5_hash='b3baf3070cc4bf476d43a2ea61b0ca5f')
            else:
                weights_path = get_file('squeezenet_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
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
                                        cache_subdir='models')
            else:
                weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano' or K.backend() == "mxnet":
                convert_all_kernels_in_model(model)

    return model
