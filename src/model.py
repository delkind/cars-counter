import keras
import keras.models
import numpy as np
from keras.utils import get_file

from src.misc import PriorProbability, UpsampleLike, smooth_l1, focal, huber_loss


def create_regression_model(
        num_values,
        num_anchors,
        pyramid_feature_size=256,
        regression_feature_size=256,
        name='regression_submodel',
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros'
):
    """
    Creates the default regression submodel.
        :param num_values: Number of values to regress.
        :param num_anchors: Number of anchors to regress for each feature level.
        :param pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
        :param regression_feature_size: The number of filters to use in the layers in the regression submodel.
        :param name: The name of the submodel.
        :param kernel_size: size of the kernel in the regression submodel conv layers
        :param strides: stride in the regression submodel conv layers
        :param padding: padding in the regression submodel conv layers
        :param kernel_initializer: initializer for kernel weights
        :param bias_initializer: intializer for bias weights
        :return: A keras.models.Model that predicts regression values for each anchor.
    """

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )(outputs)
    outputs = keras.layers.Conv2D(num_anchors * num_values,
                                  name='pyramid_regression',
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer
                                  )(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def create_classification_model(
        num_classes,
        num_anchors,
        pyramid_feature_size=256,
        prior_probability=0.01,
        classification_feature_size=256,
        name='classification_submodel',
        kernel_size=3,
        strides=1,
        padding='same'
):
    """
    Creates the default classification submodel.
    :param num_classes: number of classes in the data
    :param num_anchors: Number of anchors to regress for each feature level.
    :param pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
    :param prior_probability: Prior probability for the classification submodel weights initialization
    :param classification_feature_size: number of filters in the classification submodel conv layers
    :param name: The name of the submodel.
    :param kernel_size: size of the kernel in the classification submodel conv layers
    :param strides: stride in the  classification submodel conv layers
    :param padding: padding in the  classification submodel conv layers
    :return: A keras.models.Model that predicts classification values for each anchor.
    """
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        kernel_size=kernel_size,
        strides=strides,
        padding=padding
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


class AppResNetBackBone:
    """
    Resnet50 backbone from keras.applications
    """
    def __init__(self):
        from keras.applications.resnet50 import ResNet50
        resnet = ResNet50(include_top=False, weights='imagenet')
        layer_names = ['bn%s_branch2c' % layer_name for layer_name in ['2c', '3d', '4f', '5c']]
        outputs = [resnet.layers[i + 2].output for i, layer in enumerate(resnet.layers) if layer.name in layer_names]
        self.model = keras.models.Model(inputs=[resnet.input], outputs=outputs)

    def get_input(self):
        """
        :return: model input tensor
        """
        return self.model.input

    def get_outputs(self):
        """
        :return: list of model output tensors
        """
        return self.model.outputs

    def get_pyramid_outputs(self):
        """
        :return: list of output tensors representing feature maps pyramid
        """
        return self.get_outputs()[1:]

    @staticmethod
    def get_preprocess_image():
        """
        :return: function for image preprocessing for the model
        """
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input

    @staticmethod
    def get_custom_objects():
        """
        :return: custom objects for model loading
        """
        custom_objects = {
            'UpsampleLike': UpsampleLike,
            'PriorProbability': PriorProbability,
            # 'RegressBoxes': RegressBoxes,
            # 'FilterDetections': FilterDetections,
            # 'Anchors': Anchors,
            # 'ClipBoxes': ClipBoxes,
            '_smooth_l1': smooth_l1(),
            '_focal': focal(),
            '_huber_': huber_loss(1.0)

        }
        return custom_objects


class CustomResNetBackBone:
    def __init__(self):
        import keras_resnet.models
        inputs = keras.layers.Input(shape=(None, None, 3))
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
        layer_names = ['bn%s_branch2c' % layer_name for layer_name in ['2c', '3d', '4f', '5c']]
        outputs = [resnet.layers[i + 2].output for i, layer in enumerate(resnet.layers) if layer.name in layer_names]
        self.model = keras.models.Model(inputs=[resnet.input], outputs=outputs)
        self.model.load_weights(self.download_imagenet(), by_name=True, skip_mismatch=True)

    @staticmethod
    def download_imagenet():
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

        filename = resnet_filename.format(50)
        resource = resnet_resource.format(50)
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def get_input(self):
        """
        :return: model input tensor
        """
        return self.model.input

    def get_outputs(self):
        """
        :return: list of model output tensors
        """
        return self.model.outputs

    def get_pyramid_outputs(self):
        """
        :return: list of output tensors representing feature maps pyramid
        """
        return self.get_outputs()[1:]

    @staticmethod
    def get_preprocess_image():
        """
        :return: function for image preprocessing for the model
        """
        def do_preprocess(x, mode='caffe'):
            x = x.astype(np.float32)

            if mode == 'tf':
                x /= 127.5
                x -= 1.
            elif mode == 'caffe':
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68

            return x

        return do_preprocess

    @staticmethod
    def get_custom_objects():
        """
        :return: custom objects for model loading
        """
        custom_objects = dict(AppResNetBackBone.get_custom_objects())
        import keras_resnet
        custom_objects.update(keras_resnet.custom_objects)
        return custom_objects


def create_retinanet_train(backbone, num_classes=1, num_anchors=9, feature_size=256, name='retinanet'):
    """
    Create RetinaNet model for training (inference layers not attached)
    :param backbone: ResNet backbone
    :param num_classes: number of classes in the dataset
    :param num_anchors: number of different anchors for each pyramid level
    :param feature_size: number of filters in the regression and classification submodels conv layers
    :param name: name of the model
    :return: A keras.models.Model for RetinaNet.
    """
    pyramid_features = create_pyramid_features(backbone.get_pyramid_outputs(), feature_size)
    regression_model = create_regression_model(num_anchors=num_anchors, num_values=4)  # 4 values for bounding box
    classification_model = create_classification_model(num_classes=num_classes, num_anchors=num_anchors)
    regression_pyramid = keras.layers.Concatenate(axis=1, name='regression')([regression_model(f)
                                                                              for f in pyramid_features])
    classification_pyramid = keras.layers.Concatenate(axis=1, name='classification')([classification_model(f)
                                                                                      for f in pyramid_features])
    return keras.models.Model(inputs=[backbone.get_input()], outputs=[regression_pyramid,
                                                                      classification_pyramid], name=name)


def create_counting_layers(x):
    x = keras.layers.Conv2D(filters=x.shape[-1].value // 32, kernel_size=1, strides=1, activation='relu')(x)
    return keras.layers.Flatten()(x)


def create_retinanet_counting(base_model, freeze_base_model=True):
    """
    Create counting model based on RetinaNet classification head
    :param base_model: base RetinaNet model
    :param freeze_base_model: True if base model should be frozen during training
    :return: counting model
    NOTE. Unlike the pure RetinaNet, the counting model assumes images dimensions of 1280x720
    """

    new_input = keras.layers.Input(shape=(720, 1280, 3))
    if freeze_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    x = keras.models.Model(inputs=base_model.inputs, outputs=base_model.outputs[1])(new_input)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='linear')(x)
    return keras.models.Model(inputs=[new_input], outputs=[x])


def create_pyramid_features(pyramid_outputs, feature_size):
    """
    Create feature extraction layers on top of the feature maps pyramid
    :param pyramid_outputs: feature maps
    :param feature_size: number of filters
    :return:
    """
    c3, c4, c5 = pyramid_outputs
    p5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(c5)
    p5_upsampled = UpsampleLike(name='P5_upsampled')([p5, c4])
    p5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(p5)
    # add P5 elementwise to C4
    p4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(c4)
    p4 = keras.layers.Add(name='P4_merged')([p5_upsampled, p4])
    p4_upsampled = UpsampleLike(name='P4_upsampled')([p4, c3])
    p4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(p4)
    # add P4 elementwise to C3
    p3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(c3)
    p3 = keras.layers.Add(name='P3_merged')([p4_upsampled, p3])
    p3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(p3)
    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    p6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(c5)
    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    p7 = keras.layers.Activation('relu', name='C6_relu')(p6)
    p7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(p7)
    return [p3, p4, p5, p6, p7]
