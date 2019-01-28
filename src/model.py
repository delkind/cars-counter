import keras
import keras.models
from keras.applications.resnet50 import ResNet50

from src.misc import PriorProbability, UpsampleLike, smooth_l1, focal


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
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


class ResNetBackBone:
    def __init__(self):
        resnet = ResNet50(include_top=False, weights='imagenet')
        layer_names = ['bn%s_branch2c' % layer_name for layer_name in ['2c', '3d', '4f', '5c']]
        outputs = [resnet.layers[i + 2].output for i, layer in enumerate(resnet.layers) if layer.name in layer_names]
        self.model = keras.models.Model(inputs=[resnet.input], outputs=outputs)

    def get_input(self):
        return self.model.input

    def get_outputs(self):
        return self.model.outputs

    def get_pyramid_outputs(self):
        return self.get_outputs()[1:]


def create_retinanet_train(backbone, num_classes=1, num_anchors=9, feature_size=256, name='retinanet'):
    pyramid_features = create_pyramid_features(backbone.get_pyramid_outputs(), feature_size)
    regression_model = create_regression_model(num_anchors=num_anchors, num_values=4)  # 4 values for bounding box
    classification_model = create_classification_model(num_classes=num_classes, num_anchors=num_anchors)
    regression_pyramid = keras.layers.Concatenate(axis=1, name='regression')([regression_model(f)
                                                                              for f in pyramid_features])
    classification_pyramid = keras.layers.Concatenate(axis=1, name='classification')([classification_model(f)
                                                                                      for f in pyramid_features])
    return keras.models.Model(inputs=[backbone.get_input()], outputs=[regression_pyramid,
                                                                      classification_pyramid], name=name)


def create_pyramid_features(pyramid_outputs, feature_size):
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


def retinanet_custom_objects():
    return {
        'UpsampleLike': UpsampleLike,
        'PriorProbability': PriorProbability,
        # 'RegressBoxes': RegressBoxes,
        # 'FilterDetections': FilterDetections,
        # 'Anchors': Anchors,
        # 'ClipBoxes': ClipBoxes,
        '_smooth_l1': smooth_l1(),
        '_focal': focal(),

    }

