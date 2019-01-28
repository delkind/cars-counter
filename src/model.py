import os

from keras.applications.resnet50 import ResNet50
import keras
import keras.models
import tensorflow as tf
import numpy as np
import math

from src.generator import CarsGenerator
from src.utils.callbacks import RedirectModel
from src.utils.transform import random_transform_generator


class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        return np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return tf.image.resize_images(source, (target_shape[1], target_shape[2]),
                                      tf.image.ResizeMethod.NEAREST_NEIGHBOR, False)

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressionModel(keras.models.Model):
    def __init__(self,
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
        super().__init__(inputs=inputs, outputs=outputs, name=name)


class ClassificationModel(keras.models.Model):
    def __init__(self,
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
        super().__init__(inputs=inputs, outputs=outputs, name=name)


class ResNetBackBone(keras.models.Model):
    def __init__(self):
        resnet = ResNet50(include_top=False, weights='imagenet')
        layer_names = ['bn%s_branch2c' % layer_name for layer_name in ['2c', '3d', '4f', '5c']]
        outputs = [resnet.layers[i + 2].output for i, layer in enumerate(resnet.layers) if layer.name in layer_names]
        super().__init__(inputs=[resnet.input], outputs=outputs)

    def pyramid_outputs(self):
        return self.outputs[1:]


class RetinaNetTrain(keras.models.Model):
    def __init__(self, backbone, num_classes=1, num_anchors=9, feature_size=256, name='retinanet'):
        pyramid_features = self.create_pyramid_features(backbone.pyramid_outputs(), feature_size)
        regression_model = RegressionModel(num_anchors=num_anchors, num_values=4)  # 4 values for bounding box
        classification_model = ClassificationModel(num_classes=num_classes, num_anchors=num_anchors)
        regression_pyramid = keras.layers.Concatenate(axis=1, name='regression')([regression_model(f)
                                                                                  for f in pyramid_features])
        classification_pyramid = keras.layers.Concatenate(axis=1, name='classification')([classification_model(f)
                                                                                          for f in pyramid_features])
        super().__init__(inputs=[backbone.input], outputs=[regression_pyramid, classification_pyramid], name=name)

    @staticmethod
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


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5).
                The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def create_callbacks(model,
                     batch_size,
                     snapshot_path='./',
                     backbone='app_resnet',
                     tensorboard_dir='logs/'):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            update_freq=100
        )
        callbacks.append(tensorboard_callback)

    # save the model
    if snapshot_path:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(snapshot_path, exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=backbone,
                                                                    dataset_type='cars')
            ),
            verbose=1,
            save_best_only=True,
            monitor="loss",
            mode='min'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks


BATCH_SIZE = 1


def main():
    prev_path = ''
    custom_objects = {
        'UpsampleLike': UpsampleLike,
        'PriorProbability': PriorProbability,
        # 'RegressBoxes': layers.RegressBoxes,
        # 'FilterDetections': layers.FilterDetections,
        # 'Anchors': layers.Anchors,
        # 'ClipBoxes': layers.ClipBoxes,
        '_smooth_l1': smooth_l1(),
        '_focal': focal(),
    }
    model = keras.models.load_model('', custom_objects=custom_objects)
    model = RetinaNetTrain(ResNetBackBone())
    generator = CarsGenerator('../',
                              transform_generator=random_transform_generator(flip_x_chance=0.5),
                              batch_size=BATCH_SIZE,
                              image_min_side=800,
                              image_max_side=1333)
    steps_per_epoch = len(generator)
    callbacks = create_callbacks(model, batch_size=BATCH_SIZE)
    model.compile(loss={'regression': smooth_l1(), 'classification': focal()},
                  optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001))

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        epochs=150,
        verbose=1,
        workers=1,
        use_multiprocessing=True,
        max_queue_size=10
    )


if __name__ == '__main__':
    main()
