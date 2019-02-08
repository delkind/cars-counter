import os

import keras

from src.generator import CarsGenerator, CarsDataset
from src.misc import RedirectModel, smooth_l1, focal, huber_loss
from src.model import create_retinanet_train, CustomResNetBackBone, AppResNetBackBone, create_retinanet_counting
from src.utils.transform import random_transform_generator


def create_callbacks(model,
                     batch_size,
                     snapshot_path='./',
                     tensorboard_dir='logs/',
                     snapshot_name_base="custom"):
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

    # Tensorboard callback - log data for tensorboard
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
    # we want to save the whole model, so we use RedirectModel callback to redirect
    # saving to the whole model
    if snapshot_path:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(snapshot_path, exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                snapshot_path,
                '{base}_{{epoch:02d}}.h5'.format(base=snapshot_name_base)
            ),
            verbose=1,
            save_best_only=True,
            monitor="loss",
            mode='min'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    # Reduce learning rate by 10 if no improvement in loss is seen for 2 epochs
    # improvements less than 0.001 are considered insignificant and ignored
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


def create_retinanet_model(backbone, start_snapshot):
    """
    Create RetinaNet model for detection
    :param backbone: Backbone class to use (can be either AppResNetBackbone to use keras.applications
        or CustomResNetBackBone to use keras_resnet)
    :param start_snapshot: path to saved model start from (to resume interrupted training)
    :return: the model
    """
    if start_snapshot:
        model = keras.models.load_model(start_snapshot, custom_objects=backbone.get_custom_objects())
    else:
        model = create_retinanet_train(backbone())
    return model


def train_detection(dataset_path='../datasets/', batch_size=1, epochs=150, lr=1e-5, start_snapshot=None,
                    validation_split=0.1,
                    tensorboard_dir='logs/', custom_resnet=True, augmentation=True, snapshot_path='model_snapshots',
                    snapshot_base_name="resnet", validation_set=None, random_occlusions=False,
                    steps_per_epoch=None):
    """
    Create and train detection model
    :param dataset_path: path to the cars datasets
    :param batch_size: training batch size (usually 1, higher than 5 causes OOM on Colab)
    :param epochs: number of epochs
    :param lr: learning rate
    :param start_snapshot: path to saved model start from (to resume interrupted training)
    :param validation_split: fraction of the training set to put aside as validation set, 0 for no validation
    :param tensorboard_dir: path to directory where tensorboard logs should be placed
    :param custom_resnet: whether to use custom (keras_resnet) or applications (keras.application) resnet implementation
    :param augmentation: True to perform data augmentations
    :param snapshot_path: path where snapshots are saved
    :param snapshot_base_name: base name template for model snapshots saved during training
    :param validation_set: text file specifying image names that should be used as validation set. If this is not None,
        validation_split is ignored
    :param random_occlusions: True to augment training set with randomly occluded images
    :param steps_per_epoch: Number of steps per training epoch
    """
    backbone = CustomResNetBackBone if custom_resnet else AppResNetBackBone
    model = create_retinanet_model(backbone, start_snapshot)

    model.compile(loss={'regression': smooth_l1(), 'classification': focal()},
                  optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001))

    initiate_training(augmentation, backbone, batch_size, dataset_path, epochs, model, random_occlusions,
                      snapshot_base_name, snapshot_path, steps_per_epoch, tensorboard_dir, validation_set,
                      validation_split, counting_model=False)


def train_counting(dataset_path='../datasets/', batch_size=1, epochs=150, lr=1e-5, start_snapshot=None,
                   validation_split=0.1, retinanet_snapshot=None,
                   tensorboard_dir='logs/', custom_resnet=True, augmentation=True, snapshot_path='model_snapshots',
                   snapshot_base_name="resnet", validation_set=None, random_occlusions=False,
                   freeze_base_model=True, steps_per_epoch=None):
    """
    Train the counting model based on ResNet
    :param dataset_path: path to the cars datasets
    :param batch_size: training batch size (usually 1, higher than 5 causes OOM on Colab)
    :param epochs: number of epochs
    :param lr: learning rate
    :param start_snapshot: path to saved model start from (to resume interrupted training)
    :param validation_split: fraction of the training set to put aside as validation set, 0 for no validation
    :param retinanet_snapshot: Path to the RetinaNet snapshot to use (if none specified, new RetinaNet will be built)
    :param tensorboard_dir: path to directory where tensorboard logs should be placed
    :param custom_resnet: whether to use custom (keras_resnet) or applications (keras.application) resnet implementation
    :param augmentation: True to perform data augmentations
    :param snapshot_path: path where snapshots are saved
    :param snapshot_base_name: base name template for model snapshots saved during training
    :param validation_set: text file specifying image names that should be used as validation set. If this is not None,
        validation_split is ignored
    :param random_occlusions: True to add randomly occluded images to the training
    :param freeze_base_model: True to freeze base RetinaNet model during training
    :param steps_per_epoch: Number of steps per training epoch
    """
    backbone = CustomResNetBackBone if custom_resnet else AppResNetBackBone
    if start_snapshot:
        model = keras.models.load_model(start_snapshot, custom_objects=backbone.get_custom_objects())
    else:
        model = create_retinanet_model(backbone, start_snapshot=retinanet_snapshot)
        model = create_retinanet_counting(model, freeze_base_model=freeze_base_model)
        model.compile(loss=huber_loss(clip_delta=3), optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001))

    initiate_training(augmentation, backbone, batch_size, dataset_path, epochs, model, random_occlusions,
                      snapshot_base_name, snapshot_path, steps_per_epoch, tensorboard_dir, validation_set,
                      validation_split, counting_model=True)


def initiate_training(augmentation, backbone, batch_size, dataset_path, epochs, model, random_occlusions,
                      snapshot_base_name, snapshot_path, steps_per_epoch, tensorboard_dir, validation_set,
                      validation_split, counting_model):
    """
    Initiate training for the specified model
    :param augmentation: True to perform data augmentations
    :param backbone: backbone class
    :param batch_size: training batch size (usually 1, higher than 5 causes OOM on Colab)
    :param dataset_path: path to the cars datasets
    :param epochs: number of epochs
    :param model: the model to train
    :param random_occlusions: True to add randomly occluded images to the training
    :param snapshot_base_name: base name template for model snapshots saved during training
    :param snapshot_path: path where snapshots are saved
    :param steps_per_epoch: Number of steps per training epoch
    :param tensorboard_dir: path to directory where tensorboard logs should be placed
    :param validation_set: text file specifying image names that should be used as validation set. If this is not None,
        validation_split is ignored
    :param validation_split: fraction of the training set to put aside as validation set, 0 for no validation
    :param counting_model: True if training the counting model
    """
    if augmentation:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5)
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    dataset = CarsDataset(dataset_path, validation_split=validation_split, validation_set=validation_set)

    val_generator = None
    validation_steps = None
    train_generator = CarsGenerator(dataset.train,
                                    preprocess_image=backbone.get_preprocess_image(),
                                    counting_model=counting_model,
                                    transform_generator=transform_generator,
                                    batch_size=batch_size,
                                    image_min_side=720,
                                    image_max_side=1280,
                                    perform_random_occlusions=random_occlusions)

    if dataset.validation:
        val_generator = CarsGenerator(dataset.validation,
                                      preprocess_image=backbone.get_preprocess_image(),
                                      counting_model=counting_model,
                                      batch_size=batch_size,
                                      image_min_side=720,
                                      image_max_side=1280,
                                      perform_random_occlusions=False)
        validation_steps = len(val_generator)
        os.makedirs(snapshot_path, exist_ok=True)
        with open('{}/validation.txt'.format(snapshot_path), "wt") as f:
            for img_path in dataset.validation.keys():
                print(img_path, file=f)

    callbacks = create_callbacks(model,
                                 batch_size=batch_size,
                                 tensorboard_dir=tensorboard_dir,
                                 snapshot_path=snapshot_path,
                                 snapshot_name_base=snapshot_base_name)
    if steps_per_epoch is None:
        steps_per_epoch = len(train_generator)

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=1,
        workers=1,
        use_multiprocessing=True,
        max_queue_size=10
    )


def parse_and_print_args(parser):
    args = vars(parser.parse_args())
    print("Starting training with the following parameters:")
    for name, val in args.items():
        print('{}: {}'.format(name, val))
    return args


def add_common_arguments(parser):
    parser.add_argument('--dataset_path', help='path to the cars datasets', action='store', required=True)
    parser.add_argument('--batch_size', help='training batch size (usually 1, higher than 5 causes OOM on Colab)',
                        action='store', type=int, default=1)
    parser.add_argument('--epochs', help='number of epochs', action='store', type=int, default=50)
    parser.add_argument('--lr', help='learning rate', action='store', type=float, default=1e-5)
    parser.add_argument('--start_snapshot', help='path to saved model start from (to resume interrupted training)',
                        action='store', default=None)
    parser.add_argument('--tensorboard_dir', help='path to directory where tensorboard logs should be placed',
                        action='store', default='./logs/')
    parser.add_argument('--custom_resnet', help='whether to use custom (keras_resnet) or applications '
                                                '(keras.application) resnet implementation', action='store_true')
    parser.add_argument('--augmentation', help='perform data augmentation', action='store_true')
    parser.add_argument('--snapshot_path', help='path where snapshots are to be saved',
                        action='store', default='./snapshots/')
    parser.add_argument('--snapshot_base_name', help='base name template for model snapshots saved during training',
                        action='store', default='resnet')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--validation_set', help='text file specifying image names that should be used as validation '
                                                'set.', action='store', default=None)
    group.add_argument('--validation_split',
                       help='fraction of the training set to put aside as validation set, 0 for no validation',
                       action='store', type=float, default=0.1)
    parser.add_argument('--random_occlusions', help='augment training set with randomly occluded images',
                        action='store_true')
    parser.add_argument('--steps_per_epoch', help='steps per epoch (if not specified, all the '
                                                  'training set will be iterated once per epoch)',
                        type=int, action='store', default=None)
