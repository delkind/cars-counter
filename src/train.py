import os

import keras

from src.generator import CarsGenerator, CarsDataset
from src.misc import RedirectModel, smooth_l1, focal
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


def normalized_mean_squared_error(y_true, y_pred):
    import keras.backend as K
    return K.mean(K.square(y_pred - y_true), axis=-1) / y_true + 1e-20


def train(dataset_path='../datasets/', batch_size=1, epochs=150, lr=1e-5, start_snapshot=None, validation_split=0.1,
          tensorboard_dir='logs/', custom_resnet=True, augmentation=True, snapshot_path='model_snapshots',
          snapshot_base_name="resnet", validation_set=None, random_occlusions=False, counting_model=True,
          freeze_base_model=True, steps_per_epoch=None):
    dataset = CarsDataset(dataset_path, validation_split=validation_split, validation_set=validation_set)

    backbone = CustomResNetBackBone if custom_resnet else AppResNetBackBone
    if start_snapshot:
        model = keras.models.load_model(start_snapshot, custom_objects=backbone.get_custom_objects())
    else:
        model = create_retinanet_train(backbone())
        if not counting_model:
            model.compile(loss={'regression': smooth_l1(), 'classification': focal()},
                          optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001))

    if counting_model:
        model = create_retinanet_counting(model, freeze_base_model=freeze_base_model)
        model.compile(loss=normalized_mean_squared_error, optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001))

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

    if validation_split > 0:
        train_generator = CarsGenerator(dataset.train,
                                        preprocess_image=backbone.get_preprocess_image(),
                                        transform_generator=transform_generator,
                                        batch_size=batch_size,
                                        image_min_side=720,
                                        image_max_side=1280,
                                        random_occlusions=random_occlusions)
        val_generator = CarsGenerator(dataset.validation,
                                      preprocess_image=backbone.get_preprocess_image(),
                                      batch_size=batch_size,
                                      image_min_side=720,
                                      image_max_side=1280,
                                      random_occlusions=False)
        validation_steps = len(val_generator)
        os.makedirs(snapshot_path, exist_ok=True)
        with open('{}/validation.txt'.format(snapshot_path), "wt") as f:
            for img_path in dataset.validation.keys():
                print(img_path, file=f)

    else:
        train_generator = CarsGenerator(dataset.train,
                                        preprocess_image=backbone.get_preprocess_image(),
                                        transform_generator=random_transform_generator(flip_x_chance=0.5),
                                        batch_size=batch_size,
                                        image_min_side=720,
                                        image_max_side=1280,
                                        random_occlusions=random_occlusions)
        val_generator = None
        validation_steps = None

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


if __name__ == '__main__':
    train(custom_resnet=True, snapshot_base_name='augmented', validation_set='validation.txt')
