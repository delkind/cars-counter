import os

import keras

from src.generator import CarsGenerator, CarsDataset
from src.misc import RedirectModel, smooth_l1, focal
from src.model import create_retinanet_train, ResNetBackBone, retinanet_custom_objects
from src.utils.transform import random_transform_generator


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


def train(dataset_path='../datasets/',
          batch_size=1,
          epochs=150,
          lr=1e-5,
          start_snapshot=None,
          validation_split=0.1):
    if start_snapshot:
        model = keras.models.load_model(start_snapshot, custom_objects=retinanet_custom_objects())
    else:
        model = create_retinanet_train(ResNetBackBone())
        model.compile(loss={'regression': smooth_l1(), 'classification': focal()},
                      optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001))
    dataset = CarsDataset(dataset_path, validation_split=0.1)
    if validation_split > 0:
        train_generator = CarsGenerator(dataset.train,
                                        transform_generator=random_transform_generator(flip_x_chance=0.5),
                                        batch_size=batch_size,
                                        image_min_side=800,
                                        image_max_side=1333)
        val_generator = CarsGenerator(dataset.validation,
                                      batch_size=batch_size,
                                      image_min_side=800,
                                      image_max_side=1333)
        validation_steps = len(val_generator)
    else:
        train_generator = CarsGenerator(dataset.train,
                                        transform_generator=random_transform_generator(flip_x_chance=0.5),
                                        batch_size=batch_size,
                                        image_min_side=800,
                                        image_max_side=1333)
        val_generator = None
        validation_steps = None

    callbacks = create_callbacks(model, batch_size=batch_size)
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
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
    train()
