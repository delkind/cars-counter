import argparse

from src.training import train_detection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for training a RetinaNet car detection model.')
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
                       action='store', type=float, default=0)
    parser.add_argument('--random_occlusions', help='augment training set with randomly occluded images',
                        action='store_true')
    parser.add_argument('--steps_per_epoch', help='steps per epoch (if not specified, all the '
                                                  'training set will be iterated once per epoch)',
                        type=int, action='store', default=None)
    args = vars(parser.parse_args())
    print("Starting training with the following parameters:")
    for name, val in args.items():
        print('{}: {}'.format(name, val))
    train_detection(**args)
