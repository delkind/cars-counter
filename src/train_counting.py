import argparse

from src.training import train_counting, add_common_arguments, parse_and_print_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for training a RetinaNet-based car counting model.')
    add_common_arguments(parser)
    parser.add_argument('--retinanet_snapshot', help='for counting model only: path to the RetinaNet snapshot to use '
                                                     '(if none specified, new RetinaNet will be built)',
                        action='store', default=None)
    parser.add_argument('--freeze_base_model', help='for counting model only: freeze base RetinaNet model during '
                                                    'training',
                        action='store_true')
    args = parse_and_print_args(parser)
    train_counting(**args)
