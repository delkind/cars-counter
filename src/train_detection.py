import argparse

from src.training import train_detection, add_common_arguments, parse_and_print_args


def main():
    parser = argparse.ArgumentParser(description='Training script for training a RetinaNet car detection model.')
    add_common_arguments(parser)
    print("Starting training with the following parameters:")
    args = parse_and_print_args(parser)
    train_detection(**args)


if __name__ == '__main__':
    main()
