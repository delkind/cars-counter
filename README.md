# Sources for Cars Counting challenge (Computer Vision course, Open University)

## Overview
The sources are intended to allow re-training the model as well as using the pre-trained model to run inference 
on test sets and compute the accuracy metrics (MAE and RMSE)

## Setting up
1. Clone the repository: <br>
`git clone https://github.com/delkind/cars-counter.git`<br>
`cd cars-counter`
1. Create and activate Python 3 virtual environment (note that TensorFlow does not yet support Python 3.7, si it is advisable to use Python 3.6 or lower)
`virtualenv --python=python3 .env`
`source .env/bin/activate`
1. Perform the setup<br> 
`pip install .`<br>
`python setup.py build_ext --inplace`

## Reproducing training
###RetinaNet model (detection)
Run the following command:<br>
`src/train_detection.py --dataset_path .env/datasets/ --custom_resnet --augmentation --tensorboard_dir /content/logs/
--snapshot_path './model_snapshots/' --snapshot_base_name counter`

###Counter model (regression)
Run the following command:<br>
`src/train_counting.py --dataset_path .env/datasets/ --custom_resnet --augmentation ----retinanet_snapshot .env/models/retinanet_augmented_data.h5 --tensorboard_dir /content/logs/
--snapshot_path './model_snapshots/' --snapshot_base_name counter`

