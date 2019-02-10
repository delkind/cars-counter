import os
import random
import warnings

import keras
import numpy as np
from PIL import Image
from src.utils.compute_overlap import compute_overlap

from src.anchors import anchor_targets_bbox, guess_shapes, anchors_for_shape
from src.utils.image import adjust_transform_for_image, apply_transform, resize_image, TransformParameters
from src.utils.transform import transform_aabb, random_occlusions


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


class CarsDataset:
    """
    Dataset processing - this class processes the dataset
    Tailored specifically for processing the PUCPR+ and CARPK datasets
    """

    def __init__(self, base_dir, annotation='train', validation_split=0, validation_set=None, balance_datasets=False):
        """
        Initialize the dataset - process annotation files and build batches
        :param base_dir: directory where both datasets reside. Certain directory structure is assumed
        :param annotation: text/train - which annotations to use
        :param validation_split: fraction of dataset to use
        :param validation_set: file containing set of images that should constitute validation set
            if this parameter is specified, validation_split parameter is ignored
        :param balance_datasets: True to repeat the same image 9 times in PUCPR+ dataset to balance its size with CARPK
            since all the images are randomly warped, the exact same image is not supposed to appear many times
        """
        # create dictionary containing both datasets parse results
        images = self._process_dataset_(base_dir + '/PUCPR+_devkit/data', annotation, repeat_times=balance_datasets * 9)
        images += self._process_dataset_(base_dir + '/CARPK_devkit/data', annotation)

        # for each image process the annotations and turn them to list of bounding boxes
        images = [(k, [[int(n) for n in s.split()] for s in v]) for k, v in images]

        # remove invalid bounding boxes
        images = self.clean_data(images)

        if validation_set:
            validation_set = set(open(validation_set, "rt").read().splitlines())
            pure_validation_set = set(os.path.split(img)[1] for img in validation_set)
            pure_images = {os.path.split(img_path)[1]: ann for img_path, ann in images}
            self.validation = dict((img, pure_images[os.path.split(img)[1]]) for img in validation_set)
            self.train = dict([(image, ann) for (image, ann) in images if os.path.split(image)[1] not in
                               pure_validation_set])
        elif validation_split > 0:
            # shuffle to create truly random split
            random.shuffle(images)

            # split the set into training and validation
            split = int(len(images) * (1 - validation_split))
            self.train = dict(images[:split])
            self.validation = dict(images[split:])
        else:
            # use the whole set as training - no validation
            self.train = dict(images)
            self.validation = None

    @staticmethod
    def _process_dataset_(root, annotation, repeat_times=0):
        """
        Process the directory containing the dataset - create dictionary containing images and annotations.
        Cars datasets directory structure assumed
        :param root: root directory of the dataset
        :param annotation: which annotation file to use (train/test)
        :param repeat_times: how many times should the same image be repeated in the dataset. Paths made unique by
            adding '/./' different number of times before the actual filename
        :return: dictionary containing images and respective bounding boxes (pre-cleaning)
        """
        train_set = set(open(root + '/ImageSets/{}.txt'.format(annotation)).read().splitlines())
        images = [
            (root + '/Images/' + img,
             open(root + '/Annotations/' + os.path.splitext(img)[0] + '.txt').read().splitlines())
            for img in os.listdir(root + '/Images') if os.path.splitext(img)[0] in train_set]
        new_images = []
        for i in range(1, repeat_times + 1):
            new_images += [(os.path.split(img_path)[0] + '/.' * i + '/' + os.path.split(img_path)[1], ann)
                           for img_path, ann in images]

        images += new_images
        return images

    @staticmethod
    def filter_bboxes(bboxes):
        """
        Filter invalid bboxes
        :param bboxes: list of bounding boxes
        :return: indices of valid bounding boxes. Boxes with IoU more than 0.5 will be filtered out leaving just one,
        see project report for details.
        """
        overlaps = compute_overlap(bboxes, bboxes)
        for i in range(overlaps.shape[0]):
            overlaps[i, i] = 0
        full_overlap_indices = np.argwhere(overlaps > 0.5)
        if full_overlap_indices.tolist():
            full_overlaps = np.array(list(set([tuple(sorted(pair)) for pair in full_overlap_indices.tolist()])))
            redundant = set(full_overlaps[:, 1].tolist())
            return np.array(list(set(range(0, bboxes.shape[0])) - redundant))
        else:
            return np.arange(bboxes.shape[0])

    @staticmethod
    def clean_data(images):
        """
        Perform the data cleaning
        :param images: images list of tuples (image, annotations)
        :return: list of filtered tuples
        """
        out = []
        for image, annotations in images:
            if annotations:
                annotations = np.array(annotations, dtype=np.float64)
                indices = CarsDataset.filter_bboxes(annotations[:, :-1])
                annotations = annotations[indices, ...].tolist()
            out += [(image, annotations)]
        return out


class CarsGenerator(keras.utils.Sequence):
    """ Generate data for cars datasets.
    """

    def __init__(
            self,
            images,
            preprocess_image,
            counting_model=False,
            transform_generator=None,
            batch_size=1,
            group_method='ratio',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            image_min_side=720,
            image_max_side=1280,
            transform_parameters=None,
            perform_random_occlusions=False
    ):
        """ Initialize a cars data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched.
        """

        self.counting_model = counting_model
        self.image_names = []
        self.image_data = {}

        self.labels = {0: '1'}
        self.classes = {'1': 0}

        self.image_names = list(images.keys())
        self.image_data = {
            img: [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3], 'class': 0}
                  for box in data if box[2] - box[0] > 0 and box[3] - box[1] > 0]
            for img, data in images.items()}
        self.transform_generator = transform_generator
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.transform_parameters = transform_parameters or TransformParameters()
        self.preprocess_image = preprocess_image
        self.random_occlusions = perform_random_occlusions

        # Define groups
        self.groups = self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def size(self):
        """
        :return: number of different images
        """
        return len(self.image_names)

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return self.image_names[image_index]

    def image_aspect_ratio(self, image_index):
        """
        Returns aspect ratio for the image
        :param image_index: index of the image in the list
        :return: ratio
        """
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """
        Load image by index
        :param image_index: index of the image path
        :return: numpy array containing image pixels in BGR format
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """
        Process annotations and create annotation dictionary containing labels and bboxes
        :param image_index: index of image
        :return: annotations dictionary
        """
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [0]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations

    def on_epoch_end(self):
        """
        Shuffle images at the end of every epoch for gradient descent to be truly stochastic
        """
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def load_annotations_group(self, group):
        """
        Perform annotation loading for the whole group
        :param group: group (list)
        :return: annotations for the group
        """
        return [self.load_annotations(image_index) for image_index in group]

    @staticmethod
    def filter_annotations(image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image,
                                                       self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index],
                                                                                             annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                  range(0, len(order), self.batch_size)]

        complete_groups = list(zip(groups, (False,) * len(groups)))

        if self.random_occlusions:
            complete_groups += list(zip(groups, (True,) * len(groups)))

        return complete_groups

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    @staticmethod
    def compute_targets(image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors = anchors_for_shape(max_shape)

        batches = anchor_targets_bbox(
            anchors,
            image_group,
            annotations_group,
            1
        )

        return list(batches)

    def compute_detection_input_output(self, group):
        """ Compute inputs and target outputs batch for the detection (the original RetinaNet) network.
        """
        # load images and annotations
        image_group = self.load_image_group(group[0])
        annotations_group = self.load_annotations_group(group[0])

        if group[1]:
            for image, annotations in zip(image_group, annotations_group):
                random_occlusions(image, annotations['bboxes'])

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def compute_counting_input_output(self, group):
        """ Compute inputs and target outputs batch for the counting network.
        """
        # load images and annotations
        image_group = self.load_image_group(group[0])
        annotations_group = self.load_annotations_group(group[0])

        if group[1]:
            for image, annotations in zip(image_group, annotations_group):
                random_occlusions(image, annotations['bboxes'])

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = [len(a['bboxes']) for a in annotations_group]

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]

        if self.counting_model:
            inputs, targets = self.compute_counting_input_output(group)
        else:
            inputs, targets = self.compute_detection_input_output(group)

        return inputs, targets
