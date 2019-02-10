import argparse
import pickle

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, '.')

from src.generator import CarsDataset, read_image_bgr
from src.inference import add_inference
from src.model import CustomResNetBackBone, AppResNetBackBone
from src.training import parse_and_print_args
from src.utils.image import resize_image


def predict_image(model, img_path, gt, preprocess_image):
    """
    Make prediction for a single image
    :param model: the predicting model
    :param img_path: path to the image
    :param gt: ground truth
    :param preprocess_image: preprocess image function
    :return: (image_path, prediction, ground_truth) tuple
    """
    image = read_image_bgr(img_path)

    image = preprocess_image(image)
    image, scale = resize_image(image)
    preds = model.predict_on_batch(np.expand_dims(image, axis=0))

    # Process detected object locations
    if len(preds) > 1:
        # correct for image scale
        boxes, scores, labels = preds
        boxes /= scale
        preds = [(box, score, label) for box, score, label in list(zip(boxes[0], scores[0], labels[0])) if score > 0.5]

    return img_path, (preds, gt)


def visualize_predictions(img_path, annotations):
    """
    Visualize prediction for an image - draw image, green boxes for ground truth and red for predicted
    :param img_path: path to the image
    :param annotations: (prediction, ground_truth) tuple
    :return:
    """

    def draw_box(image, box, color, thickness=2):
        b = np.array(box).astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    green = (0, 255, 0)
    red = (255, 0, 0)

    draw = read_image_bgr(img_path)
    preds, gt = annotations
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    for box in gt:
        draw_box(draw, box, color=green, thickness=4)
    if len(preds[0]) > 1:
        for box, score, _ in preds:
            b = box.astype(int)
            draw_box(draw, b, color=red)
            caption = "{:.3f}".format(score)
            draw_caption(draw, b, caption)
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


def fit_confidence_threshold(model, dataset_root, validation_set, preprocess_image):
    """
    Use validation set to determine the confidence threshold that yields the best RMSE
    :param model: model used for predictions
    :param dataset_root: root directory of the dataset
    :param validation_set: path to validation set file
    :param preprocess_image: preprocess image function
    :return: fitted confidence
    """
    if validation_set is None:
        raise Exception("Error: validation set is required to fit the confidence threshold")

    eval_set = CarsDataset(dataset_root, validation_set=validation_set)
    print("Fitting confidence threshold, {} items in validation set...".format(len(eval_set.validation)))
    preds = list(map(lambda tup: predict_image(model, tup[0], tup[1], preprocess_image=preprocess_image),
                     eval_set.validation.items()))
    errors_by_conf = [(confidence / 100, np.array([abs(len([p for p in pred if p[1] > confidence / 100]) - len(gt))
                                                   for image, (pred, gt) in preds])) for confidence in range(50, 99)]
    metrics = [(confidence, np.sqrt(np.mean(errors ** 2))) for (confidence, errors) in errors_by_conf]

    index = np.argmin(list(zip(*metrics))[1])

    print("Fitted {}".format(metrics[index][0]))

    return metrics[index][0]


def predict_dataset(model_path, custom_resnet, dataset_root, validation_set=None, confidence_threshold=None):
    """
    Predict results for the dataset
    :param model_path: path to the model for prediction
    :param custom_resnet: True if the model is using custom (keras_resnet) backbone
    :param dataset_root: root directory of the datasets
    :param validation_set: path to the validation set file
    :param confidence_threshold: confidence threshold (None to fit)
    :return: prediction results list
    """
    backbone = CustomResNetBackBone if custom_resnet else AppResNetBackBone

    model = keras.models.load_model(model_path, custom_objects=backbone.get_custom_objects())
    if len(model.outputs) > 1:
        model = add_inference(model)

    if len(model.outputs) > 1 and confidence_threshold is None:
        confidence_threshold = fit_confidence_threshold(model, dataset_root, validation_set,
                                                        backbone.get_preprocess_image())

    eval_set = CarsDataset(dataset_root, 'test')
    return list(
        (map(lambda tup: predict_image(model, tup[0], tup[1], preprocess_image=backbone.get_preprocess_image()),
             eval_set.train.items()))), confidence_threshold


def translate_to_count(predictions, confidence_threshold):
    """
    Translate detection model results to counts
    :param predictions: predicted detections
    :param confidence_threshold: confidence threshold
    :return: list of count predictions
    """
    if len(predictions[0][1][0]) > 1:
        return [(image, (len([p for p in pred if p[1] > confidence_threshold]), gt)) for image, (pred, gt)
                in predictions]
    else:
        return [(image, (int(pred[0][0] + 0.5), gt)) for image, (pred, gt) in predictions]


def calculate_errors(predictions):
    """
    Calculate errors for predictions
    :param predictions: predictions list
    :return: lists of errors for the combined dataset, for CARPK and for PUCPR+
    """
    errors = [abs(pred - len(gt)) for image, (pred, gt) in predictions]
    errors_carpk = [abs(pred - len(gt)) for image, (pred, gt) in predictions if 'CARPK' in image]
    errors_pucpr = [abs(pred - len(gt)) for image, (pred, gt) in predictions if 'PUCPR+' in image]
    return errors, errors_carpk, errors_pucpr


def evaluate_results(confidence_threshold, preds, top_misses_to_visualize):
    """
    Evaluate the results list
    :param confidence_threshold: confidence threshold for the detection model
    :param preds: results list
    :param top_misses_to_visualize: number of top misses to visualize (0 to omit visualization)
    """
    counts = translate_to_count(preds, confidence_threshold)
    errors, errors_carpk, errors_pucpr = calculate_errors(counts)
    errs = np.array(errors)
    errs_carpk = np.array(errors_carpk)
    errs_pucpr = np.array(errors_pucpr)
    print("Test set consists of {} samples from CARPK dataset and {} samples from PUCPR+ dataset".format(
        len(errs_carpk), len(errs_pucpr)))
    print("Combined: MAE: {}, RMSE: {}, CARPK: MAE: {}, RMSE: {}, PUCPR+: MAE: {}, RMSE: {}".format(
        np.mean(errs),
        np.sqrt(np.mean(errs ** 2)),
        np.mean(errs_carpk),
        np.sqrt(np.mean(errs_carpk ** 2)),
        np.mean(errs_pucpr),
        np.sqrt(np.mean(errs_pucpr ** 2))
    ))
    # visualize top misses
    preds_dict = dict(preds)
    top_misses = [(image, preds_dict[image]) for image, _ in
                  list(sorted(counts, key=lambda tup: abs(tup[1][0] - len(tup[1][1])),
                              reverse=True))[: top_misses_to_visualize]]
    list(map(lambda tup: visualize_predictions(tup[0], tup[1]), top_misses))


def evaluate(model_path,
             custom_resnet,
             dataset_root,
             validation_set=None,
             confidence_threshold=None,
             results_path=None,
             top_misses_to_visualize=0):
    """
    Evaluate the trained model, print the accuracy metrics and plot top misses
    :param model_path: path to the trained model
    :param custom_resnet: True if custom (keras_resnet) backbone was used for the model
    :param dataset_root: root directory of the datasets
    :param validation_set: path to the validation set file (only for detection model)
    :param confidence_threshold: confidence threshold for the detection model
    :param results_path: path to save the prediction results
    :param top_misses_to_visualize: number of top misses to visualize (0 to omit visualization)
    """
    preds, confidence_threshold = predict_dataset(model_path,
                                                  custom_resnet=custom_resnet,
                                                  dataset_root=dataset_root,
                                                  validation_set=validation_set,
                                                  confidence_threshold=confidence_threshold)
    if results_path:
        with open(results_path, 'wb') as f:
            pickle.dump((preds, confidence_threshold), file=f)

    print("Confidence threshold: {}".format(confidence_threshold))

    evaluate_results(confidence_threshold, preds, top_misses_to_visualize)


def evaluate_saved(results_path, top_misses_to_visualize=0):
    """
    Evaluate saved prediction results
    :param results_path: path to saved results
    :param top_misses_to_visualize: number of top misses to visualize (0 to omit visualization)
    """
    preds, confidence_threshold = pickle.load(open(results_path, 'rb'))
    evaluate_results(confidence_threshold, preds, top_misses_to_visualize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for training a RetinaNet-based car counting model.')
    parser.add_argument('--top_misses_to_visualize',
                        help='number of top misses to visualize (0 to omit visualization)',
                        action='store', type=int, default=0)

    evaluation_type = parser.add_mutually_exclusive_group()
    evaluation_type.add_argument('--model_path', help='path to the trained model', action='store')
    evaluation_type.add_argument('--saved_results_path', help='path to saved prediction results', action='store')

    parser.add_argument('--custom_resnet', help='custom (keras_resnet) backbone was used for the model',
                        action='store_true')
    parser.add_argument('--dataset_root', help='root directory of the datasets', action='store', default=None)
    parser.add_argument('--validation_set', help='path to the validation set file (only for detection model)',
                        action='store', default=None)
    parser.add_argument('--confidence_threshold', help='confidence threshold for the detection model',
                        action='store',
                        type=float, default=None)
    parser.add_argument('--results_path', help='path to save the prediction results', action='store', default=None)
    args = parse_and_print_args(parser)

    if args['model_path'] is None:
        del args['model_path']
        if args['saved_results_path'] is None:
            parser.error('either --model_path or --saved_results_path must be specified')
        else:
            evaluate_saved(results_path=args['saved_results_path'],
                           top_misses_to_visualize=args['top_misses_to_visualize'])
    else:
        del args['saved_results_path']
        evaluate(**args)
