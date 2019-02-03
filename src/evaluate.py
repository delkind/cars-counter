import math
import pickle

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from src.utils.compute_overlap import compute_overlap

from src.generator import CarsDataset, read_image_bgr
from src.inference import add_inference
from src.model import CustomResNetBackBone, AppResNetBackBone
from src.utils.image import resize_image


def create_inference_model(trained_model_path, backbone):
    model = keras.models.load_model(trained_model_path, custom_objects=backbone.get_custom_objects())
    return add_inference(model)


def predict_image(model, tup, preprocess_image):
    img_path, gt = tup

    image = read_image_bgr(img_path)

    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale
    preds = [t for t in list(zip(boxes[0], scores[0], labels[0])) if t[1] > 0.5]

    return img_path, (list(zip(*list(zip(*preds))[:-1])), gt)


def draw_box(image, box, color, thickness=2):
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


GREEN = (0, 255, 0)
RED = (255, 0, 0)


def visualize_predictions(img_path, annotations):
    draw = read_image_bgr(img_path)
    preds, gt = annotations
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    for box in gt:
        draw_box(draw, box, color=GREEN, thickness=4)
    for box, score in preds:
        b = box.astype(int)
        draw_box(draw, b, color=RED)
        caption = "{:.3f}".format(score)
        draw_caption(draw, b, caption)
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    return len(preds) - len(gt)


def evaluate(trained_model_path, custom_resnet, dataset_root):
    backbone = CustomResNetBackBone if custom_resnet else AppResNetBackBone
    model = create_inference_model(trained_model_path, backbone)
    eval_set = CarsDataset(dataset_root, 'test')
    return list(map(lambda tup: predict_image(model, tup, preprocess_image=backbone.get_preprocess_image()),
                    eval_set.train.items()))


def filter_bboxes(bboxes):
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


def calculate_errors(predictions, conf_start, conf_end=0):
    if conf_end != 0:
        errors_by_conf = [(confidence, np.array([abs(len([p for p in pred if p[1] > confidence / 100]) - len(gt))
                                                 for image, (pred, gt) in predictions])) for confidence in
                          range(50, 95)]
        return [(confidence, np.mean(errors), np.sqrt(np.mean(errors ** 2))) for (confidence, errors) in errors_by_conf]
    else:
        errors = np.array([abs(len([p for p in pred if p[1] > conf_start / 100]) - len(gt))
                           for image, (pred, gt) in predictions])
        return [(conf_start, np.mean(errors), np.sqrt(np.mean(errors ** 2)))]


def filter_pred(pred):
    image, (pred, gt) = pred

    if gt:
        gt = np.array(gt, dtype=np.float64)
        gt = gt[filter_bboxes(gt), :]

    if pred:
        pred_bboxes = np.array(list(zip(*pred))[0], dtype=np.float64)
        pred_indices = filter_bboxes(pred_bboxes)
        pred = np.array(pred)[pred_indices].tolist()

    return image, (pred, gt)


if __name__ == '__main__':
    # preds = evaluate('./app_resnet_cars_10.h5', custom_resnet=True, dataset_root='../datasets')
    preds = pickle.load(open('resnet_09.h5.results.pkl', 'rb'))

    preds = list(map(filter_pred, preds))

    metrics = calculate_errors(preds, 50, 95)
    for confidence, me, rmse in metrics:
        print("Confidence: {}%, ME: {}, RMSE: {}".format(confidence, me, rmse))

    # visualize 10 top misses
    preds = list(map(lambda tup: (tup[0], ([t for t in tup[1][0] if t[1] > 0.54], tup[1][1])), preds))
    top_misses = list(sorted(preds, key=lambda tup: abs(len([t for t in tup[1][0] if t[1] > 0.54]) - len(tup[1][1])), reverse=True))[:10]
    print(list(map(lambda tup: visualize_predictions(tup[0], tup[1]), top_misses)))
