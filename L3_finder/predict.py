import sys
from collections import namedtuple

import numpy as np
from scipy.ndimage import zoom
from skimage.draw import line_aa
from tqdm import tqdm

from ct_slice_detection.io.preprocessing import preprocess_to_8bit
from ct_slice_detection.models.detection import build_prediction_model
from ct_slice_detection.utils.testing_utils import preprocess_test_image, predict_slice

Prediction = namedtuple('Prediction', ['predicted_y', 'probability', 'prediction_map', 'image'])
Output = namedtuple('OutputData', ['prediction', 'image_with_predicted_line'])
Result = namedtuple('Result', ['prediction', 'display_image'])


def make_predictions_for_images(dataset, model_path):
    model = load_model(model_path)
    for image, truth, image_height in tqdm(zip(*preprocess_images(dataset))):
        prediction = make_prediction(model, image)
        unpadded_image, _ = undo_padding(prediction, image_height)
        display_image = draw_line_on_predicted_image(prediction, unpadded_image, console=tqdm)
        yield Result(prediction, display_image)


def load_model(model_path):
    model = build_prediction_model()
    model.load_weights(model_path)
    return model


def preprocess_images(dataset):
    images, truths, heights = normalise_spacing_and_preprocess(dataset)
    return np.array([preprocess(img) for img in images]), truths, heights


def normalise_spacing_and_preprocess(dataset, new_spacing=1):
    images_norm = []
    slice_loc_norm = []
    heights = []
    for image, l3_location, spacing in zip(dataset['images_s'], dataset['ydata'].tolist()['A'], dataset['spacings']):
        img = zoom(image, [spacing[2] / new_spacing, spacing[0] / new_spacing])
        images_norm.append(preprocess_to_8bit(img))
        slice_loc_norm.append(int(l3_location * spacing[2] / new_spacing))
        heights.append(img.shape[0])

    return np.array(images_norm), np.array(slice_loc_norm), np.array(heights)


def handle_ydata(ydata):
    """ct-slice-detection application expects ydata to be in this weird
    dict format corresponding to mutltiple people manually finding L3, so this handles that case.
    Numpy doesn't save dictionaries well, and depending on whether you're loading the saved file
    vs generating it live, handles it a little differently"""
    try:
        return ydata.tolist()['A']
    except AttributeError:
        return ydata['A']


def preprocess(image):
    result = preprocess_test_image(image)
    return result[:, :, np.newaxis]


def make_prediction(model, image):
    shape = image.shape  # assuming 512x512x1 for now
    return Prediction(*predict_slice(model, image.reshape(shape[0], shape[1]), ds=1))  # dataset factor??? From ct-slice-detection


def undo_padding(prediction, image_height):
    image = prediction.image
    prediction_map = prediction.prediction_map

    return image[:, :image_height, :, :], prediction_map[:image_height, :],


def draw_line_on_predicted_image(prediction, unpadded_image, console):
    rr, cc, val = line_aa(
        r0=prediction.predicted_y,
        c0=0,
        r1=prediction.predicted_y,
        c1=unpadded_image.shape[2] - 1
    )
    output = unpadded_image.reshape(unpadded_image.shape[1], unpadded_image.shape[2])
    try:
        output[rr, cc] = val * 255
    except IndexError:
        console.write("error drawing line on image for:", file=sys.stderr)
        console.write(
            "shape: {}, prediction: {}\n".format(
                unpadded_image.shape, prediction.predicted_y
            ),
            file=sys.stderr
        )
    finally:
        return output
