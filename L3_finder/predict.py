from matplotlib import pyplot as plt

import sys
from collections import namedtuple
import numpy as np
from skimage.draw import line
from tqdm import tqdm

from ct_slice_detection.models.detection import build_prediction_model
from ct_slice_detection.utils.testing_utils import predict_slice
from util.iterable import batch_to_ndarray

Output = namedtuple('OutputData', ['prediction', 'image_with_predicted_line'])
Result = namedtuple('Result', ['prediction', 'display_image'])


def make_predictions_for_images(preprocessed_images, model_path, shape):
    model = load_model(model_path)

    # can't fit every picture in memory, so have to do this unfortunately
    image_gen = []
    unpadded_height_gen = []

    for i in preprocessed_images:
        image_gen.append(i.pixel_data)
        unpadded_height_gen.append(i.unpadded_height)

    batches = zip(
        batch_to_ndarray(
            iterable=image_gen,
            batch_size=512,
            item_shape=shape,
        ),
        batch_to_ndarray(
            iterable=unpadded_height_gen,
            batch_size=512,
            item_shape=(),
            dtype=np.int32,
        )
    )

    for images, unpadded_heights in batches:
        predictions = predict_batch(images, model)

        unpadded_images = [
            undo_padding(pred, height)
            for pred, height
            in zip(predictions, unpadded_heights)
        ]
        display_images = [
            draw_line_on_predicted_image(p, upi[0])
            for p, upi
            in zip(predictions, unpadded_images)
        ]

        yield from (
            Result(prediction, display_image)
            for prediction, display_image
            in zip(predictions, display_images)
        )

    # for image, unpadded_height in tqdm(preprocessed_images):
        # prediction = make_prediction(model, image)
        # unpadded_image, _ = undo_padding(prediction, unpadded_height)

        # display_image = draw_line_on_predicted_image(
            # prediction,
            # unpadded_image,
            # console=tqdm
        # )
        # yield Result(prediction, display_image)

def predict_batch(batch, model):
    predictions = model.predict(batch)

    # removes unnecessary extra axis
    reshaped_preds = predictions.reshape(len(predictions), -1)
    indices_of_maxes = np.argmax(reshaped_preds, axis=1)
    maxes = np.max(reshaped_preds, axis=1)

    return [Prediction(*p) for p in zip(indices_of_maxes, maxes, predictions, batch)]


Prediction = namedtuple(
    'Prediction',
    ['predicted_y_in_px', 'probability', 'prediction_map', 'image']
)


def load_model(model_path):
    model = build_prediction_model()
    model.load_weights(model_path)
    return model


def make_prediction(model, image):
    shape = image.shape  # assuming 512x512x1 for now

    # dataset factor??? From ct-slice-detection
    return Prediction(
        *predict_slice(model, image.reshape(shape[0], shape[1]), ds=1))


def undo_padding(prediction, image_height):
    image = prediction.image
    prediction_map = prediction.prediction_map

    # for image, first index is actually the column...
    return image[:image_height, :, :], prediction_map[:image_height, :],


def draw_line_on_predicted_image(prediction, unpadded_image):
    rr, cc = line(
        r0=prediction.predicted_y_in_px,
        c0=0,
        r1=prediction.predicted_y_in_px,
        c1=unpadded_image.shape[1] - 1
    )
    output = unpadded_image.reshape(
        unpadded_image.shape[0], unpadded_image.shape[1]
    )
    try:
        output[rr, cc] = np.iinfo(output.dtype).max
    except IndexError:
        print("error drawing line on image for:", file=sys.stderr)
        print(
            "shape: {}, prediction: {}\n".format(
                unpadded_image.shape, prediction.predicted_y_in_px
            ),
            file=sys.stderr
        )

    return output



