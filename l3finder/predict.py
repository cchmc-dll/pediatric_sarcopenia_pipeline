import attr
from matplotlib import pyplot as plt

import sys
from collections import namedtuple
import numpy as np
from skimage.draw import line

from ct_slice_detection.models.detection import build_prediction_model
from ct_slice_detection.utils.testing_utils import predict_slice

from l3finder import preprocess as prep

Output = namedtuple('OutputData', ['prediction', 'image_with_predicted_line'])


@attr.s
class Result:
    prediction = attr.ib()
    display_image = attr.ib()
    input_mip = attr.ib()


def make_predictions_for_sagittal_mips(sagittal_mips, model_path, shape):
    model = load_model(model_path)

    # can't fit every picture in memory, so have to do this unfortunately
    images = np.empty(shape=(len(sagittal_mips), *shape), dtype=np.int8)
    unpadded_heights = np.empty(shape=len(sagittal_mips), dtype=np.int32)
    for index, mip in enumerate(sagittal_mips):
        images[index] = mip.preprocessed_image.pixel_data
        unpadded_heights[index] = mip.preprocessed_image.unpadded_height

    predictions = predict_batch(images, model)

    unpadded_images = [
        undo_padding(pred, height)
        for pred, height
        in zip(predictions, unpadded_heights)
    ]
    display_images = [
        draw_line_on_predicted_image(p, upi[0], sag_mip)
        for p, upi, sag_mip
        in zip(predictions, unpadded_images, sagittal_mips)
    ]

    return [
        Result(prediction, display_image, input_mip)
        for prediction, display_image, input_mip
        in zip(predictions, display_images, sagittal_mips)
    ]


def predict_batch(batch, model):
    predictions = model.predict(batch)

    # removes unnecessary extra axis
    reshaped_preds = predictions.reshape(len(predictions), -1)
    indices_of_maxes = np.argmax(reshaped_preds, axis=1)
    maxes = np.max(reshaped_preds, axis=1)

    return [Prediction(*p) for p in zip(indices_of_maxes, maxes, predictions, batch)]


Prediction = namedtuple(
    'Prediction',
    ['predicted_y_in_px', 'probability', 'prediction_map', 'preprocessed_image']
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
    image = prediction.preprocessed_image
    prediction_map = prediction.prediction_map

    return image[:image_height, :, :], prediction_map[:image_height, :],


def draw_line_on_predicted_image(prediction, unpadded_image, sag_mip):
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
        print(
            "error drawing line on preprocessed_image for:",
            sag_mip.subject_id,
            file=sys.stderr,
        )
        print(
            "shape: {}, prediction: {}\n".format(
                unpadded_image.shape, prediction.predicted_y_in_px
            ),
            file=sys.stderr
        )

    return output



