import sys
from collections import namedtuple

from skimage.draw import line_aa
from tqdm import tqdm


from ct_slice_detection.models.detection import build_prediction_model
from ct_slice_detection.utils.testing_utils import predict_slice

Prediction = namedtuple(
    'Prediction',
    ['predicted_y_in_px', 'probability', 'prediction_map', 'image']
)

Output = namedtuple('OutputData', ['prediction', 'image_with_predicted_line'])
Result = namedtuple('Result', ['prediction', 'display_image'])


def make_predictions_for_images(preprocessed_images, model_path):
    model = load_model(model_path)
    for image, unpadded_height in tqdm(preprocessed_images):
        prediction = make_prediction(model, image)
        unpadded_image, _ = undo_padding(prediction, unpadded_height)

        display_image = draw_line_on_predicted_image(
            prediction,
            unpadded_image,
            console=tqdm
        )
        yield Result(prediction, display_image)


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

    return image[:, :image_height, :, :], prediction_map[:image_height, :],


def draw_line_on_predicted_image(prediction, unpadded_image, console):
    rr, cc, val = line_aa(
        r0=prediction.predicted_y_in_px,
        c0=0,
        r1=prediction.predicted_y_in_px + 1,
        c1=unpadded_image.shape[2] - 1
    )
    output = unpadded_image.reshape(
        unpadded_image.shape[1], unpadded_image.shape[2]
    )

    try:
        output[rr, cc] = val * 255
    except IndexError:
        console.write("error drawing line on image for:", file=sys.stderr)
        console.write(
            "shape: {}, prediction: {}\n".format(
                unpadded_image.shape, prediction.predicted_y_in_px
            ),
            file=sys.stderr
        )
    finally:
        return output



