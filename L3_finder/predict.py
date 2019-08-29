import sys
from collections import namedtuple

import numpy as np
from scipy.ndimage import zoom
from skimage.draw import line_aa
from toolz import pipe

from ct_slice_detection.models.detection import build_prediction_model
from ct_slice_detection.utils.testing_utils import preprocess_test_image, predict_slice, create_output_image, \
    place_line_on_img
from ct_slice_detection.io.preprocessing import preprocess_to_8bit

Prediction = namedtuple('Prediction', ['predicted_y', 'probability', 'prediction_map', 'image'])
Output = namedtuple('OutputData', ['prediction', 'image_with_predicted_line'])
Result = namedtuple('Result', ['prediction', 'display_image'])


def make_predictions_for_images(dataset, model_path):
    model = load_model(model_path)
    for image, truth, image_height in zip(*preprocess_images(dataset)):
        prediction = make_prediction(model, image)
        unpadded_image, _ = undo_padding(prediction, image_height)
        # display_image = create_output_image(
        #     height=image_height,
        #     image=image.reshape(image.shape[0], image.shape[1]),
        #     unpadded_image=unpadded_image,
        #     unpadded_pred_map=unpadded_pred_map,
        #     y=truth,
        #     pred_y=prediction.predicted_y,
        # )
        display_image = draw_line_on_predicted_image(prediction, unpadded_image)
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


def draw_line_on_predicted_image(prediction, unpadded_image):
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
        print("error drawing line on image for:", file=sys.stderr)
        print(unpadded_image.shape, prediction.predicted_y, file=sys.stderr)
    finally:
        return output





# def create_image_with_lines_from_prediction(prediction, truth):
#     s = prediction.image.shape
#
#     return place_line_on_img(
#         img=prediction.image.reshape(s[1], s[2]),
#         y=truth,
#         pred=prediction.predicted_y,
#         r=1
#     )


#
# pred_y, prob, pred_map, img = predict_slice(modelwrapper.model, image, ds=ds)
# pred_map = np.expand_dims(zoom(np.squeeze(pred_map), ds), 2)
#
# img = img[:, :height, :, :]
# pred_map = pred_map[:height, :]
# e = args.input_spacing * abs(pred_y - y)
# e_s = e / slice_thickness
# df.loc[name, 'y'] = y
# df.loc[name, 'pred_y'] = pred_y
# df.loc[name, 'error_mm'] = e
# df.loc[name, 'error_slice'] = e_s
# df.loc[name, 'slice_thickness'] = slice_thickness
# df.loc[name, 'max_prob'] = prob
#
# sub_dir = os.path.join(out_path, str(5 * (e // 5)))
# os.makedirs(sub_dir, exist_ok=True)
#
# img = to256(img)
#
# if pred_map.shape[1] == 1:
#     pred_map = np.expand_dims(np.concatenate([pred_map] * img.shape[2], axis=1), 2)
# img = overlay_heatmap_on_image(img, pred_map)
# img = np.hstack([img[0], gray2rgb(to256(preprocess_test_image(image)[:height, :]))])
# img = place_line_on_img(img, y, pred_y, r=1)
