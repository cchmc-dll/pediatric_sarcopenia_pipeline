import numpy as np
from scipy.ndimage import zoom

from ct_slice_detection.models.detection import build_prediction_model
from ct_slice_detection.utils.testing_utils import preprocess_test_image
from ct_slice_detection.io.preprocessing import preprocess_to_8bit


def make_predictions_for_images(dataset, model_path):
    images = preprocess_images(dataset)
    model = load_model(model_path)
    predictions = make_predictions(model, images)

    from matplotlib import pyplot as plt
    plt.imshow(images[0].reshape(512, 512))
    plt.show()
    import pdb; pdb.set_trace()

def load_model(model_path):
    model = build_prediction_model()
    model.load_weights(model_path)
    return model

def preprocess_images(dataset):
    images, _ = normalise_spacing_and_preprocess(dataset)
    return np.array([preprocess(img) for img in images])

def normalise_spacing_and_preprocess(dataset, new_spacing=1):
    images_norm = []
    slice_loc_norm = []
    for image, l3_location, spacing in zip(dataset['images_s'], dataset['ydata']['A'], dataset['spacings']):
        img = zoom(image, [spacing[2] / new_spacing, spacing[0] / new_spacing])
        images_norm.append(preprocess_to_8bit(img))
        slice_loc_norm.append(int(l3_location * spacing[2] / new_spacing))

    return np.array(images_norm), np.array(slice_loc_norm)

def preprocess(image):
    result = preprocess_test_image(image)
    return result[:, :, np.newaxis]


def make_predictions(model, images):
    return model.predict(images)
