import cv2
import numpy as np
from imgaug import augmenters as iaa


def slice_thickness_func_images(images, random_state, parents, hooks):
    result = []
    for image in images:
        image_aug = augment_slice_thickness(image, max_r=8)
        result.append(image_aug)

    return result


def slice_thickness_func_keypoints(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


def get_augmentation_sequence():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    slice_thickness_augmenter = iaa.Lambda(
        func_images=slice_thickness_func_images,
        func_keypoints=slice_thickness_func_keypoints
    )

    seq = iaa.Sequential([
        sometimes(iaa.Fliplr(0.5)),
        iaa.Sometimes(0.1, iaa.Add((-70, 70))),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}  # scale images to 80-120% of their size, individually per axis
        )),
        # sometimes(iaa.Multiply((0.5, 1.5))),
        # sometimes(iaa.ContrastNormalization((0.5, 2.0))),
        #     sometimes(iaa.Affine(
        #     translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, # translate by -20 to +20 percent (per axis)
        #     rotate=(-2, 2), # rotate by -45 to +45 degrees
        #     shear=(-2, 2), # shear by -16 to +16 degrees
        #     order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        #     cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        #     mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
        #     sometimes(iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05))),
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
        iaa.Sometimes(0.1,
                      iaa.SimplexNoiseAlpha(iaa.OneOf([iaa.Add((150, 255)), iaa.Add((-100, 100))]), sigmoid_thresh=5)),
        iaa.Sometimes(0.1, iaa.OneOf([iaa.CoarseDropout((0.01, 0.15), size_percent=(0.02, 0.08)),
                                      iaa.CoarseSaltAndPepper(p=0.2, size_percent=0.01),
                                      iaa.CoarseSalt(p=0.2, size_percent=0.02)
                                      ])),

        iaa.Sometimes(0.25, slice_thickness_augmenter)

    ])
    return seq


def shift_intensity(img, r):
    return img + np.random.randint(-r, r)


def augment_slice_thickness(image, max_r=5):
    r = np.random.randint(1, max_r + 1)
    return np.expand_dims(cv2.resize(image[::r], image.shape[:2][::-1]), 2)
