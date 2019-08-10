import numpy as np
import cv2



def normalise(img):
    img = img.astype(np.float16)
    return (img - img.min())/(img.max()-img.min()+1e-7)*255

def to256(img):
    img = img.astype(np.float16)
    return 255*(img - img.min())/(img.max()-img.min()+1e-7)

def mat2gray(img):
    img = img.astype(np.float32)
    return (img - np.min(img))/(np.max(img)-np.min(img)+1e-7)


def blend2d(image, labelmap, alpha):

    # for label in np.unique(labelmap):
    label = 1
    image = np.stack((image,)*3,axis=-1)
    labelmap = np.stack((labelmap,)*3, axis=-1)
    labelmap[:,:,1:2] = 0
    return alpha*labelmap + \
           np.multiply((1-alpha)*mat2gray(image),mat2gray(labelmap==label))\
           + np.multiply(mat2gray(image),1-mat2gray(labelmap==label))


def normalise_image_zero_one(image, eps=1e-7):

    image = image.astype(np.float32)
    v = (image - np.min(image))
    v /= (np.max(image) - np.min(image) + eps)

    return v


def normalise_image_one_one(image):

    v = 2 * normalise_image_zero_one(image) - 1

    return v

def preprocess_to_8bit(img, minv=100, maxv=1500):
    img = np.clip(img, minv, maxv)
    img = 255 * normalise_image_zero_one(img)

    return img


def gray2rgb(img):
    if len(img.shape)==2 or img.shape[2] == 1:
        return np.dstack([img]*3)
    else:
        return img

# def to256(img):
#     return 255 * normalise_image_zero_one(img).astype(np.uint8)


def overlay_heatmap_on_image(img, heatmap):
    pred_map_color = cv2.applyColorMap((255 * (1 - heatmap)).astype(np.uint8), cv2.COLORMAP_JET)
    return (img * (1 - heatmap) + heatmap * pred_map_color).astype(np.uint8)



def extract_random_example_array(image_list, example_size=[64, 64], n_examples=1,
                                 loc=[50, 50], anywhere=False, border_shift=10):
    """
        Randomly extract training examples from image (and corresponding label).
        Returns an image example array and the corresponding label array.

        Note: adapted from https://github.com/DLTK/DLTK
        Parameters
        ----------
        image_list: np.ndarray or list or tuple
            image(s) to extract random patches from
        example_size: list or tuple
            shape of the patches to extract
        n_examples: int
            number of patches to extract in total

        Returns
        -------
        examples
            random patches extracted from bigger images with same type as image_list with of shape
            [batch, example_size..., image_channels]
    """

    def get_range(img_idx):
        if anywhere:
            valid_loc_range = [image_list[img_idx].shape[i] - example_size[i] for i in range(rank)]

            rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_examples)
                       if valid_loc_range[dim] > 0 else np.zeros(n_examples, dtype=int) for dim in range(rank)]
        else:
            low_valid_loc_range = [max(loc[i] - example_size[i ] +border_shift ,0) for i in range(rank)]
            #             high_valid_loc_range = [min(loc[i] + example_size[i]//2,image_list[img_idx].shape[i])
            #                                     for i in range(rank)]
            high_valid_loc_range = \
                [min(loc[i] - border_shift, image_list[img_idx].shape[i] - example_size[i] - border_shift)
                for i in range(rank)]
            rnd_loc = [np.random.randint(low_valid_loc_range[dim], high_valid_loc_range[dim], size=n_examples)
                       if high_valid_loc_range[dim] > low_valid_loc_range[dim] else np.zeros(n_examples, dtype=int)
                       for dim in range(rank)]
        for i in range(0, len(rnd_loc[1])):
            rnd_loc[1][i] = (image_list[img_idx].shape[1] - example_size[1]) // 2

        return rnd_loc

    assert n_examples > 0

    was_singular = False
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        was_singular = True

    assert all([i_s >= e_s for i_s, e_s in zip(image_list[0].shape, example_size)]), \
        'Image must be bigger than example shape'
    assert (image_list[0].ndim - 1 == len(example_size) or image_list[0].ndim == len(example_size)), \
        'Example size doesnt fit image size'

    for i in image_list:
        if len(image_list) > 1:
            assert (
            i.ndim - 1 == image_list[0].ndim or i.ndim == image_list[0].ndim or i.ndim + 1 == image_list[0].ndim), \
                'Example size doesn''t fit image size'
            # assert all([i0_s == i_s for i0_s, i_s in zip(image_list[0].shape, i.shape)]), \
            #     'Image shapes must match'

    rank = len(example_size)

    # extract random examples from image and label


    examples = [[]] * len(image_list)
    y = [0] * n_examples

    for i in range(n_examples):
        rnd_loc = get_range(0)
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim]) for dim in range(rank)]
        y[i] = loc[0] - rnd_loc[0][i]
        #         if y[i] >=100 or y[i] <=28:
        #             y[i] = 0
        #         else:
        #             y[i]= 1
        for j in range(len(image_list)):
            ex_img = image_list[j][slicer][np.newaxis]
            # concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_img), axis=0) if (len(examples[j]) != 0) else ex_img

    if was_singular:
        return examples[0], y
    return examples, y




def pad_image_to_size(image, img_size=[64,64,64],loc=[2,2,2], **kwargs):
    """Image resizing

    Resizes image by cropping or padding dimension to fit specified size.

    Note: adapted from https://github.com/DLTK/DLTK
    Parameters
    ----------
    image : np.ndarray
        image to be resized
    img_size : list or tuple
        new image size
    kwargs
        additional arguments to be passed to np.pad

    Returns
    -------
    np.ndarray
        resized image

    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # find image dimensionality
    rank = len(img_size)

    # create placeholders for new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    for i in range(rank):
        # for each dimensions find whether it is supposed to be cropped or padded
        if image.shape[i] < img_size[i]:
            if loc[i] == 0:
                to_padding[i][0] = (img_size[i] - image.shape[i])
                to_padding[i][1] = 0
            elif loc[i] == 1:
                to_padding[i][0] = 0
                to_padding[i][1] = (img_size[i] - image.shape[i])
            else:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            to_padding[i][0] = 0
            to_padding[i][1] = 0



    # pad the cropped image to extend the missing dimension
    return np.pad(image, to_padding, **kwargs)
