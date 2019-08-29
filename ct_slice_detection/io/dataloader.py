import os

import imageio
from keras.utils import Sequence
from sklearn.model_selection import ShuffleSplit
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
import imgaug as ia

from ct_slice_detection.core.data_loader import BaseDataLoader
from ct_slice_detection.io.augmentation import *
from ct_slice_detection.io.generators import threadsafe_generator
from ct_slice_detection.io.preprocessing import *


np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def load_data(data_path):
    print('loading data')
    data = np.load(data_path)
    images = data['images_f']
    images_sagittal = data['images_s']
    ydata = data['ydata']
    names = data['names']
    spacings = data['spacings']
    data.close()
    slice_locations = np.zeros_like(names, dtype=np.float)
    n = len(ydata.item())
    for k, v in ydata.item().items():
        slice_locations += v
    slice_locations /= n
    print('done')
    return images, images_sagittal, spacings, slice_locations, names


def normalise_spacing_and_preprocess(images, images_sagittal, slice_locations, spacings, new_spacing=1):
    images_norm = []
    images_s_norm = []
    slice_loc_norm = []
    for image, image_s, loc, s in zip(images, images_sagittal, slice_locations, spacings):
        img = zoom(image, [s[2] / new_spacing, s[0] / new_spacing])
        img_s = zoom(image_s, [s[2] / new_spacing, s[0] / new_spacing])
        images_norm.append(preprocess_to_8bit(img))
        images_s_norm.append(preprocess_to_8bit(img_s))
        slice_loc_norm.append(int(loc * s[2] / new_spacing))

    return np.array(images_norm), np.array(images_s_norm), np.array(slice_loc_norm)


def y_to_keypoint(X_data, y_data):
    """
    convert y location to imgaug keypoint class
    """
    keypoints = []
    for idx in range(len(y_data)):
        x = X_data[idx].shape[1] // 2
        s = X_data[idx].shape + (1,)
        keypoint = ia.KeypointsOnImage([ia.Keypoint(x=x, y=y_data[idx])], shape=s)
        keypoints.append(keypoint)
    return keypoints


def adjust_input_image_size(img, input_size):
    s = img.shape
    new_s = [max(d, input_size[j]) for j, d in enumerate(s)]
    if sum(new_s) != sum(s):
        img = pad_image_to_size(img, img_size=input_size[0:2], mode='constant', loc=[1, 2, 1])
    return img


def create_heatmap(lmap, y, sigma=1.5, radius_prct=0.2, downsample=2):
    lmap[y, :] = 1

    c = lmap.shape[1] // 2
    if c >= 16:
        lmap[:, :c - int(radius_prct * c)] = 0
        lmap[:, c + int(radius_prct * c):] = 0

    # apply blur on subsection of image [-10, 10]
    lmap[max(y - 10, 0):y + 10, :] = gaussian_filter(
        lmap[max(y - 10, 0):y + 10, :],
        max(1.5, sigma))

    lmap = lmap / (lmap.max() + 0.00001)
    return lmap


class ImgSequence(Sequence):
    def __init__(self, x_train, y_train, batch_size=2, img_batch=2, ds=1, rate=0.1, border_shift=10,
                 input_size=[256, 256, 1], do_augment=True, sigma=3, do_flatten=False):
        self.x, self.y = x_train, y_train
        self.batch_size = batch_size
        self.input_size = input_size
        self.ds = ds
        self.rate = rate
        self.img_batch = img_batch
        self.do_augment = do_augment
        self.sigma = sigma
        self.do_flatten = do_flatten
        self.border_shift = border_shift
        self.seq = get_augmentation_sequence()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch_all = []
        y_batch_all = []

        for img, y in zip(batch_x, batch_y):

            anywhere = np.random.rand(1)[0] > self.rate

            img = adjust_input_image_size(img, self.input_size)

            x_batch, y_batch = extract_random_example_array(img,
                                                            example_size=self.input_size[0:2],
                                                            n_examples=self.img_batch,
                                                            loc=[y, img.shape[1] // 2],
                                                            anywhere=anywhere,
                                                            border_shift=self.border_shift)

            y_batch = y_to_keypoint(x_batch, y_batch)

            if self.do_augment:
                seq_det = self.seq.to_deterministic()

                # augment keypoints and images
                x_batch = seq_det.augment_images(x_batch)
                y_batch = seq_det.augment_keypoints(y_batch)

            # generate labelmap from keypoint
            if self.do_flatten:
                labelmap = np.zeros((self.img_batch, self.input_size[0] // self.ds, 1))
            else:
                labelmap = np.zeros((self.img_batch, self.input_size[0] // self.ds, self.input_size[1] // self.ds, 1))
            for j in range(self.img_batch):
                yb = int(y_batch[j].keypoints[0].y) // self.ds
                if yb >= x_batch[j].shape[0] // self.ds or yb <= 0:
                    pass
                else:
                    hmap = create_heatmap(labelmap[j], yb, sigma=self.sigma)
                    labelmap[j] = hmap

            x_batch_all.append(x_batch)
            y_batch_all.append(labelmap)

        x_batch_all = np.expand_dims(np.vstack(x_batch_all), 3) - 128
        y_batch_all = np.vstack(y_batch_all)

        return x_batch_all, y_batch_all


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)

        try:
            self.mode = config.mode
        except:
            self.mode = 'heatmap'

    def get_num_samples(self):
        if self.num_samples is None:
            data = np.load(self.config.dataset_path)
            self.num_samples = data['num_images']
            data.close()
        return self.num_samples

    def load_data(self):
        images, images_sagittal, spacings, slice_locations, names = self.load_and_preprocess()

        if self.config.image_type == 'sagittal':
            self.x_val = images_sagittal
        elif self.config.image_type == 'both':
            self.x_val = [images, images_sagittal]
        else:
            self.x_val = images
        self.y_val = slice_locations
        self.names_val = names
        self.spacings_val = spacings

    def split_data(self, train_idx=None, val_idx=None):

        images, images_sagittal, spacings, slice_locations, names = self.load_and_preprocess()
        if train_idx is None:
            print('random split')
            rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
            for train_idx, val_idx in rs.split(list(range(len(images)))):
                pass
            self.train_idx = np.random.permutation(train_idx)
            self.val_idx = val_idx

        # training
        if self.config.image_type == 'sagittal':
            self.x_train = images_sagittal[train_idx]
        elif self.config.image_type == 'both':
            self.x_train = [images[train_idx], images_sagittal[train_idx]]
        else:
            self.x_train = images[train_idx]
        self.y_train = slice_locations[train_idx]
        self.names_train = names[train_idx]
        self.spacings_train = spacings[train_idx]

        # validation
        if self.config.image_type == 'sagittal':
            self.x_val = images_sagittal[val_idx]
            self.x_val2 = images[val_idx]
        elif self.config.image_type == 'both':
            self.x_val = [images[val_idx], images_sagittal[val_idx]]
        else:
            self.x_val = images[val_idx]
            self.x_val2 = images_sagittal[val_idx]
        self.y_val = slice_locations[val_idx]
        self.names_val = names[val_idx]
        self.spacings_val = spacings[val_idx]

        self.batch_loader = None

        self.train_generator = self.create_generator(self.mode, self.x_train, self.y_train,
                                                     batch_size=self.config.batch_size,
                                                     img_batch=self.config.img_batch_size,
                                                     input_size=self.config.input_shape, ds=self.config.ds_factor,
                                                     do_augment=self.config.do_augment,
                                                     rate=self.config.sampling_rate,
                                                     sigma=self.config.sigma,
                                                     bool_output=self.config.regression_dual_output,
                                                     do_flatten=self.config.flatten_output)
        self.val_generator = self.create_generator(self.mode, self.x_val, self.y_val,
                                                   batch_size=self.config.batch_size,
                                                   img_batch=self.config.img_batch_size,
                                                   do_augment=False, input_size=self.config.input_shape,
                                                   rate=self.config.sampling_rate,
                                                   ds=self.config.ds_factor, sigma=self.config.sigma,
                                                   do_flatten=self.config.flatten_output)

        self.save_train_val_split()

    def get_validation_data(self):
        return self.x_val, self.y_val, self.names_val, self.spacings_val

    def load_and_preprocess(self):
        self.data_path = self.config.dataset_path
        cache_filename = os.path.basename(self.data_path).split('.')[0]
        print(cache_filename)
        if self.config.cache_path is None:
            self.config.cache_path = self.config.model_path

        cache_path = os.path.join(self.config.cache_path, cache_filename + '_s' + str(self.config.input_spacing) \
                                  + '_cache.npz')

        if self.config.use_cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            images = data['images']
            images_sagittal = data['images_sagittal']
            slice_locations = data['slice_locations']
            names = data['names']
            spacings = data['spacings']
        else:
            images, images_sagittal, spacings, slice_locations, names = load_data(self.data_path)
            images, images_sagittal, slice_locations = \
                normalise_spacing_and_preprocess(images, images_sagittal,
                                                 slice_locations, spacings,
                                                 new_spacing=self.config.input_spacing)
            np.savez_compressed(cache_path, images=images, images_sagittal=images_sagittal,
                                slice_locations=slice_locations, names=names, spacings=spacings)

        return images, images_sagittal, spacings, slice_locations, names

    def preview_generator_output(self, num=10):
        gen = self.create_generator(self.mode, self.x_train, self.y_train,
                                    self.config.batch_size,
                                    input_size=self.config.input_shape,
                                    ds=self.config.ds_factor,
                                    sigma=self.config.sigma,
                                    rate=self.config.sampling_rate,
                                    do_augment=self.config.do_augment,
                                    do_flatten=self.config.flatten_output)
        output_path = os.path.join(self.config.model_path, 'input_generator_output')
        os.makedirs(output_path, exist_ok=True)
        i = 0
        while i < num:
            i += 1
            image_batch, y_batch = next(gen)
            for j in range(image_batch.shape[0]):

                img = gray2rgb(image_batch[j, :])
                if self.mode == 'heatmap':

                    lmap = np.expand_dims(zoom(np.squeeze(y_batch[j]), self.config.ds_factor), 2)
                    out = overlay_heatmap_on_image(to256(img), lmap)
                else:
                    out = img.copy()
                    try:
                        y = int(y_batch[j])

                    except:
                        y = int(y_batch[0][j])
                    r = 2
                    if y >= 0 and y < out.shape[0]:
                        out[y - r:y + r, :, 0] = out.max()
                imageio.imwrite(os.path.join(output_path, str(i) + '_' + str(j) + '_out.jpg'), out)

    def create_generator(self, mode='heatmap', x_train=None, y_train=None, batch_size=2, img_batch=3,
                         ds=2, rate=0.1, border_shift=10,
                         input_size=[256, 256, 1], do_augment=True, sigma=1.5, bool_output=False, do_flatten=False):

        if mode == 'heatmap':
            return self.heatmap_generator(x_train, y_train, batch_size, img_batch, ds, rate, border_shift,
                                          input_size, do_augment, sigma, do_flatten)
        else:
            return self.reg_generator(x_train, y_train, batch_size, img_batch, ds, rate, border_shift,
                                      input_size, do_augment, bool_output=bool_output)

    @threadsafe_generator
    def heatmap_generator(self, x_train, y_train, batch_size=2, img_batch=2, ds=1, rate=0.1, border_shift=10,
                          input_size=[256, 256, 1], do_augment=True, sigma=3, do_flatten=False, s_sigma=10):

        num_images = len(x_train)

        seq = get_augmentation_sequence()

        while True:
            s_sigma = max(s_sigma - 1, sigma)
            for l in np.random.permutation(range(0, num_images, batch_size)):
                x_batch_all = []
                y_batch_all = []
                w_batch_all = []
                for i in range(l, min(l + batch_size, num_images)):
                    img = x_train[i].copy()
                    y = y_train[i]

                    img = adjust_input_image_size(img, input_size)

                    anywhere = np.random.rand(1)[0] > rate

                    x_batch, y_batch = extract_random_example_array(img,
                                                                    example_size=input_size[0:2],
                                                                    n_examples=img_batch,
                                                                    loc=[y, img.shape[1] // 2],
                                                                    anywhere=anywhere,
                                                                    border_shift=border_shift)

                    y_batch = y_to_keypoint(x_batch, y_batch)

                    if do_augment:
                        seq_det = seq.to_deterministic()

                        # augment keypoints and images
                        x_batch = seq_det.augment_images(x_batch)
                        y_batch = seq_det.augment_keypoints(y_batch)

                    # generate labelmap from keypoint
                    if do_flatten:
                        labelmap = np.zeros((img_batch, input_size[0] // ds, 1))
                    else:
                        labelmap = np.zeros((img_batch, input_size[0] // ds, input_size[1] // ds, 1))
                    for j in range(img_batch):
                        yb = int(y_batch[j].keypoints[0].y) // ds
                        if yb >= x_batch[j].shape[0] // ds or yb <= 0:
                            pass
                        else:
                            hmap = create_heatmap(labelmap[j], yb, sigma=s_sigma)
                            labelmap[j] = hmap

                    x_batch_all.append(x_batch)
                    y_batch_all.append(labelmap)
                a = np.expand_dims(np.vstack(x_batch_all), 3) - 128

                yield a, np.vstack(y_batch_all)

    @threadsafe_generator
    def reg_generator(self, x_train, y_train, batch_size=2, img_batch=3, ds=1, rate=1, border_shift=10,
                      input_size=[256, 256, 1], do_augment=True, bool_output=False):

        num_images = len(x_train)
        seq = get_augmentation_sequence()

        while True:
            for l in range(0, num_images, batch_size):
                x_batch_all = []
                y_batch_all = []
                y2_batch_all = []
                w_batch_all = []
                for i in range(l, min(l + batch_size, num_images)):
                    img = x_train[i].copy()
                    y = y_train[i]

                    s = img.shape
                    new_s = [max(d, input_size[1]) for d in s]
                    if sum(new_s) != sum(s):
                        img = pad_image_to_size(img, img_size=input_size[0:2], mode='edge')

                    anywhere = np.random.rand(1) > rate

                    x_batch, y_batch = extract_random_example_array(img,
                                                                    example_size=input_size[0:2],
                                                                    n_examples=img_batch,
                                                                    loc=[y, img.shape[1] // 2],
                                                                    anywhere=anywhere,
                                                                    border_shift=1)

                    y_batch = y_to_keypoint(x_batch, y_batch)

                    if do_augment:
                        seq_det = seq.to_deterministic()

                        # augment keypoints and images
                        x_batch = seq_det.augment_images(x_batch)
                        y_batch = seq_det.augment_keypoints(y_batch)

                    # generate labelmap from keypoint
                    inview = np.ones((img_batch, 1))
                    ys = np.zeros((img_batch, 1))
                    for j in range(img_batch):
                        yb = int(y_batch[j].keypoints[0].y) // ds
                        ys[j] = yb
                        if yb >= x_batch[j].shape[0] // ds or yb <= 0:
                            inview[j] = 0

                    x_batch_all.append(x_batch)
                    y_batch_all.append(ys)
                    y2_batch_all.append(inview)
                    # weights = 1.0 / (1 + 0.5 * np.sqrt(0.5 + np.abs(np.squeeze(ys) - input_size[0] // 2)))
                    # w_batch_all.append(weights)

                x_batch_all = np.expand_dims(np.vstack(x_batch_all), 3) - 116.779
                x_batch_all = np.concatenate([x_batch_all] * 3, axis=3)

                if bool_output:
                    yield x_batch_all, [np.vstack(y_batch_all), np.vstack(y2_batch_all)]  # , np.vstack( w_batch_all)
                else:
                    yield x_batch_all, np.vstack(y_batch_all)  # , np.vstack( w_batch_all)


def image_slide_generator(image, label, input_size, start=0, step=10):
    img = image.copy()
    y = label
    s = img.shape
    new_s = [max(d, input_size[1]) for d in s]
    if sum(new_s) != sum(s):
        img = pad_image_to_size(img, img_size=input_size[0:2], mode='edge')
    for i in range(start, img.shape[0] - input_size[0] + 1, step):
        simg = img[i:i + input_size[0], 0:input_size[1]]
        a = np.expand_dims(np.expand_dims(np.array(simg), 0), 3) - 116.779
        yield np.concatenate((a, a, a), axis=3), y - i
