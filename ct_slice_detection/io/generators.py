import threading
import numpy as np
from .preprocessing import *
from .augmentation import *
from scipy.ndimage.filters import gaussian_filter


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def generator_lmap(x_train, y_train, batch_size, img_batch=1, ds=16, rate=2, border_shift=10, input_size=[256, 256, 1]):
    SEED = 42
    num_images = len(x_train)

    while True:
        for l in range(0, num_images, img_batch):
            x_batch_all = []
            y_batch_all = []
            w_batch_all = []
            for i in range(l, min(l + img_batch, num_images)):
                img = x_train[i].copy()
                #             img[img>3000] = 3000
                #             img[img<100] = 100
                #             img = 255*normalise_zero_one(img)

                y = y_train[i]

                s = img.shape
                new_s = [max(d, input_size[1]) for d in s]
                if sum(new_s) != sum(s):
                    img = pad_image_to_size(img, img_size=input_size[0:2], mode='constant')

                anywhere = np.random.rand(1) > rate

                x_batch, y_batch = extract_random_example_array(img,
                                                                example_size=input_size[0:2],
                                                                n_examples=batch_size,
                                                                loc=[y, img.shape[1] // 2],
                                                                anywhere=anywhere, border_shift=border_shift)

                for k in range(batch_size):
                    if np.random.rand(1) < 0.5:
                        x_batch[k] = augment_slice_thickness(x_batch[k])
                    x_batch[k] = preprocess(x_batch[k])

                labelmap = np.zeros((batch_size, input_size[0] // ds, input_size[1] // ds, 1))
                for j in range(batch_size):
                    yb = y_batch[j]
                    if yb // ds >= x_batch[j].shape[0] // ds or yb <= 0:
                        pass
                    else:
                        labelmap[j][yb // ds, :] = 1
                        #                     print(labelmap[j].shape)
                        labelmap[j] = labelmap[j] * np.expand_dims(x_batch[j][::ds, ::ds] > -50, 3).astype(np.float32)
                        v = labelmap[j].shape[1] // 2
                        if v >= 16:
                            labelmap[j][:, :v - int(0.2 * v)] = 0
                            labelmap[j][:, v + int(0.2 * v):] = 0
                        labelmap[j] = gaussian_filter(labelmap[j], max(4 / ds * 1.5, 1.5))
                        labelmap[j] = labelmap[j] / (labelmap[j].max() + 0.001)

                    if np.random.rand(1) < 0.6:
                        x_batch[j] = random_occlusion(x_batch[j], 0.5, 2)

                x_batch_all.append(x_batch)
                y_batch_all.append(labelmap)
            a = np.expand_dims(np.vstack(x_batch_all), 3)
            #             a = np.concatenate([a,a,a],axis=3)
            yield a, np.vstack(y_batch_all)

    @threadsafe_generator
    def generator_lmap_dual(x_train, x_train2, y_train, batch_size, img_batch=1, ds=16, rate=2, border_shift=10,
                            input_size=[256, 256, 1]):
        SEED = 42
        num_images = len(x_train)

        while True:
            for l in range(0, num_images, img_batch):
                x_batch_all = []
                x2_batch_all = []
                y_batch_all = []
                w_batch_all = []
                for i in range(l, min(l + img_batch, num_images)):
                    #                 img = x_train[i].copy()
                    img = np.dstack([x_train[i], x_train2[i]])
                    #             img[img>3000] = 3000
                    #             img[img<100] = 100
                    #             img = 255*normalise_zero_one(img)

                    y = y_train[i]

                    s = img.shape
                    #                 new_s = [max(d,input_size[1]) for d in s]
                    #                 if sum(new_s) != sum(s):
                    img = pad_image_to_size(img, img_size=input_size, mode='constant')

                    anywhere = np.random.rand(1) > rate

                    x_batch, y_batch = extract_random_example_array(img,
                                                                    example_size=input_size[0:2],
                                                                    n_examples=batch_size,
                                                                    loc=[y, img.shape[1] // 2],
                                                                    anywhere=anywhere, border_shift=border_shift)
                    x_batch2 = x_batch[:, :, :, 1]
                    x_batch = x_batch[:, :, :, 0]
                    for k in range(batch_size):
                        if np.random.rand(1) < 0.5:
                            x_batch[k] = augment_slice_thickness(x_batch[k])
                        x_batch[k] = preprocess(x_batch[k])
                        x_batch2[k] = preprocess(x_batch2[k])

                    labelmap = np.zeros((batch_size, input_size[0] // ds, input_size[1] // ds, 1))
                    for j in range(batch_size):
                        yb = y_batch[j]
                        if yb // ds >= x_batch[j].shape[0] // ds or yb <= 0:
                            pass
                        else:
                            labelmap[j][yb // ds, :] = 1
                            #                     print(labelmap[j].shape)
                            labelmap[j] = labelmap[j] * np.expand_dims(x_batch[j][::ds, ::ds] > -50, 3).astype(
                                np.float32)
                            v = labelmap[j].shape[1] // 2
                            if v >= 16:
                                labelmap[j][:, :v - int(0.2 * v)] = 0
                                labelmap[j][:, v + int(0.2 * v):] = 0
                            labelmap[j] = gaussian_filter(labelmap[j], 8 / ds * 1.5)
                            labelmap[j] = labelmap[j] / (labelmap[j].max() + 0.001)

                        if np.random.rand(1) < 0.5:
                            x_batch[j] = random_occlusion(x_batch[j])

                    x_batch_all.append(x_batch)
                    x2_batch_all.append(x_batch2)
                    y_batch_all.append(labelmap)
                x_batch_all = np.expand_dims(np.vstack(x_batch_all), 3)
                x2_batch_all = np.expand_dims(np.vstack(x2_batch_all), 3)
                #             a = np.concatenate([a,a,a],axis=3)
                yield [x_batch_all, x2_batch_all], np.max(np.vstack(y_batch_all), axis=2, keepdims=True)




def slide_generator(image, label, input_size, start=0, step=10):

    img = image.copy()
    y = label
    s = img.shape
    new_s = [max(d, input_size[1]) for d in s]
    if sum(new_s) != sum(s):
        img = pad_image_to_size(img, img_size=input_size[0:2], mode='edge')
    for i in range(start, img.shape[0] - input_size[0] + 1, step):
        simg = img[i:i + input_size[0], 0:input_size[1]]
        a = np.expand_dims(np.expand_dims(np.array(simg), 0), 3)
        yield np.concatenate((a, a, a), axis=3), y - i


def generator_reg(x_train, y_train, batch_size, img_batch=2, rate=0.5):
    SEED = 42
    num_images = len(x_train)

    while True:
        for j in range(0, num_images, img_batch):
            x_batch_all = []
            y_batch_all = []
            w_batch_all = []
            in_view_all = []
            for i in range(j, min(j + img_batch, num_images)):
                img = x_train[i].copy()
                #             img[img>3000] = 3000
                #             img[img<100] = 100
                #             img = 255*normalise_zero_one(img)

                y = y_train[i]

                s = img.shape
                new_s = [max(d, input_size[1]) for d in s]
                if sum(new_s) != sum(s):
                    img = pad_image_to_size(img, img_size=input_size[0:2], mode='edge')

                anywhere = np.random.rand(1) > rate
                #             img2 = elastic_transform(img,[20,20],[5,5])

                x_batch, y_batch = extract_random_example_array(img,
                                                                example_size=input_size[0:2],
                                                                n_examples=batch_size,
                                                                loc=[y, img.shape[1] // 2],
                                                                anywhere=anywhere, border_shift=10)
                for k in range(batch_size):
                    x_batch[k] = preprocess(x_batch[k])
                    y = y_batch[k]
                    in_view_all.append(int(y > 0 and y < x_batch[k].shape[0]))

                a = np.expand_dims(np.array(x_batch), 3)
                x_batch_all.extend(a)  # np.concatenate((a,a,a),axis=3))
                y_batch_all.extend(np.array(y_batch))
                #                 if anywhere:
                weights = 1.0 / (1 + 0.5 * np.sqrt(0.5 + np.abs(np.array(y_batch) - input_size[0] // 2)))
                #                 else:
                #                 weights = np.ones_like(y_batch)

                w_batch_all.extend(weights)
            yield np.array(x_batch_all), [np.array(y_batch_all).reshape((-1, 1, 1, 1)), np.array(in_view_all).reshape(
                (-1, 1, 1, 1))]  # , [np.array(w_batch_all), np.ones_like(w_batch_all)]
