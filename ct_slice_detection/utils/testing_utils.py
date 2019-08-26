import os

import imageio
import pandas as pd
import time

from scipy.stats import pearsonr
from scipy.ndimage import zoom

from ct_slice_detection.io.dataloader import image_slide_generator
from ct_slice_detection.io.preprocessing import *
from .generic_utils import printProgressBar


def get_best_loc(loc, height=100,step=1):
    d = height
    s = d-np.array(list(range(0,d,step)))

    max_v = 0
    max_i = 0
    for i in range(len(loc)-d):
        v = pearsonr(loc[i:i+d],s)[0]
        if v > max_v:
            max_v = v
            max_i = i
    return int(max_i+height/2)


def predict_reg(model, image, y, input_shape, start=0, step=1):
    gen = image_slide_generator(image, y, input_shape, start=start, step=step)
    loc = []
    loc_abs = []
    weights = []
    height = input_shape[0]
    mid = height//2
    for i, (image_batch, y_batch) in enumerate(gen):
        preds = model.predict(image_batch[:,:,:,:])
        v = int(preds[0])
        t = y_batch + start + step * i
        loc.append(v)
        # if preds[1] > 0.5:
        # if v > 0 or v < height:
        loc_abs.append(v + start + step * i)
        if len(preds) == 2:
            weights.append(preds[1])
        else:
            weights.append(1)

    # if dual output
    if len(preds) == 2:
        p = np.dot(np.squeeze(np.array(loc_abs)), np.squeeze(np.array(weights)))/np.sum(weights)
    else:
        i_best = get_best_loc(loc, step=step)
        try:
            p = loc_abs[i_best]
        except:
            p = np.mean(loc_abs)
    # avg_pred0 = int(sum(np.array(weights) * np.array(loc_abs)) / sum(weights))
    # avg_pred0 = int(np.array(loc_abs[i_best + height // 3:i_best + 2 * height // 3]).mean())
    return int(p), 1.0  # prob 1 as no prob value



def find_max(img):
    return np.unravel_index(np.argmax(img, axis=None), img.shape)[0]



def place_line_on_img(img, y, pred, r=2):
    if len(img.shape)==2 or img.shape[2] != 3:
        img = np.dstack([img]*3)
    v = img.max()
    img[pred-r:pred+r,:,0] = 0.5*v
    img[y-r:y+r,:,1] = 0.5*v
    return img

def preprocess_test_image(img):
    height = 512
    width = 512
    if img.shape[0] <= height:
        v = height
    else:
        v = 2*height
    img_size = [v, width]
    img = pad_image_to_size(img, img_size, loc=[1,-1], mode='constant')
    return  img[:v, :width] - 128


def predict_slice(model, img, ds):
    img = img[np.newaxis, :, :, np.newaxis]
    preds = model.predict(img)

    m = ds * find_max(preds[0, :]) + ds // 2
    max_pred = preds.max()
    return m, max_pred, preds, img


def predict_and_evaluate(args, test_data, modelwrapper, suffix=''):
    ds = args.ds_factor

    out_path = os.path.join(args.model_path, 'preds')
    os.makedirs(out_path, exist_ok=True)
    df = pd.DataFrame(columns=['y', 'pred_y', 'error_mm', 'error_slice', 'slice_thickness'])
    if args.mode == 'heatmap':
        for i, (image, y, name, spacing) in enumerate(zip(test_data.x_val, test_data.y_val,
                                                          test_data.names_val, test_data.spacings_val)):
            printProgressBar(i, len(test_data.x_val))
            slice_thickness = spacing[2]
            height = image.shape[0]

            preprocessed_image = preprocess_test_image(image)
            pred_y, prob, pred_map, img = predict_slice(modelwrapper.model, preprocessed_image, ds=ds)
            pred_map = np.expand_dims(zoom(np.squeeze(pred_map), ds), 2)

            img = img[:, :height, :, :]
            pred_map = pred_map[:height, :]
            e =  args.input_spacing*abs(pred_y - y)
            e_s = e / slice_thickness
            df.loc[name, 'y'] = y
            df.loc[name, 'pred_y'] = pred_y
            df.loc[name, 'error_mm'] = e
            df.loc[name, 'error_slice'] = e_s
            df.loc[name, 'slice_thickness'] = slice_thickness
            df.loc[name, 'max_prob'] = prob

            sub_dir = os.path.join(out_path, str(5 * (e // 5)))
            os.makedirs(sub_dir, exist_ok=True)

            img = create_output_image(height, image, img, pred_map, pred_y, y)

            imageio.imwrite(os.path.join(sub_dir, str(i) + '_' + str(name) + '_map'+suffix+'.jpg'),
                            np.clip(img, 0, 255).astype(np.uint8))

            # img = place_line_on_img(np.hstack([X[:, :, np.newaxis], X_s[:, :, np.newaxis]]), y, m, r=1)
            # imageio.imwrite(os.path.join(out_path, str(i) + '_' + str(int(max_pred * 100)) + '_otest.jpg'), img)

            df.to_csv(os.path.join(args.model_path, modelwrapper.name + '_preds.csv'))
    else:
        for i, (image, y, name, spacing) in enumerate(zip(test_data.x_val, test_data.y_val,
                                                          test_data.names_val, test_data.spacings_val)):

            printProgressBar(i, len(test_data.x_val))
            slice_thickness = spacing[2]
            height = image.shape[0]
            img = image.copy()
            start_time = time.time()
            pred_y, prob = predict_reg(modelwrapper.model, img, y, args.input_shape)
            end_time = time.time()
            dt = end_time - start_time
            dt/height

            e = args.input_spacing*abs(pred_y - y)
            e_s = e / slice_thickness
            df.loc[name, 'y'] = y
            df.loc[name, 'pred_y'] = pred_y
            df.loc[name, 'error_mm'] = e
            df.loc[name, 'error_slice'] = e_s
            df.loc[name, 'slice_thickness'] = slice_thickness
            df.loc[name, 'max_prob'] = prob
            df.loc[name, 'time'] = dt
            df.loc[name, 'height'] = height

            img = place_line_on_img(gray2rgb(img), y, pred_y, r=1)
            img = to256(img)
            sub_dir = os.path.join(out_path, str(5 * (e // 5)))
            os.makedirs(sub_dir, exist_ok=True)
            imageio.imwrite(os.path.join(sub_dir, str(i) + '_' + name + '_map.jpg'),
                            np.clip(img, 0, 255).astype(np.uint8))

            df.to_csv(os.path.join(args.model_path, modelwrapper.name + '_preds.csv'))


def create_output_image(height, image, unpadded_image, unpadded_pred_map, pred_y, y):
    unpadded_image = to256(unpadded_image)
    if unpadded_pred_map.shape[1] == 1:
        unpadded_pred_map = np.expand_dims(np.concatenate([unpadded_pred_map] * unpadded_image.shape[2], axis=1), 2)
    unpadded_image = overlay_heatmap_on_image(unpadded_image, unpadded_pred_map)
    unpadded_image = np.hstack([unpadded_image[0], gray2rgb(to256(preprocess_test_image(image)[:height, :]))])
    unpadded_image = place_line_on_img(unpadded_image, y, pred_y, r=1)
    return unpadded_image
