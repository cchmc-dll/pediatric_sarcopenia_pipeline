import csv
import functools
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import toolz
from matplotlib import pyplot as plt
from imageio import imsave
from tables import open_file
from tqdm import tqdm



def output_images(l3_images, args):
    output_dir_path = _ensure_output_dir_exists(args['output_directory'])
    images_dir_path = _ensure_images_dir_exists(output_dir_path)
    csv_path = Path(output_dir_path, 'l3_prediction_results.csv')

    _write_prediction_csv_header(
        csv_path=csv_path, should_overwrite=args['should_overwrite']
    )

    output_pipeline = make_output_pipeline(
        args=args, images_dir_path=images_dir_path, csv_path=csv_path
    )

    image_outputter = functools.partial(_output_image, output_pipeline)
    print("Slow unless axial images already loaded...")

    # with multiprocessing.Pool(48) as pool:
        # # Could use map, but imap lets me get a progress bar
        # l3_images = list(
            # tqdm(
                # pool.imap(image_outputter, l3_images),
                # total=len(l3_images),
            # )
        # )
        # pool.close()
        # pool.join()
    # return l3_images

    return [image_outputter(l3_image) for l3_image in tqdm(l3_images)]


def _output_image(output_pipeline, l3_image):
    try:
        toolz.pipe(l3_image, *output_pipeline)
    except IndexError as e:
        import pdb; pdb.set_trace()
    finally:
        l3_image.free_pixel_data()
    return l3_image


def _ensure_output_dir_exists(output_dir):
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    return path


def _ensure_images_dir_exists(output_dir):
    images_dir_path = Path(output_dir, 'images')
    images_dir_path.mkdir(exist_ok=True)
    return images_dir_path


def _write_prediction_csv_header(csv_path, should_overwrite):
    if not should_overwrite and csv_path.exists():
        raise FileExistsError(csv_path)

    with open(str(csv_path), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'subject_id',
                'predicted_y_in_px',
                'probability',
                'sagittal_series_path',
                'axial_series_path',
            ]
        )


def make_output_pipeline(args, images_dir_path, csv_path):
    pipeline = [
        functools.partial(
            save_l3_image_to_png,
            base_dir=images_dir_path,
            should_overwrite=args['should_overwrite']
        ),
        functools.partial(save_prediction_results_to_csv, csv_path=csv_path)
    ]

    if args['should_plot'] or args['should_save_plots']:
        pipeline.append(
            functools.partial(plot_l3_image, output_args=args)
        )

    return pipeline


def save_l3_image_to_png(l3_image, base_dir, should_overwrite):
    image_dir = _create_directory_for_l3_image(
        base_dir, l3_image, should_overwrite
    )
    return _save_l3_image_to_png(image_dir, l3_image, should_overwrite)


def _create_directory_for_l3_image(base_dir, l3_image, should_overwrite):
    image_dir = Path(base_dir, str(l3_image.subject_id))
    image_dir.mkdir(exist_ok=should_overwrite)  # don't overwrite for now
    return image_dir


def _save_l3_image_to_png(image_dir, l3_image, should_overwrite):
    file_name = 'subject_{subject_id}_IM-CT-{image_number}.png'.format(
        subject_id=l3_image.subject_id,
        image_number=l3_image.prediction_index
    )
    save_path = Path(image_dir, file_name)

    if not should_overwrite and save_path.exists():
        raise FileExistsError(save_path)

    imsave(str(save_path), _make_image_uint16(l3_image.pixel_data))

    return l3_image


def _make_image_uint16(old_image):
    img_min = old_image.min()
    img_max = old_image.max()

    # Make sure we don't overflow
    assert img_max + (-img_min) < 65536

    new_image = old_image - old_image.min()
    return new_image.astype(np.uint16)


def plot_l3_image(l3_image, output_args):
    try:
        _generate_l3_image_figure(l3_image)
    except IndexError as e:
        _generate_l3_prediction_out_of_bounds_figure(l3_image)

    if output_args['should_save_plots']:
        save_plot(
            image=l3_image,
            output_directory=output_args['output_directory'],
            should_overwrite=output_args['should_overwrite']
        )

    if output_args['should_plot']:
        plt.show()

    plt.close()
    return l3_image


def _generate_l3_image_figure(l3_image):
    fig = plt.figure(figsize=(14,10))
    plt.suptitle(_in_bounds_title(l3_image))
    plt.figimage(l3_image.prediction_result.display_image, 25, 25, cmap='bone')
    plt.figimage(l3_image.pixel_data, 600, 25, cmap='bone')
    return fig


def _generate_l3_prediction_out_of_bounds_figure(l3_image):
    print(
        "Prediction: {predicted_y}cm is out of bounds of preprocessed_image for "
        "subject_id: {id_}".format(
            predicted_y=l3_image.prediction_result.prediction.predicted_y_in_px,
            id_=l3_image.axial_series.subject.id_
        )
    )
    plt.title(_out_of_bounds_title(l3_image))
    plt.imshow(l3_image.prediction_result.display_image)


def _in_bounds_title(image):
    title = (
        'Subject: {} - Predicted axial slice (dicom #): {} / {}\n'
        'Predicted L3 in pixels: {}, {}'
        .format(
            image.subject_id, image.prediction_index, image.number_of_axial_dicoms,
            image.predicted_y_in_px, image.height_of_sagittal_image
        )
    )

    return title


def _out_of_bounds_title(image):
    return 'Subject: {} out of bounds prediction: {}'.format(
        image.subject_id, image.prediction_index
    )


def save_plot(image, output_directory, should_overwrite):
    plots_dir = Path(output_directory, 'plots')
    plots_dir.mkdir(exist_ok=should_overwrite)

    file_name = '{}-plot.png'.format(image.subject_id)
    plt.savefig(str(plots_dir.joinpath(file_name)))


def save_prediction_results_to_csv(l3_image, csv_path):
    with open(str(csv_path), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(l3_image.as_csv_row())

    return l3_image


# Potentially will go unused
def output_l3_images_to_h5(l3_images, h5_file_path):
    expanded_path = Path(h5_file_path).expanduser()

    with open_file(str(expanded_path), mode='w') as h5_file:

        first_l3_image = next(l3_images)
        imdata_array = h5_file.create_earray(
            where=h5_file.root,
            name='imdata',
            obj=np.expand_dims(first_l3_image.pixel_data, axis=0)
        )
        plot_l3_image(first_l3_image)
        subject_ids = [first_l3_image.axial_series.subject.id_]

        for l3_image in l3_images:
            plot_l3_image(l3_image)
            imdata_array.append(np.expand_dims(l3_image.pixel_data, axis=0))
            # terrible, but I'm really fighting against PyTables here...
            subject_ids.append(l3_image.axial_series.subject.id_)

        h5_file.create_array(
            where=h5_file.root, name='subject_ids', obj=subject_ids
        )


