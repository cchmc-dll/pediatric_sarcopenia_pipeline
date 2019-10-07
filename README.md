
# Muscle Segmentation Project

## L3_finder_training_formatter.py
Usage:
```
usage: L3_finder_training_formatter.py [-h] dicom_dir dicom_csv nifti_dir output_path

positional arguments:
  dicom_dir    Root directory containing dicoms in format output by Tim's
               script
  dicom_csv    CSV outlining which series and slices for a subject id
  output_path  output .npz file path

optional arguments:
  -h, --help   show this help message and exit
```

Example:
```
python L3_finder_training_formatter.py ..\Dataset2\ .\dataset_manifests\dataset_2_manifest.csv .\tmp\ .\datasets\ds2_px.npz
```

This program reformats our dicom data in the format of the output of Tim's scripts into the structure of the .npz file that the ct-slice-detection programs expect.

That is:
images_f - frontal MIP images, 2D
images_s - sagittal MIP images, 2D
spacings - the pixel spacings of the ct scans, (x, y, z)
names - the subject names for each image
ydata - the distance to L3 from the top of the image in _pixels_
num_images - the number of images

In the code that looks like:
```python
dict(
    images_f=sagittal_mips,  # for now...
    images_s=sagittal_mips,
    spacings=sagittal_spacings,
    names=names,
    ydata=ydata,
    num_images=len(sagittal_mips)
)
```

The dataset manifest is a csv file that outlines the instance number of the axial CT that corresponds with the L3 vertebra. There is an example in the dataset_manifests folder. The format is like this:

```csv
subject_id,axial_series,axial_l3,sagittal_series,sagittal_midsag
200,4,32,6,30
201,3,19,5,19
202,4,32,6,30
203,3,19,5,19
204,4,29,11.2,27
```

## L3_finder.py
Usage:
```
usage: L3_finder.py [-h] --dicom_dir DICOM_DIR --model_path MODEL_PATH
                    --output_directory OUTPUT_DIRECTORY [--show_plots]
                    [--overwrite] [--save_plots]

optional arguments:
  -h, --help            show this help message and exit
  --dicom_dir DICOM_DIR
                        Root directory containing dicoms in format output by
                        Tim's script
  --model_path MODEL_PATH
                        Path to .h5 model trained using
                        https://github.com/fk128/ct-slice-detection Unet
                        model.
  --output_directory OUTPUT_DIRECTORY
                        Path to directory where output files will be saved.
                        Will be created if it does not exist
  --show_plots          Path to directory where output files will be saved
  --overwrite           Overwrite files within target folder
  --save_plots          If true, will save side-by-side plot of predicted L3
                        and the axial slice at that level
```

The program expects a directory in the format of:
```
dataset/
├── Subject_001
│   └── Accession_XYZ
│       ├── SE-2-Stnd_Pediatric_5.0_CE
│       │   ├── IM-CT-1.dcm
│       │   ├── ...
│       └── SE-6-Stnd_Pediatric_3.0_CE
│       │   ├── IM-CT-1.dcm
|       |   ├── ...
├── Subject_002
...
```

where the names of the directories do not matter except for the subject, where
the string after the last underscore is used as the subject ID. So in this example
we would have subjects 001 and 002, and subject 001 has two image series.

The orientations of the images are determined from the dicoms themeselves. 
Right now, I've only tested with 1 sagittal and 1 axial image series in each accession,
though it might work with more than one.

The output directory should end up with an images folder and csv of the prediction 
results (at l3_prediction_results.csv), and optionally a plots folder.

The program can also show the plots immediately if you'd like.
