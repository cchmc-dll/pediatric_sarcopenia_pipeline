
# Muscle Segmentation Project

## L3_finder.py
Usage:
```
usage: L3_finder.py [-h] dicom_dir dicom_csv nifti_dir output_path

positional arguments:
  dicom_dir    Root directory containing dicoms in format output by Tim's
               script
  dicom_csv    CSV outlining which series and slices for a subject id
  nifti_dir    Dir for intermediately created niftis
  output_path  output .npz file path

optional arguments:
  -h, --help   show this help message and exit
```

Example:
```
python L3_finder.py ..\Dataset2\ .\dataset_manifests\dataset_2_manifest.csv .\tmp\ .\datasets\ds2_px.npz
```

Currently, this program just reformats our dicom data in the format of the output of Tim's scripts into the structure of the .npz file that the ct-slice-detection programs expect.

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

Currently, this program creates intermediate nifti files along the way to creating the MIP images, and they are stored in the nifti_dir.