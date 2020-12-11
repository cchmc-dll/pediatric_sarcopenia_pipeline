
# Muscle Segmentation Project

Most important files are run_sma_experiment.py and L3_finder.py. These and other files
including those in the util folder are defined in a similar way. There will be a ```main(argv)```
function, a ```parse_args(argv)``` function, top level functions to perform the task, and then
some sort of ```output_XXX(_)``` function. Main and parse_args both take ```sys.argv[1:]``` so
that I can reuse the top level function from the script in a different script if desired.

I tried where possible to not mutate the arguments passed into these top level functions, and just return
new values without changing the input. This style (not mutating state, functional programming)
generally makes it easier to follow along what a program is doing since it isn't changing things from
underneath you while executing.

## Loading the dicoms, l3finder/ingest.py
This module is the module responsible for loading the dicoms into something we can use.
In particular, the ```ImageSeries``` class is very important for this. This class acts
as a go between for the pydicom dataset objects that comprise slices of the series. 
It also defines functions like ```image_index_at_pos``` which is what actually
gives you the particular axial L3 slice for a given y position.  

## Specific patterns I used and why

#### Multiprocessing pool.imap()?
Example:
```python
def load_l3_ndas(l3_images):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        ndas = list(tqdm(pool.imap(_load_l3_pixel_data, l3_images)))
        pool.close()
        pool.join()

    return ndas
```
pool.imap functions similarly to the builtin map function, i.e. applies a function (_load_l3_pixel_data), to
all of the l3_images. I use this function instead of just pool.map so that I can use the tqdm
module, which prints out a nice progress bar. The multiprocessing module allows you to
utilize additional cores instead of just single threading (with some limitations, arguments and return values must be pickleable).

#### @attr.s?
The attrs library lets you define classes more succiently and gives a lot of nice
functionality out of the gate without having to define a ton of methods on your own.
In particular, it implements ```__repr__``` by default which makes it easier to debug
because it shows what the instance variables of a class are by default.

#### @reify?
Whenever the method on the object is called, it computes and stores the value on the object, so that the
next time you call that method the result is already in memory.


## run_sma_experiment.py

This is the script that does the entire SMA experiment, so L3 localizing and muscle segmenting.
It uses configuration .json files to set up arguments. Examples are located in the `config/sma` directory.

Here is an example with explanations for the options.

```json
{
    "l3_finder": {
        "dicom_dir": "Path to parent directory with folders for each subject.",
        "model_path": "Path to the trained model .h5 file to use for l3 localization",
        "output_directory": "Directory where output will go. This must exist before hand",
        "overwrite": true, 
        "save_plots": true,
        "show_plots": "true or false; will show plots with matplotlib",
        "new_tim_dicom_dir_structure": "true or false; New tim dicom structure does not have a folder for the accession within top level subject folder",
        "series_to_skip_pickle_file": "Path to a file with ImageSeries objects pickled that you want to exclude in addition to those excluded by the filters"
    },
    "muscle_segmentor": {
        "model_path": "Path to trained model .h5 file for muscle segmentation",
        "output_directory": "folder where segmentation predictions will go, must exist"
    }
}
```

The "l3_finder" options are just passed to the L3_finder.py script, which runs the L3 finding.
The "muscle_segmentor" options are just used within the run_sma_experiment.py script itself.
Within run_sma_experiment.py, the functions to do segmentation predictions are defined, including
the function to remove the table. 

The most important data structure in this module is the ```SegmentedImages``` class.
The instance variables of this class are numpy arrays that are generated while
performing the segmentation in the ```segment_muscle()``` function.


## L3_finder.py
This program will find series, filter unwanted series out, construct sagittal
series for subjects w/o sagittal images, create sagittal mips,
preprocess the images, and run the predictions.

Contains the L3Image class, which is comprised of an axial series and a sagittal series.
This is the class that manages retrieving the L3 slice from a given axial for a 
predicted l3 location.



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


#### l3finder/ingest.py
This contains the logic to load the images from the folders of dicom that Tim
retrieved from PACS. It also contains the ImageSeries and ConstructedImageSeries
classes, which represent the series as a whole and provide many methods needed
to do the predictions.

Also contains the StudyImageSet class, which was used when formatting the images
for L3 finder training.



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

