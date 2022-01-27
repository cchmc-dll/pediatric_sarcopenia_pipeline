# import the necessary packages
import os
import tables
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import tables

def normalize_image(image):
    image = sitk.GetArrayFromImage(image)
    mean = image.mean()
    std = image.std()
    #print("orig im mean: ", mean,"orig im std: ", std)
    norm_im = (image - mean) / std
    #print("norm im mean: ", norm_im.mean(),"norm im std: ", norm_im.std())
    return sitk.GetImageFromArray(norm_im)

def _load_l3_tif(imagepath,normalize=True):
    img = sitk.GetArrayFromImage(sitk.ReadImage(imagepath))
    if normalize:
        img = normalie_image(img)
    return img

def invertimage_excluding_background(MR_nda,l_thresh_p=5,h_thresh_p=85,seed=[128, 128]): #v3
    MR = normalize_image(MR_nda)    
    #** Only for V2 and above
    MR_range = max(MR) - min(MR)
    #     print('MR min', min(MR))
    l_thresh = MR_range*l_thresh_p/100 + min(MR)
    h_thresh = MR_range*h_thresh_p/100 + min(MR)
    #    print("new h_thresh: ", h_thresh)
    # Blur using CurvatureFlowImageFilter
    #
    blurFilter = sitk.CurvatureFlowImageFilter()
    blurFilter.SetNumberOfIterations(5)
    blurFilter.SetTimeStep(0.125)
    image = blurFilter.Execute(MR)
    #
    # Set up ConneMRedThresholdImageFilter for segmentation
    #
    segmentationFilter = sitk.ConnectedThresholdImageFilter()
    segmentationFilter.SetLower(float(l_thresh))
    segmentationFilter.SetUpper(float(h_thresh))
    segmentationFilter.SetReplaceValue(1)
    segmentationFilter.AddSeed(seed)
    segmentationFilter.AddSeed([64,128]) # Adding additional seeds for safety
    segmentationFilter.AddSeed([128,64])
    # Run the segmentation filter
    image = segmentationFilter.Execute(image)
    image[seed] = 1
     # Fill holes
    image = sitk.BinaryFillhole(image);
    # Intensity Invert FIlter
    invertfilter = sitk.InvertIntensityImageFilter()
    MR_invert = normalize_image(invertfilter.Execute(MR))
    #     print("MR_invert min: ", min(MR_invert))
    #     print("MR_invert max: ", max(MR_invert))
    # Masking FIlter
    maskingFilter = sitk.MaskImageFilter()
    background_value = min(MR_invert)-0.5
    maskingFilter.SetOutsideValue(background_value)
    #     print("Name: ", maskingFilter.GetName())
    #     print("Masking Value: ", maskingFilter.GetMaskingValue())
    #     print("Outside Value: ", maskingFilter.GetOutsideValue())
    MR_noTable = maskingFilter.Execute(MR_invert,image)
    #     print("MR_noTable min: ", min(MR_noTable))
    return MR_noTable, image

def _load_l3_tif_invert(imagepath):
    img = sitk.ReadImage(imagepath)
    desired_img, mask = invertimage_excluding_background(img)
    return sitk.GetArrayFromImage(desired_img) #, mask, img

def reslice_image_set_TIF(in_files, image_shape, out_files=None, label_indices=None,normalize=True,invert_image=False):
    images = list()
    for f, file in enumerate(in_files):
         # Check if LABEL
        label = False
        image = None
        if f == label_indices:
            label = True
            image = _load_l3_tif(file,False)
            image[image > 0] = 1
        else:
            if invert_image:
                image = _load_l3_tif_invert(file)
            else:
                image = _load_l3_tif(file,normalize)
        # Resize
        if (image_shape[0] != image.shape[0]) or (image_shape[0] != image.shape[1]):
            desired_size = image_shape[0]
            image = resize_pad(image,desired_size,label)
        print('After Resizing image max: ', np.max(image))
        print('After Resizing image min: ', np.min(image))
        images.append(image)

    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images

class mri_TIF_loader:
    def __init__(self,problem_type='Classification',input_images=None,input_shape=(128,128),image_modalities=['T2'],mask=None,slice_number=0,invert_image=False,normalize_on_load=True):
        self.input_images = input_images
        self.input_shape = input_shape
        self.problem_type = problem_type
        self.image_modalities = image_modalities
        self.mask = mask
        self.slice_number = slice_number
        self.invert_image = invert_image # Flag to invert or not invert MRI images
        self.normalize_on_load = normalize_on_load # Flag to normalize input images, works for MR. For CT do not use if HU needs to be preserved! Will not apply for mask/label/truth files
        if self.problem_type is 'Segmentation':
            training_data_files = list()
            subject_ids = list()
            for subject_dir in glob.glob(os.path.join(self.input_images, "*")):
                subject_ids.append(os.path.basename(subject_dir))
                subject_files = list()
                for modality in self.image_modalities + self.mask:
                    subject_files.append(os.path.join(subject_dir, modality + ".tif"))
                training_data_files.append(tuple(subject_files))
            self.data_files = training_data_files
            self.ids = subject_ids
            self.image_data_shape = tuple([0, len(self.image_modalities)] + list(self.input_shape))
            self.truth_data_shape = tuple([0, len(self.mask)] + list(self.input_shape))
            self.n_channels = len(self.image_modalities)
            
            
        elif problem_type is 'Classification':
            training_data_files = list()
            subject_ids = list()
            for classes in glob.glob(os.path.join(self.input_images, "*")):
                for subject_dir in glob.glob(os.path.join(classes, "*")):
                    subject_ids.append(os.path.basename(subject_dir))
                    subject_files = list()
                    for modality in self.image_modalities + self.mask:
                        subject_files.append(os.path.join(subject_dir, modality + ".tif"))
                    training_data_files.append(tuple(subject_files))
            self.data_files = training_data_files
            self.ids = subject_ids
            self.image_data_shape = tuple([0, len(self.image_modalities)+len(self.mask)] + list(self.input_shape))
            self.truth_data_shape = tuple([0,])
            self.n_channels = len(self.image_modalities)+len(self.mask)

        #elif problem_type is 'Regression':
        #    training_data_files = list()
        #    subject_ids = list()
        #    for subject_dir in glob.glob(os.path.join(self.input_images, "*")):
        #        subject_ids.append(os.path.basename(subject_dir))
        #        subject_files = list()
        #        for modality in self.image_modalities:
        #            subject_files.append(os.path.join(subject_dir, modality + ".TIF"))
        #        training_data_files.append(tuple(subject_files))
        #    self.data_files = training_data_files
        #    self.ids = subject_ids
    
    def get_sample_ids(self):
        return self.ids

    def set_sample_ids(self,new_ids):
        self.ids = new_ids

    def load_toHDF5(self,hdf5_file=None,verbose=-1):
		    # initialize the list of features and labels
            n_samples = len(self.ids)
            filters = tables.Filters(complevel=5, complib='blosc')
            image_storage = hdf5_file.create_earray(hdf5_file.root, 'imdata', tables.Float32Atom(), shape=self.image_data_shape, filters=filters, expectedrows=n_samples)
            
            if self.problem_type is "Classification":
                truth_storage =  hdf5_file.create_earray(hdf5_file.root, 'truth', tables.StringAtom(itemsize=15), shape=self.truth_data_shape, filters=filters, expectedrows=n_samples)
            elif self.problem_type is "Segmentation":
                truth_storage =  hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=self.truth_data_shape, filters=filters, expectedrows=n_samples)
           
            # loop over the input images
            for (i, imagePath) in enumerate(self.data_files):
			    # load the image and extract the class label assuming
			    # that our path has the following format:
			    # /path/to/dataset/{class}/{image}.jpg
                if self.problem_type is "Classification":
                    subject_name = imagePath[0].split(os.path.sep)[-2]
                    if subject_name in self.ids:
                        images = self.get_images(in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1,invert_image=self.invert_image,normalize=self.normalize_on_load)
                        label = imagePath[0].split(os.path.sep)[-3]
                        subject_data = [image for image in images]
                        image_storage.append(np.asarray(subject_data)[np.newaxis])
                        truth_storage.append(np.asarray(label)[np.newaxis])
               
                elif self.problem_type is "Segmentation":
                    subject_name = imagePath[0].split(os.path.sep)[-2]

                    if subject_name in self.ids:
                        images = self.get_images(in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1,invert_image=self.invert_image,normalize=self.normalize_on_load)
                        subject_data = [image for image in images]
                        image_storage.append(np.asarray(subject_data[:self.n_channels])[np.newaxis])
                        #DEBUG 
                        #image = np.asarray(subject_data[:self.n_channels])
                        truth_storage.append(np.asarray(subject_data[self.n_channels],dtype=np.uint8)[np.newaxis][np.newaxis])
                  
                #elif self.problem_type is "Regression":
                #    image = cv2.imread(imagePath)
                    
			    # show an update every `verbose` images
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1,len(self.ids)))
            return(image_storage)

    def get_images(self, **loader_args):
        return reslice_image_set_TIF(**loader_args)

