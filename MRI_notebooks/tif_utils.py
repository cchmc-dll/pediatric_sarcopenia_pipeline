import SimpleITK as sitk
import numpy as np

def normalize_image(image):
    image = sitk.GetArrayFromImage(image)
    mean = image.mean()
    std = image.std()
    #print("orig im mean: ", mean,"orig im std: ", std)
    norm_im = (image - mean) / std
    #print("norm im mean: ", norm_im.mean(),"norm im std: ", norm_im.std())
    return sitk.GetImageFromArray(norm_im)

def normalize_images(images):
    mean = images.mean()
    std = images.std()
    return (images - mean) / std

# v1: def invertimage_excluding_background(MR_nda,l_thresh=-0.5,h_thresh=1.7,seed=[128, 128]):
#v2 def invertimage_excluding_background(MR_nda,l_thresh_p=5,h_thresh_p=32,seed=[128, 128]): 
def invertimage_excluding_background(MR_nda,l_thresh_p=5,h_thresh_p=85,seed=[128, 128]): #v3
    MR = normalize_image(MR_nda)
    
    #** Only for V2 and above
    MR_range = max(MR) - min(MR)
#     print('MR min', min(MR))
#     print('MR max', max(MR))
    l_thresh = MR_range*l_thresh_p/100 + min(MR)
    h_thresh = MR_range*h_thresh_p/100 + min(MR)
#     print("new l_thresh: ", l_thresh)
#     print("new h_thresh: ", h_thresh)
    #**
    #
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