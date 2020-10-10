#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
#import nibabel as nib
import SimpleITK as sitk
import pandas as pd
#import gui
from enum import Enum
from shutil import copyfile

class OverlapMeasures(Enum):
    MV_dice, MV_volume_similarity,truth_area,MV_area,MV_reldiff = range(5)



def get_area(img):
    statistics_image_filter = sitk.StatisticsImageFilter()
    #segmented_surface = sitk.LabelContour(img)
    #statistics_image_filter.Execute(segmented_surface)
    statistics_image_filter.Execute(img)
    area= int(statistics_image_filter.GetSum())
    return area

def check_make_folder(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError("Input is directory; output name exists but is not a directory")
    else:  # out_dir does not exist; create it.
        os.makedirs(path)


def write_output(path,pat,MV_img,truth):
    check_make_folder(path)
    outpath = os.path.join(path,'mean_result',pat)
    check_make_folder(outpath)
    
    # Copy Truth and CT from fold 0 folders
    inpath  = os.path.join(path,'fold0',pat)
    CT = inpath+'/data_CT.TIF'
    #copyfile(truth, os.path.join(outpath,'truth.TIF'))
    copyfile(CT, os.path.join(outpath,'CT.TIF'))
    
    #OVerlay images
    overlap_MV = get_overlap_mask(truth,MV_img)
    

    CT = sitk.ReadImage(os.path.join(outpath,'CT.TIF'))
    CT_smooth = sitk.Cast(sitk.RescaleIntensity(CT), overlap_MV.GetPixelID())
    
    overlay_MV = sitk.LabelOverlay(CT_smooth,overlap_MV,opacity=0.25, backgroundValue = 0, colormap = yellow+red+green)
    
    
    sitk.WriteImage(overlay_MV, os.path.join(outpath,'MV_overlay.TIF'))
    
    # Write MV_img and Staples_img
    sitk.WriteImage(sitk.Cast(MV_img, sitk.sitkFloat32), os.path.join(outpath,'mean_prediction.TIF'))
    sitk.WriteImage(sitk.Cast(truth, sitk.sitkFloat32), os.path.join(outpath,'truth.TIF'))
    
def write_csv(path,df):
    df.to_csv (path+'/mean_result/mean_results.csv')



def compute_mean_img(inputpaths):
    # Walk through patient dir and load MR image and liver mask:
    # Use enumerations to represent the various evaluation measures
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    for in_path in in_paths:
            # Test Patient
        modeldir= next(os.walk(in_path))[1]
        #print(modeldir)

        patlist = next(os.walk(os.path.join(in_path,modeldir[0])))[1]
        #print(patlist)

        overlap_results = np.zeros((len(patlist),len(OverlapMeasures.__members__.items()))) 
        for p,pat in enumerate(patlist):
            truth = sitk.Cast(sitk.ReadImage(os.path.join(in_path,modeldir[0],pat,'truth.TIF')), sitk.sitkUInt8)

            fold0 = sitk.Cast(sitk.ReadImage(os.path.join(in_path,modeldir[0],pat,'prediction.TIF')), sitk.sitkUInt8)

            fold1 = sitk.Cast(sitk.ReadImage(os.path.join(in_path,modeldir[1],pat,'prediction.TIF')), sitk.sitkUInt8)

            fold2 = sitk.Cast(sitk.ReadImage(os.path.join(in_path,modeldir[2],pat,'prediction.TIF')), sitk.sitkUInt8)

            fold3 = sitk.Cast(sitk.ReadImage(os.path.join(in_path,modeldir[3],pat,'prediction.TIF')), sitk.sitkUInt8)

            fold4 = sitk.Cast(sitk.ReadImage(os.path.join(in_path,modeldir[4],pat,'prediction.TIF')), sitk.sitkUInt8)

            folds = [fold0,fold1,fold2,fold3,fold4]

            majority_img = sitk.LabelVoting(folds,1)

            truth_area  = get_area(truth)
            overlap_results[p,OverlapMeasures.truth_area.value] = truth_area
            seg = majority_img

            overlap_measures_filter.Execute(truth, seg)

            overlap_results[p,OverlapMeasures.MV_dice.value] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[p,OverlapMeasures.MV_volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()

            seg_area = get_area(seg)
            rel_diff = (truth_area - seg_area)/(truth_area)

            overlap_results[p,OverlapMeasures.MV_area.value] = seg_area
            overlap_results[p,OverlapMeasures.MV_reldiff.value] = rel_diff

            write_output(in_path,pat,majority_img,truth)

        # Graft our results matrix into pandas data frames 
        results_df = pd.DataFrame(data=overlap_results, index = list(range(len(patlist))), 
                                          columns=[name for name, _ in OverlapMeasures.__members__.items()]) 


        write_csv(in_path,results_df)

if __name__=='__main__':
        #os.chdir('/home/jovyan/segmentation_results')
    cwd = os.getcwd()
    print(cwd)
    in_path1= cwd + "/bin_cross"
    in_path2= cwd +"/bin_cross_aug"
    in_path3= cwd +"/dice"
    in_path4= cwd +"/dice_aug"
    in_paths = list((in_path1,in_path2,in_path3,in_path4))
    compute_mean_img(in_paths)
    