"""
Preprocess image for DICOM to Anonymized DICOM, NIFTI and TIFF
"""
from os import listdir
from os.path import isfile, join, isdir
from optparse import OptionParser
import pydicom as dicom
from pydicom.tag import Tag
import pandas as pd
import numpy as np
import nibabel as nib
from dicom_anonymizer.anonymize_dicom_Airway import anonymize_dicom
import subprocess              
import cv2
import os

#from dicom-anon import dicom_anon

#Current working directory
Output_Container = []
inDir     = r'C:\Users\somd7w\Desktop\Airway_Project\original_imagefiles\Additional_data_EC\Good_Coley_Dicom_Additional'
tifDir    = r'C:\Users\somd7w\Desktop\Airway_Project\original_imagefiles\Additional_data_EC\Coley_Good'
csvfile =   r'C:\Users\somd7w\Desktop\Airway_Project\original_imagefiles\Additional_data_EC\Coley_Good.csv'

#outDir    = r'C:\Users\somd7w\Desktop\Airway_Project\bad_anon'
#niftiDir  = r'C:\Users\somd7w\Desktop\Airway_Project\bad_nii '
#dcm2nifti = r'C:\Users\somd7w\Downloads\win\mricron\dcm2nii.exe'
#dcm2niix  = r'C:\Users\somd7w\Downloads\mricrogl\dcm2niix.exe'


print(isdir(inDir))

header = ("Filename","Station_Name","Study_Description","Series_Description","Sex","Age","Ethnic_Group","Protocol","Resolution","Rows","Columns","StoD","StoP")

if inDir is not None:
        assert isdir(inDir), ' Input Directory does not exist'
        files = [f for f in listdir(inDir) if \
                 isfile(join(inDir, f))]
        subject_ids = list()
        rows = list()
        count = 1
        for index,f in enumerate(files):
            ds = dicom.read_file(join(inDir, f))
            analysis_vars = list()
            analysis_vars.append(f)
            anonymize_dicom(ds,analysis_vars)
            subject_ids.append(analysis_vars[1]+'_'+analysis_vars[2]) # Accession Number + Image Time
            del analysis_vars[1:3]
            rows.append(analysis_vars)
            # Writing Anonymized Dicom 
            #if outDir is not None:
            #    assert isdir(outDir), 'Directory does not exist'
            #    # Save DICOM
            #    outfile = join(outDir, f)
            #    dicom.write_file(outfile, ds)
            # Writing NIFTI
            #if niftiDir is not None:
            #    assert isdir(niftiDir), 'Directory does not exist'
            #    # Save NII
            #    niftifile = subject_ids[index]
            #    #OPTS = dcm2nifti + ' -d n -e n -g n -p n -a y -f y -n y -v n -o ' + niftiDir + '  ' + outfile
            #    OPTS = dcm2niix + ' -s y -f '  + niftifile  + ' -o ' + niftiDir + '  ' + outfile
            #    subprocess.call(OPTS,shell=True)
            # Writing TIF
            if tifDir is not None:
                assert isdir(tifDir), 'Directory does not exist'
                # Save TIF
                shape = ds.pixel_array.shape
                # Convert to float to avoid overflow or underflow losses.
                image_2d = ds.pixel_array.astype(float)
                # Rescaling grey scale between 0-255
                image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
                # Convert to uint
                image_2d_scaled = np.uint8(image_2d_scaled)
                tiffile = join(tifDir, subject_ids[index]+'.tif')
                cv2.imwrite(tiffile,image_2d_scaled)
                nfiles =  len([name for name in os.listdir(tifDir) if os.path.isfile(join(tifDir,name))])
                print('Index ' ,index ,  ' nfiles ' , nfiles)
                if(index+count != nfiles):
                     count = count-1
                     print(subject_ids[index],'Check for Repetition')
        
        df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
        print(df.head())
        df.to_csv(csvfile)






        

