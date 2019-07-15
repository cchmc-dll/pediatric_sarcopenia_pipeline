"""
Preprocess image for DICOM to Anonymized DICOM, NIFTI and TIFF
"""
import csv
import glob
from argparse import ArgumentParser
from collections import namedtuple
from os import listdir
from os.path import isfile, join, isdir
from optparse import OptionParser
from pathlib import Path

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
# inDir     = r'C:\Users\somd7w\Desktop\Airway_Project\original_imagefiles\Additional_data_EC\Good_Coley_Dicom_Additional'
tifDir    = r'C:\Users\somd7w\Desktop\Airway_Project\original_imagefiles\Additional_data_EC\Coley_Good'
csvfile =   r'C:\Users\somd7w\Desktop\Airway_Project\original_imagefiles\Additional_data_EC\Coley_Good.csv'

#outDir    = r'C:\Users\somd7w\Desktop\Airway_Project\bad_anon'
#niftiDir  = r'C:\Users\somd7w\Desktop\Airway_Project\bad_nii '
#dcm2nifti = r'C:\Users\somd7w\Downloads\win\mricron\dcm2nii.exe'
dcm2niix_exe = Path(os.getcwd(), 'ext', 'dcm2niix.exe')

dataset_path = Path("\\\\vnas1\\root1\\Radiology\\SHARED\\Elan\\Projects\\Skeletal Muscle Project\\Dataset2")
sagittal_csv_path = Path("\\\\vnas1\\root1\\Radiology\\SHARED\\Elan\\Projects\\Skeletal Muscle Project\\sagittal_series_remaining.csv")

CTImage = namedtuple('DcmImage', ['subject_id', 'series', 'src_dirs'])

nifti_out_dir = Path.cwd().joinpath('tests', 'data', 'nifti_out')


def get_image_info_from(csv_path=sagittal_csv_path):
    with open(sagittal_csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        yield from csv_reader


def dicom_src_dir_for(subject_id, series, path=dataset_path):
    return CTImage(
        subject_id,
        series,
        src_dirs=path.glob(f"*{subject_id}/**/SE-{series}-*/")
    )


def nifti_from_dcm_dir(dcm_dir, filename, output_dir):
    cmd = [str(dcm2niix_exe), '-s', 'y', '-f', filename, '-o', str(nifti_out_dir), str(src_dir)]
    print(cmd)
    subprocess.check_call(cmd)
    return Path(nifti_out_dir, f'{filename}.nii')



# ct_image_generator = (dicom_src_dir_for(row['subject_id'], row['series']) for row in get_image_info_from())
# for ct_image in ct_image_generator:
#     for src_dir in ct_image.src_dirs:
#         cmd = [str(dcm2niix_exe), '-s', 'y', '-f', ct_image.subject_id, '-o', str(nifti_out_dir), str(src_dir)]
#         print(cmd)
#         subprocess.check_call(cmd)


def main():
    args = parse_args()
    convert_and_anonymize_dicom(config=args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('input_dir', help='Directory of dicom files to convert')
    parser.add_argument('to_format', help='Format to convert to', default='nifti')
    parser.add_argument('output_dir', help='Path to directory for output files')

    return parser.parse_args()


def convert_and_anonymize_dicom(config):
    header = ("Filename","Station_Name","Study_Description","Series_Description","Sex","Age","Ethnic_Group","Protocol","Resolution","Rows","Columns","StoD","StoP")

    if config.input_dir is not None:
        assert isdir(config.input_dir), ' Input Directory does not exist'
        files = [f for f in listdir(config.input_dir) if \
                 isfile(join(config.input_dir, f))]
        subject_ids = list()
        rows = list()
        count = 1
        for index, f in enumerate(files):
            ds = dicom.read_file(join(config.input_dir, f))
            analysis_vars = list()
            analysis_vars.append(f)
            anonymize_dicom(ds,analysis_vars)
            subject_ids.append(analysis_vars[1]+'_'+analysis_vars[2]) # Accession Number + Image Time
            del analysis_vars[1:3]
            rows.append(analysis_vars)

            write_anonymized_dicom(output_dir=config.output_dir, dicom_path=f, dataset=ds)
            save_as_nifti(output_dir=config.output_dir, subject_ids=subject_ids, pt_index=index)

            # Writing Anonymized Dicom
            #if outDir is not None:
            #    assert isdir(outDir), 'Directory does not exist'
            #    # Save DICOM
            #    outfile = join(outDir, f)
            #    dicom.write_file(outfile, ds)
            # Writing NIFTI
            # if niftiDir is not None:
            #    assert isdir(niftiDir), 'Directory does not exist'
            #    # Save NII
            #    niftifile = subject_ids[index]
            #    #OPTS = dcm2nifti + ' -d n -e n -g n -p n -a y -f y -n y -v n -o ' + niftiDir + '  ' + outfile
            #    OPTS = dcm2niix + ' -s y -f '  + niftifile  + ' -o ' + niftiDir + '  ' + outfile
            #    subprocess.call(OPTS,shell=True)
            # Writing TIF
            # if tifDir is not None:
            #     assert isdir(tifDir), 'Directory does not exist'
            #     # Save TIF
            #     shape = ds.pixel_array.shape
            #     # Convert to float to avoid overflow or underflow losses.
            #     image_2d = ds.pixel_array.astype(float)
            #     # Rescaling grey scale between 0-255
            #     image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
            #     # Convert to uint
            #     image_2d_scaled = np.uint8(image_2d_scaled)
            #     tiffile = join(tifDir, subject_ids[index]+'.tif')
            #     cv2.imwrite(tiffile,image_2d_scaled)
            #     nfiles =  len([name for name in os.listdir(tifDir) if os.path.isfile(join(tifDir,name))])
            #     print('Index ' ,index ,  ' nfiles ' , nfiles)
            #     if(index+count != nfiles):
            #          count = count-1
            #          print(subject_ids[index],'Check for Repetition')

        # df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
        # print(df.head())
        # df.to_csv(csvfile)


def write_anonymized_dicom(output_dir, dicom_path, dataset):
    outfile = join(output_dir, dicom_path)
    dicom.write_file(outfile, dataset)


def save_as_nifti(output_dir, subject_ids, pt_index):
    assert isdir(output_dir), 'Directory does not exist'
    # Save NII
    niftifile = os.path.abspath(subject_ids[pt_index])
    outfile = os.path.abspath(os.path.join(output_dir, str(pt_index)))

    # cmd = str(dcm2niix_exe) + ' -s y -f '  + niftifile  + ' -o ' + output_dir + '  ' + outfile
    cmd = [str(dcm2niix_exe), '-s', 'y', '-f', niftifile, '-o', output_dir, str(outfile)]
    print(cmd)
    subprocess.call(cmd,shell=True)







