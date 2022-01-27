ipython -i L3_finder.py -- \
    --dicom_dir /opt/data/ \
    --model_path /opt/smi/models/l3/child_9_slice_w_transfer_fold_3.h5 \
    --output_directory /opt/smi/debug/ \
    --overwrite \
    --save_plots \
    --new_tim_dicom_dir_structure \
    --series_to_skip_pickle_file /opt/smi/broken_sagittal_and_axial_series.pkl \

