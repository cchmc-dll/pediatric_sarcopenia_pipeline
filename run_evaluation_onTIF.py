import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import PIL
from skimage import io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


config = dict()
config['input_dir'] = 'predictions/combined_new_thresholding_fold_2'
config['truth_img'] = 'truth'
config['prediction_img'] = 'prediction'
config['output_file'] = 'fold_2.csv'



def get_mask(data):
    val = data > 0
    return val

def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

#def muscle_area(truth, prediction,pixel_size):
#    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def read_excel(filename,sheet):
    df = pd.read_excel(io=file_name, sheet_name=sheet)
    return(df)

def main():
    header = ("DSC",)
    masking_functions = (get_mask,)
    rows = list()
    subject_ids = list()
    for case_folder in glob.glob(config['input_dir'] + "/*"):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, config['truth_img']+".TIF")
        truth = io.imread(truth_file)
        prediction_file = os.path.join(case_folder, config['prediction_img']+".TIF")
        prediction = io.imread(prediction_file)
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])

    
    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    print('Index of df:', df.index,'\t', df['DSC'], '\n \n')
 
 
    df.to_csv(config['input_dir'] + '/' + config['output_file'])

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig(os.path.join(config['input_dir'], "validation_scores_boxplot_test2.png"))
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph_test1_train2.png')


if __name__ == "__main__":
    main()
