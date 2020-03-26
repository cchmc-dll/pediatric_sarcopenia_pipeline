from argparse import ArgumentParser
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool
from os.path import join
import sys
from functools import partial
from collections import namedtuple
import time
import subprocess

TrainValPair = namedtuple('TrainValPair', ["train_pkl", "val_pkl", "fold"])

def main(argv):
    args = parse_args(argv)

    with open('pass.txt', 'r') as f:
        password = f.readlines()[0]

    pickle_files = [
        TrainValPair(
            join(args.split_dir, "fold_{0}_train.pkl".format(i)),
            join(args.split_dir, "fold_{0}_val.pkl".format(i)),
            fold=i
        )
        for i
        in range(5)
    ]

    available_gpus = Queue()
    available_gpus.put(0)
    # available_gpus.put(1)
    available_gpus.put(2)
    available_gpus.put(3)

    with ThreadPool(processes=args.num_gpus) as p:
        p.map(
            func=partial(run_training, available_gpus=available_gpus, run_name=args.run_name, password=password),
            iterable=pickle_files
        )


def parse_args(argv):
    parser = ArgumentParser()

    parser.add_argument('run_name', help='Run name corresponding to .args file in config/run')
    parser.add_argument('split_dir', help="Path on remote to directory with kfold split pickle files")
    parser.add_argument('num_gpus', type=int, help="The number of gpus available. Will run 1 fold per gpu")
    
    return parser.parse_args(argv)


def run_training(train_val_pair, available_gpus, run_name, password):
    try:
        gpuid = available_gpus.get()
        extra_args = f'"--training_split={train_val_pair.train_pkl} --validation_split={train_val_pair.val_pkl} --training_model_name=combined_actually_bin_cross_219_fold_{train_val_pair.fold}"' 
        command = rf"fab -H dockeruser@10.1.32.31 run-training {run_name} --gpuids={gpuid} --extra-python-args={extra_args} > C:\Users\casju6\Desktop\out_f{train_val_pair.fold}.txt"

        # make sure docker containers have time to build
        time.sleep(gpuid * 20)

        print("Fold:", train_val_pair.fold)
        print(f'Running: "{command}""')

        # shell=True so don't run this code on a server..
        subprocess.run(command, shell=True)
    finally:
        available_gpus.put(gpuid)

if __name__ == "__main__":
    main(sys.argv[1:])