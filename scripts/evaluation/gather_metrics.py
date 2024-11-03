import os
import tyro
import numpy as np
import json
from mononphm import env_paths


def main(model_name : str):
    base_folder = f'{env_paths.TRACKING_OUTPUT}/../mononphm_evaluation/{model_name}'

    seqs = [f for f in os.listdir(base_folder) if f.endswith('evaluation')]


    metrics_dict = {}
    for seq in seqs:

        json_file = f'{base_folder}/{seq}/avg_metrics.json'

        with open(json_file, 'r') as fp:
            f = json.load(fp)

        print(f)
        for metric in f.keys():
            if metric not in metrics_dict:
                metrics_dict[metric] = []
            metrics_dict[metric].append(float(f[metric]))


    for metric in metrics_dict.keys():
        print(metric)
        print(np.mean(np.array(metrics_dict[metric])))

if __name__ == '__main__':
    tyro.cli(main)