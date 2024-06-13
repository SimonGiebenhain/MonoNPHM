import tyro
import os
import numpy as np
from mononphm import env_paths
from mononphm.utils.print_utils import print_flashy


# run MICA on the first image sequence
def main(seq_name : str):
    print_flashy(f'[ENTERING - MICA] @ {seq_name}')


    data_path = env_paths.DATA_TRACKING
    image_path = f'{data_path}/{seq_name}/source/00000.png'

    mica_input_folder = f'{env_paths.MICA_INPUT_PATH}/{seq_name}'

    if os.path.exists(f'{data_path}/{seq_name}/identity.npy'):
        print('MICA prediction already present. SKIPPING!')
        return

    os.makedirs(mica_input_folder, exist_ok=True)
    # get first frame and place into MICA input folder
    os.system(f'cp {image_path} {mica_input_folder}/')


    os.system(f'cd {env_paths.CODE_BASE}/src/mononphm/preprocessing/MICA/; python demo.py -i {mica_input_folder} -o {data_path}/{seq_name}/')
    os.system(f'mv {data_path}/{seq_name}/00000/* {data_path}/{seq_name}')
    os.system(f'rm -r {data_path}/{seq_name}/00000/')

    print_flashy(f'[EXITING - FACE SEGMENTATION] @ {seq_name}')


if __name__ == '__main__':
    tyro.cli(main)
