import tyro
import os
import numpy as np
from mononphm import env_paths
from mononphm.utils.print_utils import print_flashy


# run MICA on the first image sequence
def main(seq_name : str,
         intrinsics_provided : bool = True
         ):

    print_flashy(f'[ENTERING - METRICAL TRACKER] @ {seq_name}')

    config_contents = f'''# Tracker config

actor: '{env_paths.DATA_TRACKING}/{seq_name}/'
save_folder: '{env_paths.DATA_TRACKING}/{seq_name}/metrical_tracker/'
optimize_shape: true
optimize_jaw: true
#begin_frames: 1
begin_frames: 0
#keyframes: [0, 1]
keyframes: [0]
intrinsics_provided: {'true' if intrinsics_provided else 'false'}'''

    config_pth = f"{env_paths.CODE_BASE}/src/mononphm/preprocessing/metrical-tracker/configs/actors/{seq_name}.yml"
    with open(config_pth, "w") as text_file:
        text_file.write(config_contents)


    if os.path.exists(f'{env_paths.DATA_TRACKING}/{seq_name}/metrical_tracker/{seq_name}/video.avi'):
        print('Metrical tracking already present. SKIPPING!')
        return

    os.system(f'cd {env_paths.CODE_BASE}/src/mononphm/preprocessing/metrical-tracker/; python tracker.py --cfg {config_pth}')

    print_flashy(f'[EXITING - METRICAL TRACKER] @ {seq_name}')


if __name__ == '__main__':
    tyro.cli(main)
