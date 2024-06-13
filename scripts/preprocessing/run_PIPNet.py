import sys
import tyro
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt

import os
import numpy as np
from PIL import Image
import distinctipy
import math

from mononphm import env_paths
sys.path.append(f'{env_paths.CODE_BASE}/src/mononphm/preprocessing/PIPNet/FaceBoxesV2/')
from mononphm.preprocessing.pipnet_utils import face_detection, landmark_detection
from mononphm.utils.print_utils import print_flashy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def export_landmarks(norm_lm_obj, path):
    lms = np.zeros([478, 3])
    #for idx, landmark in enumerate(landmark_list.landmark):
    for i, lm in enumerate(norm_lm_obj.landmark):
        lms[i, 0] = lm.x
        lms[i, 1] = lm.y
        lms[i, 2] = lm.z
    np.save(path, lms)



def run_PIPnet(images, bboxes):
    lms = landmark_detection(images, bboxes, 'experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py', device)
    heatmaps = None
    return lms, heatmaps


def main_kinect(seq_name : str):

    print_flashy(f'[ENTERING - LANDMARK PREDICTION] @ {seq_name}')


    seq_folder = env_paths.DATA_TRACKING
    #seq_tags = [
    #    'example_images'
    #]
    seq_tags = [seq_name]
    for seq_tag in seq_tags:
        folder = f'{seq_folder}/{seq_tag}/source/'
        out_bbox = f'{seq_folder}/{seq_tag}/bboxes/'
        out_lms = f'{seq_folder}/{seq_tag}/pipnet/'
        out_annotations = f'{seq_folder}/{seq_tag}/annotations/'

        if os.path.exists(f'{out_lms}/test.npy'):
            print('Landmark detection already finished! SKIPPING!')
            return
        failed = []

        files = [f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')]
        files.sort()

        #print(files)

        images = []
        for file in files:
            image = np.array(Image.open(f'{folder}/{file}'))
            images.append(image)

        image_height, image_width, _ = images[0].shape

        # --------
        # Face/Bbox detection
        # --------

        bboxes, confidences = face_detection(images, use_gpu=True, device='cuda', snapshot_dir=f'{env_paths.CODE_BASE}/src/mononphm/preprocessing/PIPNet/')
        bboxes = np.array(bboxes)

        # bboxes, confidences = demo.face_detection(images, use_gpu=True, device=device)
        bboxes = np.array(bboxes)
        confidences = np.array(confidences)[:, np.newaxis]

        # normalize bbox before saving
        normalized_bboxes = bboxes.copy().astype(float)
        normalized_bboxes[:, 0] /= image_width
        normalized_bboxes[:, 2] /= image_width
        normalized_bboxes[:, 1] /= image_height
        normalized_bboxes[:, 3] /= image_height
        os.makedirs(out_bbox, exist_ok=True)
        np.save(f'{out_bbox}/test.npy', np.concatenate([normalized_bboxes, confidences], axis=-1))




        # --------
        # Landmark detection
        # --------

        print('[LOG] Detecting Landmarks')

        lms, heatmaps = run_PIPnet(images, bboxes.astype(int))
        # normalize landmarks
        normalized_lms = lms.copy().astype(float)
        normalized_lms[:, :, 0] /= image_width
        normalized_lms[:, :, 1] /= image_height
        os.makedirs(out_lms, exist_ok=True)
        np.save(f'{out_lms}/test.npy', normalized_lms)



        # --------
        # Image annotation
        # --------
        print('[LOG] Annotating Images')

        unique_col = distinctipy.get_colors(math.ceil(lms.shape[1] / 10), rng=0)
        unique_col = unique_col * 10

        for i, image in tqdm(enumerate(images), desc="Annotating images"):
            circs = []
            for j in range(lms[i].shape[0]):
                circs.append(plt.Circle((lms[i, j, 0].item(), lms[i, j, 1].item()), 2, color=unique_col[j]))

            fig, ax = plt.subplots()
            plt.imshow(image)
            plt.axis('off')
            #for j in range(lms[i].shape[0]):
            #    plt.text(lms[i, j, 0].item(), lms[i, j, 1].item(), str(j), color="red", fontsize=6)
            for circ in circs:
                ax.add_patch(circ)
            ax.add_patch(plt.Rectangle(
                (bboxes[i, 0], bboxes[i, 1]), width=bboxes[i, 2], height=bboxes[i, 3],
                edgecolor=(1., 0., 0.), facecolor='none', alpha=1
            ))

            os.makedirs(out_annotations, exist_ok=True)
            plt.savefig(f'{out_annotations}/{i:03d}.jpg', bbox_inches='tight', dpi=500)
            plt.close('all')
    print_flashy(f'[EXITING - LANDMARK PREDICTION] @ {seq_name}')




if __name__ == '__main__':
    tyro.cli(main_kinect)

