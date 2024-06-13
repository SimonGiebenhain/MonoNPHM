import os
import sys

from math import ceil
import tyro
import torch
import distinctipy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import facer


#sys.path.append('..')
from mononphm import env_paths
from mononphm.utils.print_utils import print_flashy


colors = distinctipy.get_colors(22, rng=0)


def viz_results(img, seq_classes, n_classes, suppress_plot = False):

    seg_img = np.zeros([img.shape[-2], img.shape[-1], 3])
    #distinctipy.color_swatch(colors)
    bad_indices = [
        0,  # background,
        1,  # neck
        # 2, skin
        3,  # cloth
        4,  # ear_r (images-space r)
        5,  # ear_l
        # 6 brow_r
        # 7 brow_l
        # 8,  # eye_r
        # 9,  # eye_l
        # 10 noise
        # 11 mouth
        # 12 lower_lip
        # 13 upper_lip
        14,  # hair,
        # 15, glasses
        16,  # ??
        17,  # earring_r
        18,  # ?
    ]
    bad_indices = []

    for i in range(n_classes):
        if i not in bad_indices:
            seg_img[seq_classes[0, :, :] == i] = np.array(colors[i])*255

    if not suppress_plot:
        plt.imshow(seg_img.astype(np.uint(8)))
        plt.show()
    return Image.fromarray(seg_img.astype(np.uint8))

def get_color_seg(img, seq_classes, n_classes):

    seg_img = np.zeros([img.shape[-2], img.shape[-1], 3])
    colors = distinctipy.get_colors(n_classes+1, rng=0)
    #distinctipy.color_swatch(colors)
    bad_indices = [
        0,  # background,
        1,  # neck
        # 2, skin
        3,  # cloth
        4,  # ear_r (images-space r)
        5,  # ear_l
        # 6 brow_r
        # 7 brow_l
        # 8,  # eye_r
        # 9,  # eye_l
        # 10 noise
        # 11 mouth
        # 12 lower_lip
        # 13 upper_lip
        14,  # hair,
        # 15, glasses
        16,  # ??
        17,  # earring_r
        18,  # ?
    ]

    for i in range(n_classes):
        if i not in bad_indices:
            seg_img[seq_classes[0, :, :] == i] = np.array(colors[i])*255


    return Image.fromarray(seg_img.astype(np.uint8))


def crop_gt_img(img, seq_classes, n_classes):

    seg_img = np.zeros([img.shape[-2], img.shape[-1], 3])
    colors = distinctipy.get_colors(n_classes+1, rng=0)
    #distinctipy.color_swatch(colors)
    bad_indices = [
        0,  # background,
        1,  # neck
        # 2, skin
        3,  # cloth
        4, #ear_r (images-space r)
        5, #ear_l
        # 6 brow_r
        # 7 brow_l
        #8,  # eye_r
        #9,  # eye_l
        # 10 noise
        # 11 mouth
        # 12 lower_lip
        # 13 upper_lip
        14,  # hair,
        # 15, glasses
        16,  # ??
        17,  # earring_r
        18,  # ?
    ]

    for i in range(n_classes):
        if i in bad_indices:
            img[seq_classes[0, :, :] == i] = 0


    #plt.imshow(img.astype(np.uint(8)))
    #plt.show()
    return img.astype(np.uint8)


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main(seq_name : str):
    print_flashy(f'[ENTERING - FACE SEGMENTATION] @ {seq_name}')
    seq_folder = env_paths.DATA_TRACKING
    seq_tags = [
        seq_name
    ]
    seq_tag = seq_tags[0]

    folder = f'{seq_folder}/{seq_tag}/source/'#'/home/giebenhain/GTA/data_kinect/color/'
    out_seg = f'{seq_folder}/{seq_tag}/seg/'
    out_seg_annot = f'{seq_folder}/{seq_tag}/seg_annotations/'

    os.makedirs(out_seg, exist_ok=True)
    os.makedirs(out_seg_annot, exist_ok=True)



    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    face_parser = facer.face_parser('farl/celebm/448', device=device)  # optional "farl/lapa/448"


    frames = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]

    if len(os.listdir(out_seg)) == len(frames):
        print('Facer segmentation masks already present. SKIPPIING!')
        return

    frames.sort()
    image_stack = []
    original_image_sizes = []
    for file in frames:
        resize_needed = False
        frame = int(file.split('.')[0])
        img = Image.open(f'{folder}/{file}')
        og_size = img.size
        original_image_sizes.append(og_size)
        if img.size[0] > 1000:
            img = img.resize( (int(img.size[0]/3), int(img.size[1]/3)))
        image = facer.hwc2bchw(torch.from_numpy(np.array(img)[..., :3])).to(device=device)  # image: 1 x 3 x h x w
        image_stack.append(image)

    batch_size = 8
    for batch_idx in range(ceil(len(image_stack)/batch_size)):
        image_batch = torch.cat(image_stack[batch_idx*batch_size:(batch_idx+1)*batch_size], dim=0)

        with torch.inference_mode():
            faces = face_detector(image_batch)
            torch.cuda.empty_cache()
            faces = face_parser(image_batch, faces)#, bbox_scale_factor=1.5)
            torch.cuda.empty_cache()

        seg_logits = faces['seg']['logits']
        back_ground = torch.all(seg_logits == 0, dim=1, keepdim=True).detach().squeeze(1).cpu().numpy()
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        seg_classes = seg_probs.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        seg_classes[back_ground] = seg_probs.shape[1] + 1

        for _iidx in range(seg_probs.shape[0]):

            iidx = faces['image_ids'][_iidx].item()
            I_color = viz_results(image_batch[iidx:iidx+1],
                                  seq_classes=seg_classes[_iidx:_iidx+1],
                                  n_classes=seg_probs.shape[1] + 1,
                                  suppress_plot=True)
            I = Image.fromarray(seg_classes[_iidx])
            I_color = I_color.resize(original_image_sizes[batch_size*batch_idx + iidx], Image.NEAREST)
            I = I.resize(original_image_sizes[batch_size*batch_idx + iidx], Image.NEAREST)
            I.save(f'{out_seg}/{batch_size*batch_idx + iidx}.png')
            I_color.save(f'{out_seg_annot}/color_{batch_size*batch_idx + iidx}.png')
        torch.cuda.empty_cache()


    print_flashy(f'[EXITING - FACE SEGMENTATION] @ {seq_name}')



if __name__ == '__main__':
    tyro.cli(main)