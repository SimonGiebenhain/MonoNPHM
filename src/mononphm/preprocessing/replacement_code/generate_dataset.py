import os
from abc import ABC
from glob import glob
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from face_detector import FaceDetector
from image import crop_image_bbox, squarefiy, get_bbox


class GeneratorDataset(Dataset, ABC):
    def __init__(self, source, config):
        self.device = 'cuda:0'
        self.config = config
        self.source = Path(source)

        self.initialize()
        self.face_detector_mediapipe = FaceDetector('google')
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)

    def initialize(self):
        path = Path(self.source, 'source')
        if not path.exists() or len(os.listdir(str(path))) == 0:
            video_file = self.source / 'video.mp4'
            if not os.path.exists(video_file):
                logger.error(f'[ImagesDataset] Neither images nor a video was provided! Execution has stopped! {self.source}')
                exit(1)
            path.mkdir(parents=True, exist_ok=True)
            os.system(f'ffmpeg -i {video_file} -vf fps={self.config.fps} -q:v 1 {self.source}/source/%05d.png')

        self.images = sorted(glob(f'{self.source}/source/*.jpg') + glob(f'{self.source}/source/*.png'))

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        dense_lmks = self.face_detector_mediapipe.dense(image)
        return lmks, dense_lmks

    def run(self):
        logger.info('Generating dataset...')
        bbox = None
        bbox_path = self.config.actor + "/bbox.pt"

        if os.path.exists(bbox_path):
            bbox = torch.load(bbox_path)
        pipnet_landmarks = np.load(f'{self.config.actor}/pipnet/test.npy')

        for frame, imagepath in tqdm(enumerate(self.images)):
            lmk_path = imagepath.replace('source', 'kpt').replace('png', 'npy').replace('jpg', 'npy')
            lmk_path_dense = imagepath.replace('source', 'kpt_dense').replace('png', 'npy').replace('jpg', 'npy')

            # at least for first frame, make sure that image size is specified correctly
            if frame == 0 and os.path.exists(imagepath.replace('source', 'images')):
                image = cv2.imread(imagepath.replace('source', 'images'))
                h, w, c = image.shape
                assert h == w
                self.config.image_size = (w, h)

            if not os.path.exists(lmk_path) or not os.path.exists(lmk_path_dense):
                image = cv2.imread(imagepath)
                h, w, c = image.shape

                # load landmarks and scale s.t. unit is in pixels
                #detected_lms = pipnet_landmarks[frame, :]
                #detected_lms[:, 0] *= w
                #detected_lms[:, 1] *= h

                CROP = self.config.intrinsics_provided
                if CROP:
                    # simple crop image
                    # center crop around average 2D landmark position
                    #avg_landmark = np.mean(detected_lms, axis=0).astype(int)
                    ## fixed square crop of size: 800 pixels
                    #box_size_x = 2*min(400, min(avg_landmark[0], w-avg_landmark[0]-1))
                    #box_size_y = 2*min(400, min(avg_landmark[1], h-avg_landmark[1]-1))
                    #box_size = min(box_size_x, box_size_y)

                    #box_width = box_size
                    #box_height = box_size
                    #left_start = avg_landmark[0] - box_width // 2
                    #top_start = avg_landmark[1] - box_height // 2
                    #box_size = 1080
                    if image.shape[1] == 1920:
                        image = image[:, 420:-420, :]
                    image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]),
                                       interpolation=cv2.INTER_CUBIC)
                    #image = image[top_start:top_start+box_size, left_start:left_start+box_size, :]
                    #self.config.image_size = (box_size, box_size)
                else:
                    self.config.image_size = (w, h)

                image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]),
                                   interpolation=cv2.INTER_CUBIC)

                lmk, dense_lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if lmk is None:
                    logger.info(f'Empty face_alignment lmks for path: ' + imagepath)
                    lmk = np.zeros([68, 2])

                if dense_lmk is None:
                    logger.info(f'Empty mediapipe lmks for path: ' + imagepath)
                    dense_lmk = np.zeros([478, 2])

                Path(lmk_path).parent.mkdir(parents=True, exist_ok=True)
                Path(lmk_path_dense).parent.mkdir(parents=True, exist_ok=True)
                Path(imagepath.replace('source', 'images')).parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(imagepath.replace('source', 'images'), image)
                np.save(lmk_path_dense, dense_lmk)
                np.save(lmk_path, lmk)



