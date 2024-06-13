import random
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import trimesh
from typing import Literal, Optional
import traceback
from PIL import Image
import cv2

from mononphm.data.utils import  uniform_ball
import mononphm.env_paths as env_paths
from mononphm.data.manager import DataManager



class NPHMdataset(Dataset):
    def __init__(self,
                 mode : Literal['train', 'val'],
                 n_supervision_points_face : int,
                 n_supervision_points_non_face : int,
                 n_supervision_points_off_surface : int,
                 n_supervision_points_corresp : int,
                 batch_size : int,
                 lm_inds : np.ndarray,
                 num_anchors: int,
                 num_symm: int,
                 mirror : bool = False,
                 extra_hair : bool = True,
                 neutral_type : str = 'closed',
                 neutral_only : bool = False,
                 no_validation : bool = False,
                 ):

        self.MIRROR = mirror
        self.EXTRA_HAIR = extra_hair
        self.NEUTRAL_TYPE = neutral_type
        self.manager = DataManager(num_anchors=num_anchors)
        self.num_anchors = num_anchors
        self.num_symm = num_symm

        self.neutral_expr_index = env_paths.neutrals_closed

        self.mode = mode
        self.neutral_only = neutral_only

        self.ROOT = env_paths.SUPERVISION_DEFORMATION_CLOSED
        self.lm_inds = lm_inds

        self.batch_size = batch_size
        # different category of supervision signal are differently important
        # therefor we use different numbers of points for training for the different categories
        self.n_supervision_points_face = n_supervision_points_face
        self.n_supervision_points_non_face = n_supervision_points_non_face
        self.n_supervision_points_off_surface = n_supervision_points_off_surface
        self.n_supervision_points_corresp = n_supervision_points_corresp




        ######################################################################################################
        ############################# Gather successfully prepared subjects ##################################
        ######################################################################################################
        if mode == 'train':
            self.subjects = self.manager.get_train_subjects(neutral_type=self.NEUTRAL_TYPE, exclude_missing_neutral=False)
            # put all subjects into train set
            if no_validation:
                self.subjects += self.manager.get_eval_subjects(neutral_type=self.NEUTRAL_TYPE, exclude_missing_neutral=False)
                self.subjects += self.manager.get_test_subjects(neutral_type=self.NEUTRAL_TYPE, exclude_missing_neutral=False)
        else:
            self.subjects = self.manager.get_eval_subjects(neutral_type=self.NEUTRAL_TYPE, exclude_missing_neutral=False)

        # only keep subjects that have a neutral scan
        if neutral_only:
            self.subjects = [s for s in self.subjects if self.manager.get_neutral_expression(s, neutral_type=self.NEUTRAL_TYPE) is not None and self.manager.get_neutral_expression(s, neutral_type=self.NEUTRAL_TYPE) >= 0]

        # only keep subjects for which trainin data has been successfully pre-computed
        self.subjects = [s for s in self.subjects if os.path.exists(self.manager.get_train_dir(s))]


        # do extensive checks the remove all subjects for data has not yet been successfully pre-computed
        actual_subjects = []
        for i, s in enumerate(self.subjects):

            # get all expressions for s for which training data has been successfully pre-computed
            if self.neutral_only:
                expressions = [self.manager.get_neutral_expression(s, self.NEUTRAL_TYPE)]
            else:
                expressions = self.manager.get_expressions(s)
            expressions = [e for e in expressions if os.path.exists(self.manager.get_train_path_face_region(s, e, rnd_file=99))]
            if len(expressions) == 0:
                continue

            # only keep subjects with neutral expression for which training data has been successfully pre-computed
            neutral_expr = self.manager.get_neutral_expression(s, self.NEUTRAL_TYPE)
            if neutral_expr is not None:
                if neutral_expr in expressions:
                    actual_subjects.append(s)
        self.subjects = actual_subjects

        # for some people the hair-style changed during scanning
        # Since we don't want these changes to be present in the deformation field, we treat them as extra identities
        # For these subjects we add 10000 to their id
        # Some people have more than one change of hairstyles, every change gets converted into an extra identitiy,
        #   each time we add 10000 to their id
        if self.EXTRA_HAIR:
            extra_subjects = []
            for s in self.subjects:
                if s in self.manager.extra_hairstyles.keys():
                    for extra in range(len(self.manager.extra_hairstyles[s])):
                        extra_subjects.append(s+10000*(extra+1))
            self.subjects += extra_subjects

        print('################################################################################################')
        print(f'Currently creating an {self.mode} NPHMdataset object.')
        print('The following IDs have been found and will be included:')
        print(self.subjects)
        print('################################################################################################')
        #-----------------------------------------------------------------------------------------------------


        ######################################################################################################
        ############################## Define Ordering of data for training ##################################
        ######################################################################################################
        self.subject_IDs = [] # stores subject id for each data point
        self.expression_steps = [] # stores index of expressions for each subject
        self.subject_training_ordering = [] # defines order of subjects used in training, relevant for the auto-decoder's expression codebook
        self.expression_training_ordering = [] # defines order of subjects used in training, relevant for auto-decoder's expression codebook
        self.neutral_codebook_index = {} # defines which indices in expression codebooks are neutral
        all_files = []
        count = 0
        for i, s in enumerate(self.subjects):
            # gather valid expressions
            if self.neutral_only:
                expressions = [self.manager.get_neutral_expression(s % 10000, self.NEUTRAL_TYPE)]
            else:
                expressions = self.manager.get_expressions(s, extra_hair=self.EXTRA_HAIR)
            expressions = [e for e in expressions if os.path.exists(self.manager.get_train_path_face_region(s % 10000, e, rnd_file=99))]

            # append found expressions to ordering
            self.subject_IDs += len(expressions) * [s, ]
            self.subject_training_ordering += len(expressions) * [i, ]
            self.expression_steps += expressions
            self.expression_training_ordering += [ei + count for ei in range(len(expressions))]

            neutral_expr = self.manager.get_neutral_expression(s % 10000, self.NEUTRAL_TYPE)

            if neutral_expr is None or neutral_expr not in expressions:
                self.neutral_codebook_index[s] = None
            else:
                neutral_index = expressions.index(neutral_expr)
                self.neutral_codebook_index[s] = count + neutral_index

            count += len(expressions)
            all_files.append(expressions)
        #-----------------------------------------------------------------------------------------------------

        # pre-fetch facial anchors for neutral expression
        self.anchors = {}
        for iden in self.subjects:
            if iden < 10000:
                self.anchors[iden] = self.manager.get_facial_anchors(subject=iden, expression=self.neutral_expr_index.get(iden, None))



    def __len__(self):
        # mirroring effectively doubles the dataset set
        if self.MIRROR:
            return len(2*self.expression_steps)
        else:
            return len(self.expression_steps)


    def __getitem__(self, idx):

        og_idx = idx
        # even indices will not get mirrored, odd indices will get mirrored along symmetry plane of the face
        if self.MIRROR:
            if idx % 2 != 0:
                is_mirrored = True
            else:
                is_mirrored = False
            idx = idx // 2 # since we double self.__len__() when using mirror augmentation, we half here to get back to the correct indices
        else:
            is_mirrored = False
        expr = self.expression_steps[idx]
        iden = self.subject_IDs[idx]
        subj_ind = self.subject_training_ordering[idx]

        #TODO
        # supervise neutral expression more frequently
        if np.random.rand() < 0.1: # for extra hairstyles, there are no neutral expressions
            neutral_expr = self.manager.get_neutral_expression(iden, self.NEUTRAL_TYPE)
            if neutral_expr is not None and self.neutral_codebook_index[iden] is not None:
                expr = neutral_expr
                if self.neutral_expr_index[iden] >= 0:
                    idx = self.neutral_codebook_index[iden]
                    og_idx = idx

                    if self.MIRROR:
                        idx *= 2
                        is_mirrored = False
                        if np.random.rand() < 0.5:
                            idx += 1
                            is_mirrored = True

        is_neutral = expr == self.neutral_expr_index.get(iden % 10000, None)

        #TODO
        # handling of extra hairstyle
        if iden >= 10000:
            is_neutral = False # never is a true neutral scan
            extra_hairstyle_key_exprs = [0] # 0
            for extra in self.manager.extra_hairstyles[iden % 10000]:
                extra_hairstyle_key_exprs.append(extra + 1)
            # determine which hairstyle chunk the labelled neutral expression is for
            # for this chunk don't assume that the first expression per chunk is the neutral
            # instead overwrite with labelled neutral expression
            curr_neutral_expr = self.neutral_expr_index.get(iden%10000, None)
            if curr_neutral_expr is not None:
                modified = False
                for extra in range(len(extra_hairstyle_key_exprs)-1):
                    if curr_neutral_expr < extra_hairstyle_key_exprs[extra+1]:
                        extra_hairstyle_key_exprs[extra] = curr_neutral_expr
                        modified = True
                        break
                if not modified:
                    extra_hairstyle_key_exprs[-1] = curr_neutral_expr

        # load data
        try:
            if os.path.exists(self.manager.get_train_path_deformation(iden % 10000, expr, neutral_type=self.NEUTRAL_TYPE)):
                    point_corresp = np.load(self.manager.get_train_path_deformation(iden % 10000, expr, neutral_type=self.NEUTRAL_TYPE))
                    valid = np.logical_not( np.any(np.isnan(point_corresp), axis=-1))
                    point_corresp = point_corresp[valid, :].astype(np.float32)
            else:
                point_corresp = None

            # includes points in the facial region
            points_face = np.load(self.manager.get_train_path_face_region(subject=iden % 10000,
                                                                          expression=expr))
            # points on the back of the head
            points_back = np.load(self.manager.get_train_path_non_face_region(subject=iden % 10000,
                                                                                  expression=expr))
            # points samples near the surface of the head
            points_off_surface = np.load(self.manager.get_train_path_off_surface(subject=iden % 10000,
                                                                                 expression=expr))
            # point on the front which are summed to be mainly static, i.e.
            # hair on the front-side, ears, front of the neck
            points_static_front = np.load(self.manager.get_train_path_static_front_region(subject=iden % 10000,
                                                                                        expression=expr))


            assert points_face.shape[0] > 0 and  points_back.shape[0] > 0 and points_off_surface.shape[0] > 0, f's{iden}e{expr}'
        except Exception as e:
            print('FAILED', iden, expr)
            traceback.print_exc()
            rnd_replacement = np.random.randint(0, self.__len__())
            return self.__getitem__(rnd_replacement) # avoid crashing of training by returning other random sample in case of failure

        # generate random index used for subsampling the loaded data to the required number points for training
        sup_idx_non_face = np.random.randint(0, points_back.shape[0], self.n_supervision_points_non_face)
        sup_idx_off_surface = np.random.randint(0, points_off_surface.shape[0], self.n_supervision_points_off_surface)
        sup_idx_face = np.random.randint(0, points_face.shape[0], int(self.n_supervision_points_face * 0.75))
        sup_idx_static_front = np.random.randint(0, points_static_front.shape[0], self.n_supervision_points_face - int(self.n_supervision_points_face*0.75))



        sup_points_face = points_face[sup_idx_face, :3]
        sup_normals_face = points_face[sup_idx_face, 3:6]
        sup_color_face = points_face[sup_idx_face, 6:9]
        sup_points_static_front = points_static_front[sup_idx_static_front, :3]
        sup_normals_static_front = points_static_front[sup_idx_static_front, 3:6]
        sup_color_static_front = points_static_front[sup_idx_static_front, 6:9]
        sup_points_face = np.concatenate([sup_points_face, sup_points_static_front], axis=0)
        sup_normals_face = np.concatenate([sup_normals_face, sup_normals_static_front], axis=0)
        sup_color_face = np.concatenate([sup_color_face, sup_color_static_front], axis=0)


        sup_points_non_face = points_back[sup_idx_non_face, :3]
        sup_normals_non_face = points_back[sup_idx_non_face, 3:6]
        sup_color_non_face = points_back[sup_idx_non_face, 6:9]


        sup_points_off_surface = points_off_surface[sup_idx_off_surface, :3]
        sup_color_off_surface = points_off_surface[sup_idx_off_surface, 6:9]



        # subsample points for supervision
        if point_corresp is not None:
            sup_idx = np.random.randint(0, point_corresp.shape[0], self.n_supervision_points_corresp)
            sup_points_neutral = point_corresp[sup_idx, :3]
            sup_points_posed = point_corresp[sup_idx, 3:]
        else:
            sup_points_neutral = None
            sup_points_posed = None


        has_anchors = True
        if self.anchors[iden % 10000] is not None:
            gt_anchors = self.anchors[iden % 10000].copy()
        else:
            gt_anchors = None
        if gt_anchors is None:
            has_anchors = False
            # replace anchors with zeros in the correct shape
            gt_anchors = np.zeros([self.lm_inds.shape[0], 3])


        # randomly sample points in a bounding sphere
        sup_grad_far = uniform_ball(self.n_supervision_points_face // 8, rad=1.0)


        # finally apply mirroring
        if is_mirrored:
            sup_points_face[:, 0] *= -1
            sup_normals_face[:, 0] *= -1
            sup_points_non_face[:, 0] *= -1
            sup_normals_non_face[:, 0] *= -1
            sup_points_off_surface[:, 0] *= -1
            gt_anchors[..., 0] *= -1

            # Take special care of the anchors!
            # for this we also need to change the semantics,
            #   e.g. the anchors corresponding left and right corner of the mouths need to switch label
            gt_anchors_tmp = gt_anchors.copy()
            gt_anchors_tmp[:2*self.num_symm:2, :] = gt_anchors[1:2*self.num_symm:2, :]
            gt_anchors_tmp[1:2*self.num_symm:2, :] = gt_anchors[:2*self.num_symm:2, :]
            gt_anchors = gt_anchors_tmp


        # mirrored subjects need a spearate entry in the codebook, so append them at the end
        if self.MIRROR and is_mirrored:
            subj_ind = len(self.subjects) + subj_ind

        return_dict =  {
                'points_surface': sup_points_face.astype(np.float32),
                'normals_surface': sup_normals_face.astype(np.float32),
                'color_surface': (sup_color_face.astype(np.float32) / 255 - 0.5)*2,

                'points_surface_outer': sup_points_non_face.astype(np.float32),
                'normals_surface_outer': sup_normals_non_face.astype(np.float32),
                'color_surface_outer': (sup_color_non_face.astype(np.float32) / 255 - 0.5)*2,

                'points_off_surface': sup_points_off_surface.astype(np.float32),
                'color_off_surface': (sup_color_off_surface.astype(np.float32) / 255 - 0.5)*2,

                'sup_grad_far': sup_grad_far.astype(np.float32),

                'idx': np.array([og_idx]), # index in to the expression codebook
                'iden': np.array([self.subjects.index(iden)]),
                'expr': np.array([expr]),
                'subj_ind': np.array([subj_ind]), # index into the identity codebook
                'app_ind': np.array([subj_ind]), #appearance and identity are coupled
                'gt_anchors': gt_anchors,
                'is_neutral': is_neutral,
                'supervise_hair': is_neutral or (iden >= 10000 and expr in extra_hairstyle_key_exprs),
                'has_anchors': has_anchors,
        }
        if sup_points_posed is not None:
            has_corresp = True
            if is_mirrored:
                sup_points_neutral[:, 0] *= -1
                sup_points_posed[:, 0] *= -1
            return_dict.update({'corresp_neutral': sup_points_neutral,
                                'corresp_posed': sup_points_posed,
                                'has_corresp': has_corresp})
        else:
            has_corresp = False
            # need to include zeros to enable batch of data
            return_dict.update({'corresp_neutral': np.zeros([self.n_supervision_points_corresp, 3]),
                                'corresp_posed': np.zeros([self.n_supervision_points_corresp, 3]),
                                'has_corresp': has_corresp})


        return return_dict

    def get_loader(self):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True)