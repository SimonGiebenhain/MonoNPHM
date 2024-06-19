import numpy as np


def invert_similarity_transformation(T):
    inverse_rotation_scaling = np.linalg.inv(T[:3, :3])
    inverse_translation = inverse_rotation_scaling @ -T[:3, 3]

    inverse_transform = np.eye(4)
    inverse_transform[:3, :3] = inverse_rotation_scaling
    inverse_transform[:3, 3] = inverse_translation

    return inverse_transform