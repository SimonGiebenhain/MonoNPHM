import numpy as np

def cut_throat(template_mesh, points):
    idv1 = 3276
    idv2 = 3207
    idv3 = 3310
    v1 = template_mesh.vertices[idv1, :]
    v2 = template_mesh.vertices[idv2, :]
    v3 = template_mesh.vertices[idv3, :]
    origin = v1
    line1 = v2 - v1
    line2 = v3 - v1
    normal = np.cross(line1, line2)

    direc = points - origin
    angle = np.sum(normal * direc, axis=-1)
    above = angle > 0
    return above