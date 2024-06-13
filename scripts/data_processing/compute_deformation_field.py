import trimesh
import numpy as np
import os
from multiprocessing import Pool

from mononphm import env_paths
from mononphm.data.manager import DataManager
from mononphm.utils.mesh_operations import cut_trimesh_vertex_mask


#ADDITIONAL_FOLDER = '/mnt/rohan/cluster/doriath/sgiebenhain/nphm_dataset3/'
###if not os.path.exists('/mnt/rohan/'):
#  ADDITIONAL_FOLDER = '/cluster/doriath/sgiebenhain/nphm_dataset3/'

def sample(m_neutral, m, std, n_samps):
    p_neureal, idx_neutral = m_neutral.sample(n_samps, return_index=True)
    normals_neutral = m_neutral.face_normals[idx_neutral, :]
    faces = m_neutral.faces[idx_neutral]
    faces_lin = np.reshape(faces, [-1])
    triangles_neutral = m_neutral.vertices[faces_lin, :]
    triangles_neutral = np.reshape(triangles_neutral, [-1, 3, 3])

    bary = trimesh.triangles.points_to_barycentric(triangles_neutral, p_neureal, method='cross')
    offsets = np.random.randn(p_neureal.shape[0]) * std
    offsets = np.expand_dims(offsets, axis=-1)
    p_neureal += offsets * normals_neutral

    faces = m.faces[idx_neutral]
    normals = m.face_normals[idx_neutral, :]
    faces_lin = np.reshape(faces, [-1])
    triangles = m.vertices[faces_lin, :]
    triangles = np.reshape(triangles, [-1, 3, 3])

    p = trimesh.triangles.barycentric_to_points(triangles, bary)
    p += offsets * normals
    return p_neureal, p, normals_neutral, normals


def main(s):
    manager = DataManager(high_res=True)#, additional_folder=ADDITIONAL_FOLDER)
    expressions = manager.get_expressions(subject=s)
    failed_expresssion = []
    for expression in expressions:
        try:
            #if not VIZ and os.path.exists(manager.get_train_path_deformation(s, expression, neutral_type=NEUTRAL_TYPE,
            #                                                                 rnd_file=env_paths.NUM_SPLITS_EXPR-1)):
            if NEUTRAL_TYPE == 'closed':
                if os.path.exists(f'{env_paths.SUPERVISION_DEFORMATION_CLOSED}/{s:03d}/{expression:03d}/corresp_{env_paths.NUM_SPLITS_EXPR-1}.npy') and not VIZ:
                    print('skip')
                    continue
            else:
                if os.path.exists(f'{env_paths.SUPERVISION_DEFORMATION_OPEN}/{s:03d}/{expression:03d}/corresp_{env_paths.NUM_SPLITS_EXPR-1}.npy') and not VIZ:
                    print('skip')
                    continue


            n_expr = manager.get_neutral_expression(s, neutral_type=NEUTRAL_TYPE)
            if n_expr is None or n_expr < 0:
                continue
            m_neutral = manager.get_registration_mesh(subject=s,
                                                      expression=n_expr,
                                                      )
            m = manager.get_registration_mesh(subject=s, expression=expression)

            #pl = pv.Plotter()
            #pl.add_mesh(face_region_template)
            #pl.add_mesh(m, color='red')
            #pl.show()

            invalid = face_region_template.visual.vertex_colors[:, 0] != 255
            m = cut_trimesh_vertex_mask(m, np.logical_not(invalid))
            m_neutral = cut_trimesh_vertex_mask(m_neutral, np.logical_not(invalid))


            if VIZ:
                pl = pv.Plotter()
                pl.add_mesh(m)
                pl.add_mesh(m_neutral, color='red')
                pl.show()

            p_neutral, p, normals_neutral, normals = sample(m_neutral, m, 0.01, n_samps=N_SAMPLES)#0.01)
            p_neutral2, p2, normals_neutral2, normals2 = sample(m, m_neutral, 0.01, n_samps=N_SAMPLES)#0.01)
            p_neutral = np.concatenate([p_neutral, p2], axis=0)
            p = np.concatenate([p, p_neutral2], axis=0)
            normals_neutral = np.concatenate([normals_neutral, normals2], axis=0)
            normals = np.concatenate([normals, normals_neutral2], axis=0)

            p_neutral_tight, p_tight, normals_neutral_tight, normals_tight = sample(m_neutral, m, 0.002, n_samps=N_SAMPLES)#0.002)
            p_neutral_tight2, p_tight2, normals_neutral_tight2, normals_tight2 = sample(m, m_neutral, 0.002, n_samps=N_SAMPLES)#0.002)
            p_neutral_tight = np.concatenate([p_neutral_tight, p_tight2], axis=0)
            p_tight = np.concatenate([p_tight, p_neutral_tight2], axis=0)
            normals_neutral_tight = np.concatenate([normals_neutral_tight, normals_tight2], axis=0)
            normals_tight = np.concatenate([normals_tight, normals_neutral_tight2], axis=0)


            all_p_neutral = np.concatenate([p_neutral, p_neutral_tight], axis=0)
            all_normals_neutral = np.concatenate([normals_neutral, normals_neutral_tight], axis=0)
            all_p = np.concatenate([p, p_tight], axis=0)
            all_normals = np.concatenate([normals, normals_tight], axis=0)
            perm = np.random.permutation(all_p.shape[0])
            all_p_neutral = all_p_neutral[perm, :]
            all_normals_neutral = all_normals_neutral[perm, :]
            all_p = all_p[perm, :]
            all_normals = all_normals[perm, :]
            if np.any(np.isnan(all_p)) or np.any(np.isnan(all_normals)):
                print('DONE')
                break

            if VIZ:
                pl = pv.Plotter(shape=(1, 2))
                pl.subplot(0, 0)
                pl.add_points(all_p_neutral, scalars=all_normals_neutral[:, 0])
                pl.subplot(0, 1)
                pl.add_points(all_p, scalars=all_normals_neutral[:, 0])
                pl.link_views()
                pl.show()
            data = np.concatenate([all_p_neutral, all_p], axis=-1)
            data_normals = np.concatenate([all_normals_neutral, all_normals], axis=-1)
            split_files = np.array_split(data, env_paths.NUM_SPLITS_EXPR, axis=0)
            #split_files_normals = np.array_split(data_normals, 100, axis=0)
            if not VIZ:
                export_dir_se = manager.get_train_dir_deformation(s, expression, neutral_type=NEUTRAL_TYPE)
                os.makedirs(export_dir_se, exist_ok=True)
                for i in range(len(split_files)):
                    split_file_path = manager.get_train_path_deformation(s, expression, neutral_type=NEUTRAL_TYPE,
                                                                         rnd_file=i)
                    np.save(split_file_path, split_files[i])
                #for i in range(len(split_files_normals)):
                #    split_file_path = export_dir + '/{}_{:03d}/corresp_normals_{}.npy'.format(s, expression, i)
                #    np.save(split_file_path, split_files[i])


            if VIZ and False:
                deform = all_p - all_p_neutral


                #deform -= deform.min()
                #deform /= deform.max()
                #deform *= 255
                #deform = deform.astype(np.uint8)
                #color = deform

                deform_norm = np.linalg.norm(deform, axis=-1)
                deform_norm -= np.min(deform_norm)
                deform_norm /= np.max(deform_norm)
                deform_norm *= 255
                color = np.zeros([deform_norm.shape[0], 4], dtype=np.uint8)
                color[:, 0] = deform_norm.astype(np.uint8)
                color[:, 3] = 255

                pc_neutral = trimesh.points.PointCloud(all_p_neutral)
                pc = trimesh.points.PointCloud(all_p, colors=color)

                pl = pv.Plotter()
                for i in range(1000):
                    pl.add_mesh(pv.Line(all_p_neutral[i, :], all_p[i, :]))
                #pl.add_points(all_p, scalars=all_normals_neutral[:, 0])  # color)
                #pl.link_views()
                pl.camera_position = 'xy'
                pl.camera.position = (0, 0, 3)
                #pl.camera_set = True
                pl.show()

                pl = pv.Plotter(shape=(1, 2))
                pl.subplot(0, 0)

                pl.add_mesh(m_neutral)
                #pl.add_mesh(pv.Plane((0, 0, 0), (1, 0, 0), i_size=2, j_size=10))

                pl.subplot(0, 1)
                pl.add_mesh(m)
                pl.add_points(all_p, scalars=all_normals_neutral[:, 0])#color)
                #pl.add_mesh(pv.Plane((0, 0, 0), (1, 0, 0), i_size=2, j_size=10))
                pl.link_views()
                pl.show()
        except Exception as e:
            failed_expresssion.append(expression)

    print(s, failed_expresssion)

VIZ = False

if VIZ:
    import pyvista as pv

N_SAMPLES = 250000

NEUTRAL_TYPE = 'closed'

if __name__ == '__main__':
    manager = DataManager(high_res=True)#, additional_folder=ADDITIONAL_FOLDER)

    all_subjects = manager.get_all_subjects()

    old_subs = {"17": 0, "18": 0, "19": 0, "20": 0, "22": 0, "23": -1, "24": 0, "25": 0, "26": 0, "27": 0, "28": 0, "29": 0, "31": 0, "32": 0, "33": 0, "34": 0, "35": 0, "36": 0, "37": 0, "38": 0, "39": 0, "40": 0, "41": 0, "42": 0, "43": 0, "44": 0, "45": 0, "46": 0, "48": 0, "49": 0, "50": 0, "51": 0, "52": 0, "53": 0, "54": 0, "55": 0, "56": 0, "57": 0, "58": 0, "59": 0, "60": 0, "61": 0, "62": 0, "63": 0, "64": 0, "65": 0, "67": 0, "68": 0, "69": 0, "70": 0, "71": 0, "72": 0, "73": 0, "74": 0, "75": 0, "76": 0, "77": 0, "78": 0, "79": 0, "80": 0, "81": 0, "82": 0, "83": 0, "84": 3, "85": 0, "86": 0, "87": 0, "88": 0, "89": 0, "90": 0, "91": 0, "92": 0, "93": 8, "94": 0, "95": 0, "96": 0, "97": 0, "98": 1, "99": 0, "100": -1, "102": 0, "103": 0, "104": 0, "105": 0, "106": 0, "108": 0, "109": 0, "110": 0, "111": 0, "112": 0, "113": 0, "114": 0, "115": 0, "116": 0, "117": 0, "118": 0, "120": 0, "121": 0, "122": 0, "123": 0, "124": 0, "125": 0, "126": 0, "127": 0, "128": 0, "129": 0, "130": 0, "131": 0, "132": 0, "133": 0, "134": 21, "135": 0, "136": 0, "137": 0, "138": 0, "140": 1, "141": 0, "142": 0, "143": 0, "144": 0, "145": 0, "146": 0, "147": 2, "148": 0, "149": 0, "150": 0, "151": 0, "162": 0, "163": 0, "164": 0, "165": 0, "167": 0, "168": 0, "174": 0, "179": 0, "180": 0, "181": 0, "182": 0, "183": 24, "184": 0, "185": 0, "186": 0, "187": 0, "188": 0, "189": 0, "190": 0, "191": 0, "193": 0, "194": 23, "195": 1, "196": 0, "198": 0, "199": 0, "200": 0, "201": 0, "202": 0, "204": 0, "206": 0, "207": 0, "209": 0, "210": 0, "211": 0, "212": 0, "213": 0, "214": 0, "215": 0, "216": 0, "217": 0, "218": 0, "220": 0, "221": 0, "223": 0, "224": 0, "226": 0, "227": 0, "228": 0, "229": 0, "231": 0, "232": 0, "233": 0, "234": 0, "235": 0, "236": 0, "237": 0, "238": 0, "239": 0, "240": 0, "241": 0, "242": 0, "243": 0, "244": 1, "245": 0, "246": 0, "247": 0, "248": 0, "249": 0, "250": 0, "251": 0, "252": 0, "254": 0, "255": 0, "256": 0, "257": 0, "258": 0, "259": 0, "260": 0, "261": 2, "262": 0, "263": 0, "264": 0, "265": 0, "267": -1, "268": 0, "269": 0, "270": 0, "271": 0, "272": 0, "274": 0, "275": 0, "276": 0, "277": 0, "278": 0, "279": 0, "280": 0, "281": 0, "282": 0, "283": 1, "284": 0, "285": 0, "286": 0, "287": 0, "289": 0, "290": 0, "291": 0, "292": 0, "293": 0, "294": 0, "295": 0, "297": 0, "298": 0, "334": 0, "335": -1, "336": 0, "337": 0, "338": 0, "339": 0, "340": 0, "341": 0, "342": 0, "343": 0, "344": 0, "345": 0, "346": 0, "347": 0, "348": 0, "349": 0, "350": 0, "351": 0, "352": 0, "353": 0, "354": 0, "355": 0, "356": 0, "357": 0, "358": 0, "359": 0, "360": 0, "361": 0, "362": 0, "363": 0, "364": 1, "365": 16, "107": 0, "203": 0, "205": 0, "208": 0, "222": 0, "225": 0, "230": 0, "253": 0, "288": 0, "299": 0, "300": 0, "301": 0, "302": 0, "303": 0, "304": 0, "305": 0, "306": 0, "307": 0, "308": 0, "309": 0, "310": 0, "311": 0, "312": 0, "313": 0, "314": 0, "315": 0, "316": 0, "317": 0, "318": 0, "319": 0, "320": 0, "322": 0, "323": 0, "324": 0, "325": 0, "326": 0, "327": 0, "328": 0, "329": 0, "330": 0, "331": 0, "332": 0, "333": 0, "367": 0, "368": 0, "369": 0, "370": 0, "371": 0, "372": 0, "373": 0, "374": 0, "375": 0, "376": 0, "377": 0, "378": 0, "379": 0, "380": 0, "381": 0, "382": 0, "383": 0, "385": 0, "386": 0, "387": 0, "388": 0, "389": 0, "390": 0, "391": 0, "393": 0, "395": 0, "396": 0, "397": 0, "398": 0, "399": 0, "400": 0, "401": 0,"384": 0,"392": 0,"394": 0,"402": 1,"403": 0,"404": 0,"405": 0,"406": 0,"407": 0,"408": 1,"409": 0,"410": 0,"411": 0,"412": 0,"414": 0,"415": 0,"416": 0,"417": 0,"419": 0,"420": 0,"421": 0,"422": 0,"423": 0,"424": 0,"426": 0,"427": 0,"428": -1,"429": 0,"430": 0,"431": 0,"432": 0,"433": 0,"434": 0,"435": 0,"436": 0,"437": 0,"438": 0,"439": 0,"440": 0,"441": 0,"442": 0,"443": 0,"444": 0,"448": 0,"449": 0,"453": 0,"454": 0,"457": 0,"459": 0,"460": 0,"462": 1,"464": 0,"465": 0,"467": 0,"468": 0, "446": 0, "447": 0, "450": 0, "451": 0, "452": 0, "455": 0, "456": 0, "458": 0, "461": 0, "469": 0, "471":0, "472": 0, "473": 0, "474": 0, "475": 0, "476": 0, "477": 22, "478": -1, "479": 0, "480": 0, "481": 0, "483": 0, "484": 0, "485": 0, "486": 0}
    old_subs = {
    "17": 10,
    "18": 10,
    "19": 10,
    "20": 10,
    "22": 9,
    "23": 12,
    "24": 10,
    "25": 10,
    "26": 10,
    "27": 10,
    "28": 10,
    "29": 10,
    "31": 10,
    "32": 10,
    "33": 10,
    "34": 10,
    "35": 10,
    "36": 10,
    "37": 9,
    "38": 10,
    "39": 10,
    "40": 1,
    "41": 1,
    "42": 1,
    "43": 1,
    "44": 1,
    "45": 1,
    "46": 1,
    "48": 1,
    "49": 1,
    "50": 1,
    "51": 1,
    "52": 1,
    "53": 1,
    "54": 1,
    "55": 1,
    "56": 1,
    "57": 1,
    "58": 1,
    "59": 1,
    "60": 1,
    "61": 1,
    "62": 1,
    "63": 2,
    "64": 24,
    "65": 1,
    "67": 1,
    "68": 1,
    "69": 1,
    "70": 1,
    "71": 1,
    "72": 1,
    "73": 1,
    "74": 2,
    "75": 1,
    "76": 1,
    "77": 1,
    "78": 1,
    "79": 1,
    "80": 1,
    "81": 23,
    "82": 1,
    "83": 1,
    "84": 0,
    "85": 1,
    "86": 1,
    "87": 1,
    "88": 1,
    "89": 1,
    "90": 1,
    "91": 1,
    "92": 1,
    "93": 0,
    "94": 1,
    "95": 1,
    "96": 1,
    "97": 1,
    "98": 2,
    "99": -1,
    "100": 1,
    "102": 1,
    "103": 1,
    "104": 1,
    "105": -1,
    "106": 1,
    "108": 1,
    "109": 1,
    "110": 1,
    "111": 1,
    "112": 1,
    "113": 1,
    "114": 1,
    "115": 1,
    "116": 2,
    "117": 1,
    "118": 1,
    "120": 1,
    "121": 1,
    "122": 1,
    "123": 1,
    "124": 1,
    "125": -1,
    "126": 1,
    "127": 1,
    "128": 1,
    "129": 1,
    "130": 1,
    "131": 1,
    "132": 1,
    "133": 1,
    "134": 22,
    "135": 1,
    "136": 1,
    "137": 1,
    "138": 1,
    "140": 2,
    "141": 1,
    "142": 1,
    "143": 1,
    "144": 1,
    "145": 1,
    "146": 1,
    "147": 3,
    "148": 1,
    "149": 1,
    "150": 1,
    "151": 1,
    "162": 1,
    "163": 1,
    "164": 1,
    "165": 1,
    "167": 1,
    "168": 1,
    "174": 1,
    "179": 1,
    "180": 1,
    "181": 1,
    "182": 1,
    "183": 1,
    "184": 1,
    "185": 1,
    "186": 1,
    "187": 1,
    "188": 1,
    "189": 1,
    "190": 1,
    "191": 1,
    "193": 1,
    "194": 4,
    "195": 2,
    "196": 1,
    "198": 1,
    "199": 1,
    "200": 1,
    "201": 1,
    "202": 1,
    "204": 1,
    "206": 1,
    "207": 1,
    "209": 1,
    "210": 1,
    "211": 1,
    "212": 1,
    "213": 1,
    "214": 1,
    "215": 1,
    "216": 1,
    "217": 1,
    "218": 1,
    "220": 1,
    "221": 1,
    "223": 1,
    "224": 1,
    "226": 1,
    "227": 1,
    "228": 1,
    "229": 1,
    "231": 1,
    "232": 1,
    "233": 1,
    "234": 1,
    "235": 1,
    "236": -1,
    "237": 1,
    "238": 1,
    "239": 1,
    "240": 1,
    "241": 1,
    "242": 1,
    "243": 1,
    "244": 2,
    "245": 1,
    "246": 1,
    "247": 1,
    "248": 1,
    "249": 1,
    "250": 1,
    "251": 1,
    "252": 1,
    "254": 1,
    "255": 1,
    "256": 1,
    "257": 1,
    "258": 1,
    "259": 1,
    "260": 1,
    "261": 3,
    "262": 1,
    "263": 1,
    "264": 1,
    "265": 1,
    "267": 0,
    "268": 1,
    "269": 1,
    "270": 1,
    "271": 1,
    "272": 1,
    "274": 1,
    "275": 1,
    "276": 1,
    "277": 1,
    "278": 1,
    "279": 1,
    "280": 1,
    "281": 1,
    "282": 1,
    "283": 0,
    "284": 1,
    "285": 1,
    "286": 1,
    "287": 1,
    "289": 1,
    "290": 1,
    "291": 1,
    "292": 1,
    "293": 1,
    "294": 1,
    "295": 1,
    "297": 1,
    "298": 1,
    "334": 12,
    "335": 8,
    "336": 10,
    "337": 10,
    "338": 10,
    "339": 10,
    "340": 10,
    "341": 1,
    "342": 10,
    "343": 10,
    "344": 10,
    "345": 13,
    "346": 10,
    "347": 11,
    "348": 11,
    "349": 10,
    "350": 10,
    "351": 10,
    "352": 10,
    "353": 10,
    "354": 10,
    "355": 1,
    "356": 10,
    "357": 10,
    "358": 10,
    "359": 10,
    "360": 11,
    "361": 10,
    "362": 10,
    "363": 10,
    "364": 11,
    "365": 13
}
    new_subs = []
    for s in all_subjects:
        if str(s) not in old_subs.keys():
            new_subs.append(s)
            print(s)

    print(new_subs)
    #exit()
    print(f"FOUND {len(all_subjects)} subjects!")
    #all_subjects = [536]
    if NEUTRAL_TYPE == 'closed':
        export_dir = env_paths.SUPERVISION_DEFORMATION_CLOSED
    else:
        export_dir = env_paths.SUPERVISION_DEFORMATION_OPEN

    os.makedirs(export_dir, exist_ok=True)

    face_region_template = trimesh.load(env_paths.ASSETS + '/template_face_up.ply', process=False)

    all_subjects = list(range(537, 554))

    if not VIZ:
        p = Pool(10)
        p.map(main, all_subjects)
        p.close()
        p.join()
        #main(all_subjects[0])
    else:
        main(all_subjects[0])