import json
from pathlib import Path
from environs import Env

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/mononphm/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("MONONPHM_"):
    CODE_BASE = env("CODE_BASE")
    TRAINING_SUPERVISION = env("TRAINING_SUPERVISION")
    DATA = env("DATA")
    EXPERIMENT_DIR = env("EXPERIMENT_DIR")
    DATA_TRACKING = env("DATA_TRACKING")
    TRACKING_OUTPUT = env("TRACKING_OUTPUT")

ASSETS = f'{CODE_BASE}/assets/'
SUPERVISION_IDENTITY = f'{TRAINING_SUPERVISION}/surface_pointclouds/'
SUPERVISION_DEFORMATION_CLOSED = f'{TRAINING_SUPERVISION}/deformations/'
MICA_INPUT_PATH = f'{DATA_TRACKING}/MICA_input/'
DATA_HIGH_RES = DATA


ANCHOR_INDICES_PATH = ASSETS + 'lm_inds_{}.npy'
ANCHOR_MEAN_PATH = ASSETS + 'anchors_{}.npy'

NUM_SPLITS = 200
NUM_SPLITS_EXPR = 100

with open(CODE_BASE + '/dataset/neutrals_open.json') as f:
    _neutrals = json.load(f)
with open(CODE_BASE + '/dataset/neutrals_closed.json') as f:
    _neutrals_closed = json.load(f)
neutrals = {int(k): v for k, v in _neutrals.items()}
neutrals_closed = {int(k): v for k, v in _neutrals_closed.items()}

subjects_eval = [199, 286, 290, 291, 292, 293, 294, 295, 297, 298]

subjects_test = [99, 283, 143, 38, 241, 236, 276, 202, 98, 254, 204, 163, 267, 194, 20, 23, 209, 105, 186, 343, 341,
                 363, 350, 508, 509, 510, 511]



invalid_expressions_test = {
    143: [0, 1, 5],
    163: [6],  # --> FLAME fitting failed to move in proper coordinate system
    38: [1, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19],  # hair changes --> maybe train or eval set
    236: [8, ],  # clothes not ideal ???
    202: [24, ],
    98: [0],
    254: [1, ],
    204: [16, ],
    267: [0, 7, 13, 22],
    194: [0, 1, 2, 3, 9, 11, 14, 18, 22],
    20: [17, 6, 11, 13, ],
    209: [7, 8, 9, 10, 15, 20, ],
    105: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    186: [7, 8, 9, 11, 21, ],
    343: [9, 11, ],
    363: [1, 11, 12, 14, ],
    350: [4, ],
    468: [20, 21]
}

for s in subjects_test:
    if s not in invalid_expressions_test.keys():
        invalid_expressions_test[s] = []

bad_scans = {
    261: [19],
    88: [19],
    79: [16, 17, 18, 19, 20],
    100: [0],
    125: [1, 4, 5],
    106: [20],
    362: [20],
    363: [1],
    345: [12],
    360: [6, 14],
    85: [2],
    292: [9],
    298: [23, 24, 25, 26],
    194: [1],
    312: [9],
    305: [9],
    315: [14],
    409: [11],
    447: [6],  # faile regi
    461: [2],  # failed regi
    483: [5],  # failed regi
    418: [3, 9, 11],
    502: [9, 10],
    508: [5, 8, 20],
    511: [9, 21],
    532: [9],
    535: [21],
    401: [4],
    415: [3],
    518: [0],  # generally poor quality

    503: [3], # random error in preparetion script --> need to recompute later

    549: [5],
    541: [20],
    545: [7,8],
    548: [22],

}

