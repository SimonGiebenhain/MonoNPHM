import os
import tyro
from mononphm import env_paths

def main(seq_name : str):
    MODNet_path = f'/{env_paths.CODE_BASE}/src/mononphm/preprocessing/MODNet/'
    image_folder = env_paths.DATA_TRACKING
    #seq_name = 'simon_510_s4'
    output_folder = f'{image_folder}/{seq_name}/matting/'
    os.makedirs(output_folder, exist_ok=True)

    if len(os.listdir(f'{image_folder}/{seq_name}/source/')) == len(os.listdir(output_folder)):
        print('Matting already present. SKIPPING!')
        return

    #fps = 24
    cmd = f'cd {MODNet_path}; python -m demo.image_matting.colab.inference --input-path {image_folder}/{seq_name}/source/ --output-path {output_folder} --ckpt-path ./pretrained/modnet_webcam_portrait_matting.ckpt'

    print(cmd)
    os.system(cmd)

    #cmd = f'mv {video_folder}/{video_name[:-4]}_matte.mp4 {matting_folder}/{video_name}'
    #os.system(cmd)

if __name__ == '__main__':
    tyro.cli(main)