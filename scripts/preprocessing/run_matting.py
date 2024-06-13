import os

MODNet_path = '/home/giebenhain/debugNPHM/NPHM-TUM/src/nphm_tum/preprocessing/MODNet/'
video_folder = '/home/giebenhain/TalkingHead-1KH/val/cropped_clips/'
matting_folder = '/mnt/hdd/matting_videos/'
video_name = 'A2800grpOzU_0002_S812_E1407_L227_T7_R1139_B919.mp4'

os.makedirs(matting_folder, exist_ok=True)
fps = 24
cmd = f'cd {MODNet_path}; python -m demo.video_matting.custom.run --video {video_folder}/{video_name} --result-type matte --fps {fps}'

print(cmd)
os.system(cmd)

cmd = f'mv {video_folder}/{video_name[:-4]}_matte.mp4 {matting_folder}/{video_name}'
os.system(cmd)
