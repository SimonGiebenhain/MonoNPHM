import importlib
import os
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms


from mononphm.preprocessing.PIPNet.FaceBoxesV2.faceboxes_detector import *
from mononphm.preprocessing.PIPNet.lib.networks import *
from mononphm.preprocessing.PIPNet.lib.functions import *
from mononphm.preprocessing.PIPNet.lib.mobilenetv3 import mobilenetv3_large


def face_detection(images, use_gpu, device, snapshot_dir):
	detector = FaceBoxesDetector('FaceBoxes', os.path.join(snapshot_dir, 'FaceBoxesV2/weights/FaceBoxesV2.pth'), use_gpu, device)
	my_thresh = 0.6

	print('#########################################################')
	bboxes = []
	confidences = []
	for image in images:
		image = image.copy()
		image_height, image_width, _ = image.shape
		detections, _ = detector.detect(image, my_thresh, 1)
		dets_filtered = [det for det in detections if det[0] == 'face']
		dets_filtered.sort(key=lambda x: -1*x[1])
		detections = dets_filtered
		print(detections)
		for i in range(min(1, len(detections))):
			bboxes.append(detections[i][2:])
			cur_det = detections[i][1]

			confidences.append(cur_det)
		if len(detections) < 1:
			bboxes.append([-1, -1, -1, -1])
			confidences.append(-1)

	return bboxes, confidences


def landmark_detection(images, bboxes, exp_path, device):
	experiment_name = exp_path.split('/')[-1][:-3]
	data_name = exp_path.split('/')[-2]
	config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

	my_config = importlib.import_module(config_path, package='mononphm.preprocessing.PIPNet')
	Config = getattr(my_config, 'Config')
	cfg = Config()
	cfg.experiment_name = experiment_name
	cfg.data_name = data_name
	snapshot_dir = os.path.join(os.sep, os.path.join(*__file__.split(os.path.sep)[:-2] + ['preprocessing', 'PIPNet', 'snapshots']))
	save_dir = os.path.join(snapshot_dir, cfg.data_name, cfg.experiment_name)

	if cfg.backbone == 'resnet18':
		resnet18 = models.resnet18(pretrained=cfg.pretrained)
		net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
						   net_stride=cfg.net_stride)
	elif cfg.backbone == 'resnet50':
		resnet50 = models.resnet50(pretrained=cfg.pretrained)
		net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
						   net_stride=cfg.net_stride)
	elif cfg.backbone == 'resnet101':
		resnet101 = models.resnet101(pretrained=cfg.pretrained)
		net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
							net_stride=cfg.net_stride)
	elif cfg.backbone == 'mobilenet_v2':
		mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
		net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
	elif cfg.backbone == 'mobilenet_v3':
		mbnet = mobilenetv3_large()
		if cfg.pretrained:
			mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
		net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
	else:
		print('No such backbone!')
		exit(0)

	net = net.to(device)

	weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs - 1))
	state_dict = torch.load(weight_file, map_location=device)
	net.load_state_dict(state_dict)

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	preprocess = transforms.Compose(
		[transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])


	snapshot_dir = os.path.join(os.sep, os.path.join(*__file__.split(os.path.sep)[:-2]))
	det_box_scale = 1.2
	meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join(snapshot_dir, 'preprocessing', 'PIPNet', 'data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

	net.eval()

	print('#########################################################')
	landmarks = []
	annotated_images = []
	for im_idx, image in enumerate(images):
		if bboxes[im_idx][-1] == -1:
			landmarks.append(None)
			continue
		image = image.copy()
		image_height, image_width, _ = image.shape
		det_xmin = int(bboxes[im_idx][0])
		det_ymin = int(bboxes[im_idx][1])
		det_width = int(bboxes[im_idx][2])
		det_height = int(bboxes[im_idx][3])
		det_xmax = det_xmin + det_width - 1
		det_ymax = det_ymin + det_height - 1

		det_xmin -= int(det_width * (det_box_scale - 1) / 2)
		# remove a part of top area for alignment, see paper for details
		det_ymin += int(det_height * (det_box_scale - 1) / 2)
		det_xmax += int(det_width * (det_box_scale - 1) / 2)
		det_ymax += int(det_height * (det_box_scale - 1) / 2)
		det_xmin = max(det_xmin, 0)
		det_ymin = max(det_ymin, 0)
		det_xmax = min(det_xmax, image_width - 1)
		det_ymax = min(det_ymax, image_height - 1)
		det_width = det_xmax - det_xmin + 1
		det_height = det_ymax - det_ymin + 1
		#cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
		det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
		#plt.imshow(det_crop)
		#plt.show()
		det_crop = cv2.resize(det_crop, (cfg.input_size, cfg.input_size))
		inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
		inputs = preprocess(inputs).unsqueeze(0)
		inputs = inputs.to(device)
		lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs,
																								 preprocess, cfg.input_size,
																								 cfg.net_stride, cfg.num_nb)
		tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
		tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
		tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
		tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
		lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
		lms_pred_merge = lms_pred_merge.cpu().numpy()
		pred_export = np.zeros([cfg.num_lms, 2])
		for ii in range(cfg.num_lms):
			x_pred = lms_pred_merge[ii * 2] * det_width
			y_pred = lms_pred_merge[ii * 2 + 1] * det_height
			pred_export[ii, 0] = x_pred + det_xmin
			pred_export[ii, 1] = y_pred + det_ymin
		landmarks.append(pred_export)

	hit = None
	for i in range(len(landmarks)):
		if landmarks[i] is None:
			if hit is not None:
				landmarks[i] = -np.ones_like(hit)
				continue
			j = i + 1
			while landmarks[j] is None:
				j += 1
			hit = landmarks[j]
			landmarks[i] = -np.ones_like(hit)

	return np.stack(landmarks, axis=0)
