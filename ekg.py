
# from google.colab import drive
# drive.mount('/content/drive/')
# !nvidia-smi

import os, random, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm


data_dir = 'dataset'
input, target = [], []

def showImage(windowname,image):
	cv2.imshow(windowname, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def preprocessing(data_dir):
	dataset = ImageFolder(data_dir)
	classes = dataset.classes
	print(len(dataset.imgs)) # 600 images

	for i in tqdm(range(len(dataset))):
		input.append(dataset.imgs[i][0])
		target.append(dataset.imgs[i][1])

	x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=42)
	print(len(x_train))	
	print(len(y_train))
	print(len(x_test))
	print(len(y_test))

	# example
	print(dataset.imgs[1][0])
	print(dataset.imgs[1][1])
	img = cv2.imread(dataset.imgs[1][0],cv2.IMREAD_GRAYSCALE)		
	cropping(img, dataset.imgs[1][0], dataset.imgs[1][1])

def cropping(image, image_path, image_label):
	img_save_path = image_path[:-4] + '/'
	
	# top_1
	crop = image[390:512, 115:423]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	
	# cv2.imwrite(os.path.join(img_save_path , image_path[:] + '_top1' + '.png'), crop)
	
	# middle_1
	crop = image[512:634, 115:423]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_middle1' + '.png', crop)

	# bottom_1
	crop = image[634:756, 115:423]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_bottom1' + '.png', crop)

	# top_2
	crop = image[390:512, 427:727]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_top2' + '.png', crop)

	# middle_2
	crop = image[512:634, 427:727]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + 'middle2' + '.png', crop)
	
	# bottom_2
	crop = image[634:756, 427:727]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_bottom2' + '.png', crop)
	
	# top_3
	crop = image[390:512, 731:1031]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_top3' + '.png', crop)
	
	# middle_3
	crop = image[512:634, 731:1031]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_middle3' + '.png', crop)
	
	# bottom_3
	crop = image[634:756, 731:1031]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_bottom3' + '.png', crop)
	
	# top_4
	crop = image[390:512, 1035:1335]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_top4' + '.png', crop)
	
	# middle_4
	crop = image[512:634, 1035:1335]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_middle4' + '.png', crop)

	# bottom_4
	crop = image[634:756, 1035:1335]
	crop = cv2.resize(crop, (128, 128))
	# showImage('crop', crop)
	# cv2.imwrite(image_path[:] + '_bottom4' + '.png', crop)

preprocessing(data_dir)

