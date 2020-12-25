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
import pywt
import pywt.data


input, target = [], []

def showImage(windowname,image):
	cv2.imshow(windowname, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def preprocessing(data_dir):
	dataset = ImageFolder(data_dir)
	classes = dataset.classes
	print(len(dataset.imgs)) # 600 images

	# example
	# print(dataset.imgs[1][0])
	# print(dataset.imgs[1][1])

	lead_matrix = ['none'] * len(dataset)
	for idx in range(len(dataset)):
		img = cv2.imread(dataset.imgs[idx][0],1)
		# showImage('img',img)
		lead_matrix = cropping(lead_matrix, idx, img)

	showImage('img',lead_matrix[1][2])
	
def cropping(lead_matrix, idx, image):
	
	# top_1
	top1 = image[390:512, 115:423]
	# middle_1
	mid1 = image[512:634, 115:423]
	# bottom_1
	bot1 = image[634:756, 115:423]
	# top_2
	top2 = image[390:512, 427:727]
	# middle_2
	mid2 = image[512:634, 427:727]
	# bottom_2
	bot2 = image[634:756, 427:727]
	# top_3
	top3 = image[390:512, 731:1031]
	# middle_3
	mid3 = image[512:634, 731:1031]
	# bottom_3
	bot3 = image[634:756, 731:1031]
	# top_4
	top4 = image[390:512, 1035:1335]
	# middle_4
	mid4 = image[512:634, 1035:1335]
	# bottom_4
	bot4 = image[634:756, 1035:1335]
	
	
	lead_matrix[idx] = [top1,mid1,bot1,top2,mid2,bot2,top3,mid3,bot3,top4,mid4,bot4]
	return lead_matrix


if __name__ == "__main__":
	data_dir = 'dataset'
	preprocessing(data_dir)

# training_data = []
# training_label = []

# directory_path = 'EKG_seg/'
# folder = os.listdir(directory_path)
# for entries in folder:
# 	f = os.listdir(directory_path+entries+'/')
# 	for entry in f:
# 		if entry == 'img.png':
# 			data = cv2.imread(os.path.join(directory_path,entries,entry),1)
# 			training_data.append(data[:,:,0])
# 		elif entry == 'label.png':
# 			label = cv2.imread(os.path.join(directory_path,entries,entry),1)
# 			training_label.append(label[:,:,0])
# 		else:
# 			assert('there is no img.png and label.png\n')

# print(len(training_data))
# print(len(training_label))

# training_data = training_data.astype('float32')
# training_label = training_label.astype('float32')

# batch_size = 32
