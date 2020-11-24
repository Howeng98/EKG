# %%
# from google.colab import drive
# drive.mount('/content/drive/')
!nvidia-smi
# %%
from torchvision.datasets import ImageFolder

# %%
data_dir = 'dataset'
input, target = [], []

# %%
def preprocessing(data_dir):
	dataset = ImageFolder(data_dir)
	classes = dataset.classes
	print(len(dataset.imgs)) # 600 images

	for i in range(len(dataset)):
		input.append(dataset.imgs[i][0])
		target.append(dataset.imgs[i][1])

	

preprocessing(data_dir)

# %%
