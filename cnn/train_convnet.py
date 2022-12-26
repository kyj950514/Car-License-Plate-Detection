import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from convnet import ConvNet
from trainer import Trainer
from lxml import etree
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224  # input 이미지 사이즈 (224, 224)


# 이미지 resize
def image_resize(img_dir):
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    files.sort()  # We sort the images in alphabetical order to match them to the xml files containing the annotations of the bounding boxes
    X = []
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        X.append(np.array(img))

    return X


# xml 파일에서 바운딩박스 label 추출
def resizeannotation(f):
    tree = etree.parse(f)
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text) / (width / IMAGE_SIZE)
        ymin = int(dim.xpath("ymin")[0].text) / (height / IMAGE_SIZE)
        xmax = int(dim.xpath("xmax")[0].text) / (width / IMAGE_SIZE)
        ymax = int(dim.xpath("ymax")[0].text) / (height / IMAGE_SIZE)
    return [int(xmax), int(ymax), int(xmin), int(ymin)]


path = '../archive/annotations/'
text_files = [path + f for f in sorted(os.listdir(path))]
y = []

# label 추출
for i in text_files:
    y.append(resizeannotation(i))

X = image_resize('../archive/images/')

X = np.array(X).transpose(0, 3, 1, 2)
y = np.array(y)

# normalization
X = X / 255
y = y / 255

# train, test 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

max_epochs = 20

# CNN 불러오기
network = ConvNet()

# trainer 불러오기
trainer = Trainer(network, X_train, y_train, X_test, y_test,
                  epochs=max_epochs, mini_batch_size=10,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=100)

# train 시작
trainer.train()

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

