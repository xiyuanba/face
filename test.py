import torch
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
# Load pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load and preprocess face images
image1 = cv2.imread('/home/yingtie/PycharmProjects/face/Vladimir_Putin/Vladimir_Putin_0001.jpg')
image2 = cv2.imread('/home/yingtie/PycharmProjects/face/Yingfan_Wang_0001.jpg')


face_locations1 = face_recognition.face_locations(image1)
for i, face_location1 in enumerate(face_locations1):
    top, right, bottom, left = face_location1
    face_image = image1[top:bottom, left:right]
    # 在同目录下创建一个新的 JPEG 图像文件，命名为 tmp<序号>.jpg
    filename1 = f'tmp1{i + 1}.jpg'
    cv2.imwrite(filename1, face_image)
    t1 = cv2.imread(filename1)
    tmp_img1 = cv2.resize(t1, (160, 160))

face_locations2 = face_recognition.face_locations(image2)
for i, face_location2 in enumerate(face_locations2):
    top, right, bottom, left = face_location2
    face_image = image2[top:bottom, left:right]
    # 在同目录下创建一个新的 JPEG 图像文件，命名为 tmp<序号>.jpg
    filename2 = f'tmp2{i + 1}.jpg'
    cv2.imwrite(filename2, face_image)
    t2 = cv2.imread(filename2)
    tmp_img2 = cv2.resize(t2, (160, 160))

# image1 = cv2.resize(image1, (160, 160))
# image2 = cv2.resize(image2, (160, 160))

tmp_img1 = cv2.cvtColor(tmp_img1, cv2.COLOR_BGR2RGB)
tmp_img2 = cv2.cvtColor(tmp_img2, cv2.COLOR_BGR2RGB)

tmp_img1 = (tmp_img1 / 255.).astype(np.float32)
tmp_img2 = (tmp_img2 / 255.).astype(np.float32)

# Extract features from face images
with torch.no_grad():
    features1 = model(torch.from_numpy(np.array([tmp_img1.transpose(2,0,1)])))
    features2 = model(torch.from_numpy(np.array([tmp_img2.transpose(2,0,1)])))

features1 = features1.detach().numpy()
features2 = features2.detach().numpy()

# Compute cosine similarity between features
similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
similarity_score = similarity_matrix[0][0]
print(similarity_score)