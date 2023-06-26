from pprint import pprint

from PIL import Image
import numpy as np
import cv2
import face_recognition
from jina import Document, Executor, Flow, DocumentArray, requests

import torch
from torchvision.models import resnet50
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

class FaceEmbeddingExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self._da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'face'}
        )

    @requests
    def encode(self, docs: DocumentArray, *args, **kwargs):
        # 创建一个 Document 对象并设置其属性值
        for doc in docs:
            image = cv2.imread(doc.uri)
            face_locations = face_recognition.face_locations(image)
            doc.content = image  # 设置 Document 的内容为原始图像
            doc.mime_type = 'image/jpeg'  # 设置 Document 的 MIME 类型为 JPEG 图像
            # 遍历所有人脸区域，并将其添加到 Document 的 chunk 中
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                # 将人脸图像转换为 JPEG 格式，并创建一个新的子 Document 对象
                face_doc = Document(content=face_image, mime_type='image/jpeg')
                # 在同目录下创建一个新的 JPEG 图像文件，命名为 tmp<序号>.jpg
                filename = f'tmp{i+1}.jpg'
                cv2.imwrite(filename, face_image)
                t1 = cv2.imread(filename)
                tmp_img = cv2.resize(t1, (160, 160))
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                tmp_img = (tmp_img / 255.).astype(np.float32)
                with torch.no_grad():
                    features = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
                features = features.detach().numpy()
                doc.embedding = features
                print(doc.embedding)
                with self._da:
                    self._da.append(doc)
                    self._da.sync()

    @requests(on='/search')
    def search(self, docs: DocumentArray, *args, **kwargs):
        global match_array
        match_array = DocumentArray()
        for doc in docs:
            image = cv2.imread(doc.uri)
            face_locations = face_recognition.face_locations(image)
            doc.content = image  # 设置 Document 的内容为原始图像
            doc.mime_type = 'image/jpeg'  # 设置 Document 的 MIME 类型为 JPEG 图像
            # 遍历所有人脸区域，并将其添加到 Document 的 chunk 中
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                # 将人脸图像转换为 JPEG 格式，并创建一个新的子 Document 对象
                face_doc = Document(content=face_image, mime_type='image/jpeg')
                # 在同目录下创建一个新的 JPEG 图像文件，命名为 tmp<序号>.jpg
                filename = f'tmp{i+1}.jpg'
                cv2.imwrite(filename, face_image)
                t1 = cv2.imread(filename)
                tmp_img = cv2.resize(t1, (160, 160))
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                tmp_img = (tmp_img / 255.).astype(np.float32)
                with torch.no_grad():
                    features1 = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
                features1 = features1.detach().numpy()

                doc.embedding = features1

                for d in self._da:
                    features1 = doc.embedding
                    features2 = d.embedding

                    similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
                    similarity_score = similarity_matrix[0][0]
                    print(d.uri, similarity_score)
                    if similarity_score > 0.7:
                        new_doc = Document()
                        new_doc.uri = d.uri
                        # new_doc.scores['cos'] = similarity_score
                        match_array.append(new_doc)
                    if len(match_array) > 0:
                        doc.matches.extend(match_array)

f = Flow().config_gateway(protocol='http', port=12346) \
    .add(name='face_embedding', uses=FaceEmbeddingExecutor)

with f:
    f.block()