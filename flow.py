from pprint import pprint

from PIL import Image
import numpy as np
import cv2
import face_recognition
from jina import Document, Executor, Flow, DocumentArray, requests

import torch
from torchvision.models import resnet50

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from facenet_pytorch import InceptionResnetV1
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
                tmp_img = cv2.resize(filename, (224, 224))
                features1 = self.face_model.predict(np.array([tmp_img]))
                doc.embedding = features1
                print(doc.embedding)
                with self._da:
                    self._da.append(doc)
                    self._da.sync()


    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        print('search')
        for doc in docs:
            print(doc.uri)
        return docs
class FaceSimilarityExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(pretrained=True).to(self.device)
        self._da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'face'}
        )

    @requests(on='/search')
    def search(self, docs: DocumentArray, *args, **kwargs):
        for doc in docs:
            print(doc.uri)
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

                d = (
                    Document(uri=filename)
                    .load_uri_to_image_tensor()
                    .set_image_tensor_shape(shape=(224, 224))
                    .set_image_tensor_normalization()
                    .set_image_tensor_channel_axis(-1, 0)
                )
                doc.tensor = d.tensor
                embedding = self.embed_tensor(doc.tensor)
                doc.embedding = embedding

                doc.match(self._da, limit=20, exclude_self=True, metric='euclidean', use_scipy=True)
                for match in doc.matches:
                    print(match.uri, match.scores['euclidean'])

    def embed_tensor(self, tensor: np.ndarray) -> np.ndarray:
        # 在这里使用 ResNet50 模型将张量转换为嵌入向量，并返回向量的 numpy 数组形式
        with torch.no_grad():
            input_tensor = torch.from_numpy(tensor).unsqueeze(0)
            output_tensor = self.model(input_tensor)
            embedding = output_tensor.squeeze().numpy()
        return embedding



f = Flow().config_gateway(protocol='http', port=12346) \
    .add(name='face_embedding', uses=FaceEmbeddingExecutor) \
    .add(name='face_similarity', uses=FaceSimilarityExecutor,needs='face_embedding')

with f:
    f.block()