import numpy as np
import cv2
import face_recognition
from docarray.score import NamedScore
from jina import Executor, Flow, requests
from docarray import DocumentArray, Document

import torch
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
            print(doc.text)
            image = cv2.imread(doc.uri)
            face_locations = face_recognition.face_locations(image)

            # doc.mime_type = 'image/jpeg'  # 设置 Document 的 MIME 类型为 JPEG 图像
            # 遍历所有人脸区域，并将其添加到 Document 的 chunk 中
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                # 将人脸图像转换为 JPEG 格式，并创建一个新的子 Document 对象
                face_doc = Document(content=face_image, mime_type='image/jpeg')
                # 在同目录下创建一个新的 JPEG 图像文件，命名为 tmp<序号>.jpg
                filename = f'tmp{i + 1}.jpg'
                cv2.imwrite(filename, face_image)
                t1 = cv2.imread(filename)
                tmp_img = cv2.resize(t1, (160, 160))
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                tmp_img = (tmp_img / 255.).astype(np.float32)
                with torch.no_grad():
                    features = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
                features = features.detach().numpy()
                doc.embedding = features
                # print(doc.embedding)
                print(f'encode received person name :',doc.text)
                doc.summary()
                with self._da:
                    self._da.append(doc)
                    self._da.sync()

    @requests(on='/search')
    def search(self, docs: DocumentArray, *args, **kwargs):
        for doc in docs:
            image = cv2.imread(doc.uri)
            face_locations = face_recognition.face_locations(image)
            doc.content = image  # 设置 Document 的内容为原始图像
            doc.mime_type = 'image/jpeg'  # 设置 Document 的 MIME 类型为 JPEG 图像
            doc.chunks.clear()
            doc.matches.clear()
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                filename = f'tmp{i + 1}.jpg'
                cv2.imwrite(filename, face_image)
                t1 = cv2.imread(filename)
                tmp_img = cv2.resize(t1, (160, 160))
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                tmp_img = (tmp_img / 255.).astype(np.float32)
                with torch.no_grad():
                    features1 = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
                features1 = features1.detach().numpy()
                doc.embedding = features1
                match_array = DocumentArray().empty()  # 清空匹配结果
                for d in self._da:
                    doc.chunks.clear()
                    features1 = doc.embedding
                    features2 = d.embedding
                    similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
                    similarity_score = similarity_matrix[0][0]
                    print(d.uri, similarity_score)
                    print(f'type of similarity_score:',type(similarity_score))
                    if similarity_score > 0.7:
                        match_doc = Document()
                        match_doc.id = d.id
                        match_doc.uri = d.uri
                        # match_doc.scores['cosine'] = float(similarity_score)
                        match_doc.scores['cosine'] = NamedScore(value=float(similarity_score))
                        match_doc.text = d.text
                        print(match_doc.scores['cosine'],type(match_doc.scores['cosine']))
                        # match_doc.tags['cos'] = similarity_score
                        match_array.append(match_doc)
                doc.matches.extend(match_array)
            doc.summary()
            return docs

                # doc.embedding = features1
                # query_embedding = np.array(doc.embedding)
                # target_embeddings = self._da.embeddings.reshape(-1, 512)
                # # 将嵌入向量从三维数组转换为二维数组
                # query_embedding = query_embedding.reshape(-1, 512)
                # # 计算查询文档与目标文档之间的余弦相似度
                # for target_doc in self._da:
                #     target_doc.embedding = target_doc.embedding.reshape(-1, 512)
                #     similarity_score = cosine_similarity(query_embedding, target_doc.embedding)
                #     print(similarity_score)
                #     if similarity_score > 0.7:
                #         match_doc = Document(id=target_doc.id, embedding=target_doc.embedding.copy())
                #         match_doc.uri = target_doc.uri
                #         match_doc.tags['cos_score'] = similarity_score
                #         match_array.append(match_doc)
                #     if len(match_array) > 0:
                #         doc.matches.extend(match_array)

        # other_doc.embedding = features2  # 将第二个文档对象的嵌入向量设置为 features2
        # doc.match(DocumentArray([other_doc]), metric='cosine', limit=1)  # 计算两个文档对象之间的余弦相似度
        # similarity_matrix = doc.matches[0].scores['cosine']
        # doc.match(self._da, metric='cosine', limit=3)
        # for d in self._da:
        #
        #     features1 = doc.embedding
        #     features2 = d.embedding
        #
        #     similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
        #     similarity_score = similarity_matrix[0][0]
        #     print(d.uri, similarity_score)
        #     if similarity_score > 0.7:
        #         new_doc = Document()
        #         new_doc.uri = d.uri
        #         # new_doc.scores['cos'] = similarity_score
        #         match_array.append(new_doc)
        #     if len(match_array) > 0:
        #         doc.matches.extend(match_array)


f = Flow().config_gateway(protocol='http', port=12346) \
    .add(name='face_embedding', uses=FaceEmbeddingExecutor)

with f:
    f.block()
