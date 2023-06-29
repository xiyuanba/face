import tempfile

import numpy as np
import cv2
import face_recognition
from docarray.score import NamedScore
from jina import Executor, Flow, requests
from docarray import DocumentArray, Document

import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity


class CredEmbeddingExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self._da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'cred'}
        )

    @requests(on='/index_cred')
    def cred_encode(self, docs: DocumentArray, *args, **kwargs):
        # 创建一个 Document 对象并设置其属性值
        for doc in docs:
            print(doc.text)
            image = cv2.imread(doc.uri)

            tmp_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    @requests(on='/search_cred')
    def cred_search(self, docs: DocumentArray, *args, **kwargs):
        for doc in docs:
            image = cv2.imread(doc.uri)
            tmp_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tmp_img = (tmp_img / 255.).astype(np.float32)
            with torch.no_grad():
                features = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
            features = features.detach().numpy()
            doc.embedding = features
            match_array = DocumentArray().empty()  # 清空匹配结果
            for d in self._da:
                doc.chunks.clear()
                features1 = doc.embedding
                features2 = d.embedding
                similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
                similarity_score = similarity_matrix[0][0]
                print(d.uri, similarity_score)
                print(f'type of similarity_score:',type(similarity_score))
                if similarity_score > 0.65:
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

f = Flow().config_gateway(protocol='http', port=12346) \
    .add(name='cred_embedding', uses=CredEmbeddingExecutor)

with f:
    f.block()
