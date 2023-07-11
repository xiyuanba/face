import logging
import os
import tempfile

import numpy as np
import cv2
import face_recognition
from PIL import Image
from docarray.score import NamedScore
from jina import Executor, Flow, requests
from docarray import DocumentArray, Document

import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('/mnt/md0/log/services/etsai/flow.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

storage = utils.get_storage_path()
face_storage = os.path.join(storage, "face")
cred_storage = os.path.join(storage, "cred")
print(f'face_storage:', face_storage)
print(f'cred_storage:', cred_storage)
img_tmp_path = utils.get_tmp_images_path()


class FaceEmbeddingExecutor(Executor):
    print("in FaceEmbeddingExecutor")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self._da = DocumentArray(
            storage='sqlite', config={'connection': face_storage, 'table_name': 'face'}
        )

    @requests(on='/')
    def encode(self, docs: DocumentArray, *args, **kwargs):
        print("in FaceEmbeddingExecutor encode")
        # 创建一个 Document 对象并设置其属性值
        for doc in docs:
            logger.info(f'encode received person name: {doc.text}')

            image = cv2.imread(doc.uri)
            # 使用 face_recognition 库检测人脸
            face_locations = face_recognition.face_locations(image)
            # 遍历检测到的人脸区域
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                # 截取人脸区域
                face_image = image[top:bottom, left:right]

                # 调整人脸区域大小为 160x160
                resized_image = cv2.resize(face_image, (360, 360))

                # 保存新图像到临时文件
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_filename = temp_file.name
                    cv2.imwrite(temp_filename, resized_image)

                # 从临时文件中读取图像并进行保存
                saved_image = cv2.imread(temp_filename)
                filename = os.path.join(img_tmp_path, f'index_tmp{i + 1}.jpg')
                cv2.imwrite(filename, saved_image)
                tmp_img = cv2.imread(filename)
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                tmp_img = (tmp_img / 255.).astype(np.float32)
                with torch.no_grad():
                    features = self.model(
                        torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
                features = features.detach().numpy()
                doc.embedding = features

                # 日志记录文档摘要等信息
                logger.info(f'Received person name: {doc.text}')
                logger.info(f'Document summary: {doc.summary()}')

                with self._da:
                    self._da.append(doc)
                    self._da.sync()


    @requests(on='/search')
    def search(self, docs: DocumentArray, *args, **kwargs):
        for doc in docs:
            logger.info(f'search received person name: {doc.text}')

            image = cv2.imread(doc.uri)
            # 使用 face_recognition 库检测人脸
            face_locations = face_recognition.face_locations(image)
            # 遍历检测到的人脸区域
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                # 截取人脸区域
                face_image = image[top:bottom, left:right]

                # 调整人脸区域大小为 160x160
                resized_image = cv2.resize(face_image, (360, 360))

                # 保存新图像到临时文件
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_filename = temp_file.name
                    cv2.imwrite(temp_filename, resized_image)

                # 从临时文件中读取图像并进行保存
                saved_image = cv2.imread(temp_filename)
                filename = os.path.join(img_tmp_path, f'search_tmp{i + 1}.jpg')
                cv2.imwrite(filename, saved_image)
                tmp_img = cv2.imread(filename)
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                tmp_img = (tmp_img / 255.).astype(np.float32)
                with torch.no_grad():
                    features = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
                features = features.detach().numpy()
                doc.embedding = features

                match_array = DocumentArray().empty()  # 清空匹配结果
                face_num = len(face_locations)
                for d in self._da:
                    doc.chunks.clear()
                    features1 = doc.embedding
                    features2 = d.embedding
                    similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
                    similarity_score = similarity_matrix[0][0]
                    logger.info(f'Similarity score for {d.uri}: {similarity_score}')
                    if similarity_score > 0.4:
                        match_doc = Document()
                        match_doc.id = d.id
                        match_doc.uri = d.uri
                        match_doc.scores['cosine'] = NamedScore(value=float(similarity_score))
                        match_doc.tags['face_num'] = face_num
                        match_doc.text = d.text
                        match_array.append(match_doc)

                doc.matches.extend(match_array)
            doc.summary()
            return docs


class CredEmbeddingExecutor(Executor):
    print("in CredEmbeddingExecutor")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self._da = DocumentArray(
            storage='sqlite', config={'connection': cred_storage, 'table_name': 'cred'}
        )

    @requests(on='/index_cred')
    def cred_encode(self, docs: DocumentArray, *args, **kwargs):
        print("in CredEmbeddingExecutor index_cred")
        print(f'len(docs):', len(docs))
        docs.summary()
        for doc in docs:
            print('for begin')
            logger.info(f'cred_encode received person name: {doc.text}')
            image = cv2.imread(doc.uri)
            resized_image = cv2.resize(image, (720, 454))

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_filename = temp_file.name
                # 根据文件的扩展名确定图像格式
            file_extension = os.path.splitext(doc.uri)[1].lower()
            print(file_extension)
            if file_extension == '.jpg' or file_extension == '.jpeg':
                print('in jpg state')
                # 使用Pillow库保存JPEG格式图像
                image_pil = Image.fromarray(resized_image)
                image_pil.save(temp_filename, format='JPEG')
            elif file_extension == '.png':
                print('in png state')
                # 使用Pillow库保存PNG格式图像
                image_pil = Image.fromarray(resized_image)
                image_pil.save(temp_filename, format='PNG')
            # 从临时文件中读取图像并进行保存
            saved_image = cv2.imread(temp_filename)
            filename = os.path.join(img_tmp_path, f'index_cred_tmp{file_extension}')
            cv2.imwrite(filename, saved_image)





            tmp_img = cv2.imread(filename)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = (tmp_img / 255.).astype(np.float32)
            with torch.no_grad():
                features = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
            features = features.detach().numpy()
            doc.embedding = features

            logger.info(f'Encode received person name: {doc.text}')
            doc.summary()
            print(f'Document with ID {doc.id}')
            with self._da:
                self._da.append(doc)
                self._da.sync()

    @requests(on='/search_cred')
    def cred_search(self, docs: DocumentArray, *args, **kwargs):
        for doc in docs:
            logger.info(f'cred_search received person name: {doc.text}')

            match_array = DocumentArray().empty()  # 清空匹配结果
            image = cv2.imread(doc.uri)
            resized_image = cv2.resize(image, (720, 454))

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_filename = temp_file.name
                # 根据文件的扩展名确定图像格式
            file_extension = os.path.splitext(doc.uri)[1].lower()
            print(file_extension)
            if file_extension == '.jpg' or file_extension == '.jpeg':
                print('in jpg state')
                # 使用Pillow库保存JPEG格式图像
                image_pil = Image.fromarray(resized_image)
                image_pil.save(temp_filename, format='JPEG')
            elif file_extension == '.png':
                print('in png state')
                # 使用Pillow库保存PNG格式图像
                image_pil = Image.fromarray(resized_image)
                image_pil.save(temp_filename, format='PNG')
                # 从临时文件中读取图像并进行保存
            saved_image = cv2.imread(temp_filename)
            filename = os.path.join(img_tmp_path, f'index_cred_tmp{file_extension}')
            cv2.imwrite(filename, saved_image)

            saved_image = cv2.imread(temp_filename)
            filename = os.path.join(img_tmp_path, 'search_tmp.jpg')

            cv2.imwrite(filename, saved_image)
            tmp_img = cv2.imread(filename)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = (tmp_img / 255.).astype(np.float32)
            with torch.no_grad():
                features = self.model(torch.from_numpy(np.array([tmp_img.transpose(2, 0, 1)])))
            features = features.detach().numpy()
            doc.embedding = features

            for d in self._da:
                features1 = doc.embedding
                features2 = d.embedding
                similarity_matrix = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
                similarity_score = similarity_matrix[0][0]
                logger.info(f'Similarity score for {d.uri}: {similarity_score}')
                if similarity_score > 0.8:
                    match_doc = Document()
                    match_doc.id = d.id
                    match_doc.uri = d.uri
                    match_doc.scores['cosine'] = NamedScore(value=float(similarity_score))
                    match_doc.text = d.text
                    match_array.append(match_doc)
            doc.matches.extend(match_array)
            doc.summary()
        return docs


f = Flow().config_gateway(protocol='http', port=8401) \
    .add(name='face_embedding', uses=FaceEmbeddingExecutor) \
    .add(name='cred_embedding', uses=CredEmbeddingExecutor)

with f:
    f.block()
