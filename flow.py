import logging
import os
import tempfile
import time

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
from tqdm import tqdm

from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from text2vec import SentenceModel, EncoderType
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
                resized_image = cv2.resize(face_image, (160, 160))

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
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
                    # 检查是否存在相同的 doc_id，如果存在则跳过重复插入
                    if not any(d.id == doc.id for d in self._da):
                        self._da.append(doc)
                    else:
                        print(f'Document with doc_id={doc.id} already exists in the index. Skipping insertion.')
                        logger.info(f'Document with doc_id={doc.id} already exists in the index. Skipping insertion.')
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
                resized_image = cv2.resize(face_image, (160, 160))

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
                    if similarity_score > 0.7:
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
                if similarity_score > 0.7:
                    match_doc = Document()
                    match_doc.id = d.id
                    match_doc.uri = d.uri
                    match_doc.scores['cosine'] = NamedScore(value=float(similarity_score))
                    match_doc.text = d.text
                    match_array.append(match_doc)
            doc.matches.extend(match_array)
            doc.summary()
        return docs


class FaceUpdate(Executor):
    print("in FaceUpdate")
    logger.info(f'FaceUpdate')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._da = DocumentArray(
            storage='sqlite', config={'connection': face_storage, 'table_name': 'face'}
        )

    @requests(on='/update')
    def search(self, docs: DocumentArray, *args, **kwargs):
        print("into update state")

        match_array = DocumentArray().empty()
        for doc in docs:
            logger.info(f'FaceUpdate received person name: {doc.text}')
            logger.info(f'FaceUpdate received doc_id: {doc.id}')


            if doc.id not in self._da:
                logger.info(f'doc_id not found in self._da')
                match_doc = Document()
                match_doc.tags['result'] = 'fail'
                match_array.append(match_doc)
                doc.matches.extend(match_array)
                return docs
            d = self._da[doc.id]
            new_doc = Document()
            new_doc.id = doc.id
            new_doc.text = doc.text
            new_doc.embedding = d.embedding
            new_doc.uri = d.uri
            self._da[doc.id] = new_doc
            self._da.sync()
            match_doc = Document()
            match_doc.tags['result'] = 'success'
            match_array.append(match_doc)
            doc.matches.extend(match_array)
            doc.summary()
            return docs

class CaptionEnglish(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    @requests(on="/img_caption")
    def caption(self, docs: DocumentArray, **kwargs):
        for doc in tqdm(docs):
            img_path = doc.uri
            print(img_path)
            raw_image = Image.open(img_path).convert('RGB')
            # unconditional image captioning
            inputs = self.processor(raw_image, return_tensors="pt")

            out = self.model.generate(**inputs)
            print(self.processor.decode(out[0], skip_special_tokens=True))
            doc.text = self.processor.decode(out[0], skip_special_tokens=True)

    @requests(on="/img_search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs

class EnglishToChineseTranslator(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode_name = 'kiddyt00/yt-tags-en-zh-v4'
        model = AutoModelForSeq2SeqLM.from_pretrained(mode_name)
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name)
        self.translation = pipeline("translation_en_to_zh", model=model, tokenizer=self.tokenizer)

    @requests(on='/img_caption')
    def encode(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            # 执行翻译并将结果添加到文档
            out_key = 'translation_text'
            # out_key = 'generated_text'
            translated_text = self.translation(doc.text, max_length=400)[0][out_key]



            doc.text = translated_text
            print(doc.summary())
            print(doc.text)

    @requests(on="/img_search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs

class ChineseEncoder(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = SentenceModel("shibing624/text2vec-base-chinese", encoder_type=EncoderType.FIRST_LAST_AVG,
                                    device=device)
        self._da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'images'}
        )

    @requests(on='/img_caption')
    def bar(self, docs: DocumentArray, **kwargs):
        print('start to index')
        print(f"Length of da_encode is {len(self._da)}")
        for doc in tqdm(docs):
            print("==============in embedding state")
            doc.embedding = self._model.encode(doc.text)
            print(doc.summary())
            with self._da:
                self._da.append(doc)
                self._da.sync()
                print(f"Length of da_encode is {len(self._da)}")
        print(f"Length of da_encode is {len(self._da)}")

    @requests(on='/img_search')
    def search(self, docs: DocumentArray, **kwargs):
        self._da.sync()  # Call the sync method
        print(f"Length of da_search is {len(self._da)}")
        for doc in docs:
            doc.embedding = self._model.encode(doc.text)
            print(doc.text)
            print(doc.summary())
            match = doc.match(self._da, limit=20, exclude_self=True, metric='cos', use_scipy=True)
            match.summary()
            print(type(doc.matches[:, ('scores__cos')]))
            print(doc.matches[:, ('text', 'uri', 'scores__cos')])
            print(doc.matches[0].scores['cos'])




f = Flow().config_gateway(protocol='http', port=8401) \
    .add(name='face_embedding', uses=FaceEmbeddingExecutor) \
    .add(name='cred_embedding', uses=CredEmbeddingExecutor) \
    .add(name='face_update', uses=FaceUpdate) \
    .add(name='img_caption', uses=CaptionEnglish) \
    .add(name='english_to_chinese', uses=EnglishToChineseTranslator, needs='img_caption') \
    .add(name='chinese_encode', uses=ChineseEncoder, needs='english_to_chinese')

with f:
    f.block()
