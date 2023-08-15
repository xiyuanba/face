import logging
import os
import shutil
import tempfile

import face_recognition
from flask import Flask, jsonify, request
from jina import Client, DocumentArray, Document
import utils
import cv2

ip = '127.0.0.1'
print(ip)
app = Flask(__name__)

target_path = utils.get_images_path()
print(target_path)
img_tmp_path = utils.get_tmp_images_path()


logger = logging.getLogger('api_flask')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('/mnt/md0/log/services/etsai/flow.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

@app.route('/index', methods=['POST'])
def index_face():
    data = request.get_json()
    img_uri = data['img_url']
    name = data['name']
    filename = os.path.basename(img_uri)
    target_file = os.path.join(target_path, filename)
    shutil.copyfile(img_uri, target_file)
    logger.info(f'save {img_uri} to {target_file} with name:{name}')
    host = f"http://{ip}:8401"
    doc = Document(
        uri=img_uri,
        text=name
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    ca_1 = c.post(on='/', inputs=docs, show_progress=True, timeout=3600)
    print(ca_1.summary())
    for a in ca_1:
        # 提取 tags 属性中的所有键名
        tag_keys = list(a.tags.keys())
        print(f"Tags: {tag_keys}")
        result_list.append({
            'text': a.text,
            'doc_id': a.id,
            'type': a.tags['type']
        })
    ca_2 = c.post(on='/picture_index', inputs=docs, show_progress=True, timeout=3600)
    print(ca_2.summary())
    for b in ca_2:

        # 提取 tags 属性中的所有键名
        tag_keys = list(b.tags.keys())
        print(f"Tags: {tag_keys}")
        result_list.append({
            'text': b.text,
            'doc_id': b.id,
            'type': b.tags['type']
        })
    print(len(result_list))
    print(result_list)
    return jsonify(result_list)

@app.route('/search', methods=['GET'])
def picture_search():
    data = request.get_json()
    img_uri = data['img_url']
    host = f"http://{ip}:8401"


    result_total_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    doc = Document(
        uri=img_uri,
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    matches_1 = c.post(on='/picture_search', inputs=docs, show_progress=True, timeout=360000)
    for match in matches_1:
        for m in match.matches:
            print(f"Match with URI {m.uri} has similarity score {m.scores['cosine'].value}")
            res_text = m.text
            res_uri = m.uri
            res_score = m.scores['cosine'].value
            res_face_num = int(m.tags['face_num'])
            res_type = m.tags['type']
            print(type(res_score))
            #result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score}
            result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score, 'face_num': res_face_num, 'type': res_type}
            result_list.append(result_dict)

    image = cv2.imread(img_uri)
    # 使用 face_recognition 库检测人脸
    face_locations = face_recognition.face_locations(image)
    # 遍历检测到的人脸区域
    for i, face_location in enumerate(face_locations):
        print(f"Found face {i + 1} at top: {face_location[0]}, ")
        result_total_list = []
        result_list_face = []
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
        doc = Document(
            uri=filename,
        )
        docs = DocumentArray()
        docs.append(doc)
        matches = c.post(on='/search', inputs=docs, show_progress=True, timeout=360000)
        for match in matches:
            for m in match.matches:
                print(f"Match with URI {m.uri} has similarity score {m.scores['cosine'].value}")
                res_text = m.text
                res_uri = m.uri
                res_score = m.scores['cosine'].value
                res_face_num = int(m.tags['face_num'])
                res_type = m.tags['type']
                print(type(res_score))
                # result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score}
                result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score, 'face_num': res_face_num, 'type': res_type}
                result_list_face.append(result_dict)
        result_total_list.append(result_list_face)
    resp = {
        'face': result_total_list,
        'picture': result_list
    }
    # matches_2 = c.post(on='/search', inputs=docs, show_progress=True, timeout=360000)
    # for match in matches_2:
    #     for m in match.matches:
    #         print(f"Match with URI {m.uri} has similarity score {m.scores['cosine'].value}")
    #         res_text = m.text
    #         res_uri = m.uri
    #         res_score = m.scores['cosine'].value
    #         res_face_num = int(m.tags['face_num'])
    #         res_type = m.tags['type']
    #         print(type(res_score))
    #         #result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score}
    #         result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score, 'face_num': res_face_num, 'type': res_type}
    #         result_list.append(result_dict)
    return jsonify(resp)
# @app.route('/search', methods=['GET'])
# def search_face():
#     data = request.get_json()
#     img_uri = data['img_url']
#     host = f"http://{ip}:8401"
#
#
#     result_total_list = []
#     # 发送 POST 请求并获取响应数据
#     c = Client(host=host)
#     image = cv2.imread(img_uri)
#     # 使用 face_recognition 库检测人脸
#     face_locations = face_recognition.face_locations(image)
#     # 遍历检测到的人脸区域
#     for i, face_location in enumerate(face_locations):
#         print(f"Found face {i + 1} at top: {face_location[0]}, ")
#         result_list = []
#         top, right, bottom, left = face_location
#         # 截取人脸区域
#         face_image = image[top:bottom, left:right]
#
#         # 调整人脸区域大小为 160x160
#         resized_image = cv2.resize(face_image, (360, 360))
#
#         # 保存新图像到临时文件
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
#             temp_filename = temp_file.name
#             cv2.imwrite(temp_filename, resized_image)
#
#         # 从临时文件中读取图像并进行保存
#         saved_image = cv2.imread(temp_filename)
#         filename = os.path.join(img_tmp_path, f'index_tmp{i + 1}.jpg')
#         cv2.imwrite(filename, saved_image)
#         doc = Document(
#             uri=filename,
#         )
#         docs = DocumentArray()
#         docs.append(doc)
#         matches = c.post(on='/search', inputs=docs, show_progress=True, timeout=360000)
#         for match in matches:
#             for m in match.matches:
#                 print(f"Match with URI {m.uri} has similarity score {m.scores['cosine'].value}")
#                 res_text = m.text
#                 res_uri = m.uri
#                 res_score = m.scores['cosine'].value
#                 res_face_num = int(m.tags['face_num'])
#                 print(type(res_score))
#                 #result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score}
#                 result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score, 'face_num': res_face_num}
#                 result_list.append(result_dict)
#         result_total_list.append(result_list)
#     return jsonify(result_total_list)

@app.route('/update', methods=['PUT'])
def update_face():
    data = request.get_json()
    doc_id = data['doc_id']
    name = data['tag']
    host = f"http://{ip}:8401"
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    doc = Document(
        id=doc_id,
        text=name
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list_1 = {}
    result_list_2 = {}
    ca_1 = c.post(on='/update_face', inputs=docs, show_progress=True, timeout=360000)
    logger.info(f'update {doc_id} with name:{name}')

    for match in ca_1:
        for m in match.matches:
            print(f"Match with URI {m.tags['result']}")
            res_result = m.tags['result']
            logger.info(f'update {doc_id} with name:{name} result:{res_result}')
            result_list_1 = {
                'update': res_result,
                'doc_id': doc_id,
                'text': name
            }
    logger.info(f'face update {doc_id} with name:{name}')

    ca_2 = c.post(on='/update_picture', inputs=docs, show_progress=True, timeout=360000)
    for match in ca_2:
        for m in match.matches:
            print(f"Match with URI {m.tags['result']}")
            res_result = m.tags['result']
            logger.info(f'update {doc_id} with name:{name} result:{res_result}')
            result_list_2 = {
                'update': res_result,
                'doc_id': doc_id,
                'text': name
            }
    logger.info(f'picture update {doc_id} with name:{name}')
    resp = {
        'face': result_list_1,
        'picture': result_list_2
    }
    return jsonify(resp)
@app.route('/index_cred', methods=['POST'])
def index_cred():
    data = request.get_json()
    img_uri = data['img_url']
    name = data['name']
    host = f"http://{ip}:8401"
    doc = Document(
        uri=img_uri,
        text=name
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)

    ca = c.post(on='/index_cred', inputs=docs, show_progress=True, timeout=3600)
    print(ca.summary())
    for a in ca:
        # 提取 tags 属性中的所有键名
        tag_keys = list(a.tags.keys())
        print(f"Tags: {tag_keys}")
        result_list.append({
            'text': a.text,
            'doc_id': a.id
        })
    return jsonify(result_list)


@app.route('/search_cred', methods=['GET'])
def search_cred():
    data = request.get_json()
    img_uri = data['img_url']
    host = f"http://{ip}:8401"
    doc = Document(
        uri=img_uri
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    matches = c.post(on='/search_cred', inputs=docs, show_progress=True, timeout=360000)
    for match in matches:
        for m in match.matches:
            print(f"Match with URI {m.uri} has similarity score {m.scores['cosine'].value}")
            res_text = m.text
            res_uri = m.uri
            res_score = m.scores['cosine'].value
            print(type(res_score))
            result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score}
            result_list.append(result_dict)
    return jsonify(result_list)


@app.route('/img_caption', methods=['POST'])
def img_caption():
    data = request.get_json()
    img_uri = data['path']
    img_furl = data['furl']
    filename = os.path.basename(img_uri)
    target_file = os.path.join(target_path, filename)
    shutil.copyfile(img_uri, target_file)
    logger.info(f'copy {img_uri} to {target_file} success')
    host = f"http://{ip}:8401"
    doc = Document(
        uri=img_uri,
        tags={'furl': img_furl}
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    ca = c.post(on='/img_caption', inputs=docs, show_progress=True, timeout=3600)
    for a in ca:
        # 提取 tags 属性中的所有键名
        tag_keys = list(a.tags.keys())
        print(f"Tags: {tag_keys}")
        result_list.append({
            'text': a.text,
            'doc_id': a.id
        })
        logger.info(f'path {img_uri},furl {img_furl} with caption {a.text} success')
    return jsonify(result_list)


@app.route('/img_search', methods=['GET'])
def img_search():
    data = request.get_json()
    text = data['text']
    host = f"http://{ip}:8401"


    result_total_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    doc = Document(
        text=text,
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    matches = c.post(on='/img_search', inputs=docs, show_progress=True, timeout=360000)
    matches.summary()
    for match in matches:
        match.summary()
        match.matches.summary()
        for m in match.matches:
            res_text = m.text
            res_uri = m.uri
            res_score = 1 - m.scores['cos'].value
            res_furl = m.tags['furl']
            result_dict = {'text': res_text, 'path': res_uri, 'score': res_score, 'furl': res_furl}
            if res_score > 0.5:
                result_list.append(result_dict)
            print(f"Match with URI {m.uri} has similarity score {res_score}")
            logger.info(f"Match with path {m.uri},furl {m.tags['furl']} has similarity score {res_score}")

    logger.info(f'img_search {text} result:{result_list}')
    return jsonify(result_list)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8400)

