import logging
import os
import shutil

from flask import Flask, jsonify, request
from jina import Client, DocumentArray, Document
import utils

ip = '127.0.0.1'
print(ip)
app = Flask(__name__)

target_path = utils.get_images_path()
print(target_path)

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
    host = f"http://{ip}:12346"
    doc = Document(
        uri=img_uri,
        text=name
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    ca = c.post(on='/', inputs=docs, show_progress=True, timeout=3600)
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


@app.route('/search', methods=['GET'])
def search_face():
    data = request.get_json()
    img_uri = data['img_url']
    host = f"http://{ip}:12346"
    doc = Document(
        uri=img_uri
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    matches = c.post(on='/search', inputs=docs, show_progress=True, timeout=360000)
    for match in matches:
        for m in match.matches:
            print(f"Match with URI {m.uri} has similarity score {m.scores['cosine'].value}")
            res_text = m.text
            res_uri = m.uri
            res_score = m.scores['cosine'].value
            res_face_num = int(m.tags['face_num'])
            print(type(res_score))
            result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score, 'face_num': res_face_num}
            result_list.append(result_dict)
    return jsonify(result_list)


@app.route('/index_cred', methods=['POST'])
def index_cred():
    data = request.get_json()
    img_uri = data['img_url']
    name = data['name']
    host = f"http://{ip}:12346"
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
    host = f"http://{ip}:12346"
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
