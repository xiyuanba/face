from flask import Flask, jsonify,request
from jina import Client, DocumentArray, Document


app = Flask(__name__)


@app.route('/index', methods=['POST'])
def hello():
    data = request.get_json()
    img_uri = data['img_url']
    name = data['name']
    host = "http://172.66.1.189:12346"
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
        result_list.append({'text': a.text})
    return jsonify(result_list)


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    img_uri = data['img_url']
    host = "http://172.66.1.189:12346"
    doc = Document(
        uri=img_uri
    )
    docs = DocumentArray()
    docs.append(doc)
    result_list = []
    # 发送 POST 请求并获取响应数据
    c = Client(host=host)
    ca = c.post(on='/search', inputs=docs, show_progress=True, timeout=3600)
    matches = c.post(on='/search', inputs=docs, show_progress=True, timeout=360000)
    for match in matches:
        for m in match.matches:

            print(f"Match with URI {m.uri} has similarity score {m.scores['cosine'].value}")
            res_text = m.text
            res_uri = m.uri
            res_score = m.scores['cosine'].value
            result_dict = {'text': res_text, 'uri': res_uri, 'score': res_score}
            result_list.append(result_dict)
    # for a in ca:
    #     # 提取 tags 属性中的所有键名
    #     a.summary()
    #     res_text = a.text
    #     print(a.text)
    #     res_uri = a.uri
    #     print(a.uri)
    #     res_score = a.scores['cosine'].value
    #     print("The res_score  is {:.6f}".format(res_score))
    #     # print(f'res_text: {res_text},res_uri:{res_uri},res_score:"{:.6f}".format(res_score)')
    #     result_dict = {'text': a.text,'uri': res_uri,'score': res_score}
    #     result_list.append(result_dict)
    return jsonify(result_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
