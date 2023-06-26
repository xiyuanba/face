from docarray import DocumentArray, Document
from jina import Client

client = Client(port=12346, protocol='http')
# docs = DocumentArray()
# doc = Document(uri='/home/yingtie/PycharmProjects/image_index/images/*.jpg')
# docs.append(doc)
docs = DocumentArray.from_files('/home/yingtie/PycharmProjects/face/Yingfan_Wang_0001.jpg')
docs.summary()
da = client.post(on='/search', inputs=docs, show_progress=True, timeout=360000)
for d in da:
    d.summary()
da.summary()