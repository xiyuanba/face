from docarray import DocumentArray, Document
from jina import Client

client = Client(port=12346, protocol='http')
# docs = DocumentArray()
# doc = Document(uri='/home/yingtie/PycharmProjects/image_index/images/*.jpg')
# docs.append(doc)
docs = DocumentArray.from_files('/home/yingtie/PycharmProjects/face/Vladimir_Putin/Vladimir_Putin_0030.jpg')
docs.summary()
matches = client.post(on='/search', inputs=docs, show_progress=True, timeout=360000)
for match in matches:
    # for m in match.matches:
    #     print(f"Match with URI {m.uri} has similarity score {m.scores['cos']}")
    match.summary()
matches.summary()