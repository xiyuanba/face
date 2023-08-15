import os
from docarray import DocumentArray, Document
from jina import Client

client = Client(port=8401, protocol='http')

docs = DocumentArray()

# folder_path = '/home/etsme/Downloads/archive/Celebrity Faces Dataset'
folder_path = '/home/etsme/Pictures/zhaopian'
sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
print(sub_folders)
for sub_folder in sub_folders:
    sub_folder_path = os.path.join(folder_path, sub_folder)
    print(sub_folder_path)
    for root, dirs, files in os.walk(sub_folder_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                print(image_path)

                doc = Document(uri=image_path, text=sub_folder)
                docs.append(doc)
#
# 对DocumentArray进行摘要打印
docs.summary()

#使用Client将DocumentArray发送给Jina服务端进行处理
da = client.post(on='/', inputs=docs, show_progress=True, timeout=360000)
da.summary()
ea = client.post(on='/picture_index', inputs=docs, show_progress=True, timeout=360000)
ea.summary()
