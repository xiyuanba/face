import streamlit as st
from PIL import Image
from jina import Client, DocumentArray, Document
import os
import shutil
import tempfile
import utils

ip = utils.get_local_ip()
print(ip)
img_path = utils.get_images_path()
print(img_path)
img_tmp_path = utils.get_tmp_images_path()
print(img_tmp_path)

st.set_page_config(page_title="搜索结果", page_icon=":mag:", layout="wide")
style = "<style>div.row-widget.stHorizontal{flex-wrap: nowrap !important;}</style>"
st.markdown(style, unsafe_allow_html=True)


# 上传文件并保存到指定目录中
def save_uploaded_file(uploaded_file, target_dir):
    # 如果没有上传文件，则直接返回 None
    if uploaded_file is None:
        return None

    # 创建临时文件，并将上传的二进制数据写入文件中
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.close()

        # 移动临时文件到指定目录中
        shutil.move(tmp_file.name, os.path.join(target_dir, uploaded_file.name))

    # 返回保存的文件路径
    return os.path.join(target_dir, uploaded_file.name)


# 创建上传组件
uploaded_file = st.file_uploader("图片", type=["png", "jpg", "jpeg", "gif"])
# text = uploaded_file.read().decode('utf-8')
# 添加上传按钮
if st.button('上传'):
    # 如果已经选择了文件，则进行处理
    if uploaded_file is not None:
        saved_file_path = save_uploaded_file(uploaded_file, img_path)
        st.success(f"图片已成功保存到 {saved_file_path}")

person_name = st.text_input('标签')

if st.button('人脸图像建立索引') and person_name is not None:
    # 从指定目录读取上传的文件内容
    image_uri = os.path.join(img_path, uploaded_file.name)
    doc = Document(
        uri=image_uri,
        text=person_name
    )
    doc.summary()
    docs = DocumentArray()
    docs.append(doc)
    docs.summary()
    # 发送 POST 请求并获取响应数据
    url = f"http://{ip}:8401"
    c = Client(host=url)
    da = c.post(on='/', inputs=docs, show_progress=True, timeout=3600)
    # text = uploaded_file.read().decode('utf-8')
    img = Image.open(image_uri)
    # 缩放图像大小为原来的一半
    width, height = img.size
    new_width, new_height = int(width / 2), int(height / 2)
    img = img.resize((new_width, new_height))
    da.summary()
    for d in da:
        st.image(img, caption=uploaded_file.name)
        st.write("图中显示为：", d.text)

if st.button('证件图像建立索引') and person_name is not None:
    # 从指定目录读取上传的文件内容
    image_uri = os.path.join(img_path, uploaded_file.name)

    doc = Document(
        uri=image_uri,
        text=person_name
    )
    doc.summary()
    docs = DocumentArray()
    docs.append(doc)
    docs.summary()
    # 发送 POST 请求并获取响应数据
    url = f"http://{ip}:8401"
    c = Client(host=url)
    da = c.post(on='/index_cred', inputs=docs, show_progress=True, timeout=3600)
    # text = uploaded_file.read().decode('utf-8')
    img = Image.open(image_uri)
    # 缩放图像大小为原来的一半
    width, height = img.size
    new_width, new_height = int(width / 2), int(height / 2)
    img = img.resize((new_width, new_height))
    da.summary()
    for d in da:
        st.image(img, caption=uploaded_file.name)
        st.write("图中显示为：", d.text)

uploaded_file_search = st.file_uploader("请上传需要搜索的图片", type=["png", "jpg", "jpeg", "gif"])
if st.button('人脸库匹配') and uploaded_file_search is not None:
    # 如果已经选择了文件，则进行处理
    if uploaded_file_search is not None:
        tmp_file_path = save_uploaded_file(uploaded_file_search, img_tmp_path)
        st.success(f"图片已成功保存到 {tmp_file_path}")
    image2_uri = os.path.join(img_tmp_path, uploaded_file_search.name)
    url = f"http://{ip}:8401"
    c = Client(host=url)
    da_search = DocumentArray()
    # t1 = Document(text=query_keyword)
    t1 = Document(
        uri=image2_uri
    )
    da_search.append(t1)
    print(da_search)
    matches = c.post('/search', inputs=da_search, limit=6, show_progress=True)

    # 显示输入的图片及相关信息
    if len(matches) == 0:
        st.warning("没有找到与该图片相似的结果。")
        col_input = st.columns(1)[0]
        with col_input:
            st.header("输入的图片")
            img_input = Image.open(image2_uri)
            width_input, height_input = img_input.size
            new_width_input, new_height_input = int(width_input / 2), int(height_input / 2)
            img_input = img_input.resize((new_width_input, new_height_input))
            st.image(img_input, width=new_width_input)
    else:
        col_input = st.columns(1)[0]
        with col_input:
            st.header("输入的图片")
            img_input = Image.open(image2_uri)
            width_input, height_input = img_input.size
            new_width_input, new_height_input = int(width_input / 2), int(height_input / 2)
            img_input = img_input.resize((new_width_input, new_height_input))
            st.image(img_input, width=new_width_input)

        # 显示匹配的图片及相关信息
        st.header("匹配到")
        cols = st.columns(5)
        for i, match in enumerate(matches[0].matches):
            # 打开图片并调整大小
            img = Image.open(match.uri)
            width, height = img.size
            new_width, new_height = int(width / 4), int(height / 4)
            img = img.resize((new_width, new_height))

            # 计算当前图片应该放在哪一列
            col_index = i % 5

            # 在对应的列中显示图片和相关信息
            with cols[col_index]:
                st.image(img, caption=match.text, use_column_width=True)
                st.write("余弦相似度: ", match.scores['cosine'].value)

if st.button('证件库匹配') and uploaded_file_search is not None:
    # 如果已经选择了文件，则进行处理
    if uploaded_file_search is not None:
        tmp_file_path = save_uploaded_file(uploaded_file_search, img_tmp_path)
        st.success(f"图片已成功保存到 {tmp_file_path}")
    image2_uri = os.path.join(img_tmp_path, uploaded_file_search.name)
    url = f"http://{ip}:8401"
    c = Client(host=url)
    da_search = DocumentArray()
    # t1 = Document(text=query_keyword)
    t1 = Document(
        uri=image2_uri
    )
    da_search.append(t1)
    print(da_search)
    matches = c.post('/search_cred', inputs=da_search, limit=6, show_progress=True)

    # 显示输入的图片及相关信息
    if len(matches) == 0:
        st.warning("没有找到与该图片相似的结果。")
        col_input = st.columns(1)[0]
        with col_input:
            st.header("输入的图片")
            img_input = Image.open(image2_uri)
            width_input, height_input = img_input.size
            new_width_input, new_height_input = int(width_input / 2), int(height_input / 2)
            img_input = img_input.resize((new_width_input, new_height_input))
            st.image(img_input, width=new_width_input)
    else:
        col_input = st.columns(1)[0]
        with col_input:
            st.header("输入的图片")
            img_input = Image.open(image2_uri)
            width_input, height_input = img_input.size
            new_width_input, new_height_input = int(width_input / 2), int(height_input / 2)
            img_input = img_input.resize((new_width_input, new_height_input))
            st.image(img_input, width=new_width_input)

        # 显示匹配的图片及相关信息
        st.header("匹配到")
        cols = st.columns(5)
        for i, match in enumerate(matches[0].matches):
            # 打开图片并调整大小
            img = Image.open(match.uri)
            width, height = img.size
            new_width, new_height = int(width / 4), int(height / 4)
            img = img.resize((new_width, new_height))

            # 计算当前图片应该放在哪一列
            col_index = i % 5

            # 在对应的列中显示图片和相关信息
            with cols[col_index]:
                st.image(img, caption=match.text, use_column_width=True)
                st.write("余弦相似度: ", match.scores['cosine'].value)
