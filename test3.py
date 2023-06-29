import tempfile

import cv2
import dlib

# 加载图像
image = cv2.imread('yt_left.jpg')

# 使用 dlib 的人脸检测器检测人脸
detector = dlib.get_frontal_face_detector()
face_rects = detector(image, 1)

# 遍历检测到的人脸区域
for i, rect in enumerate(face_rects):
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()

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
    cv2.imwrite(f'output{i + 1}.jpg', saved_image)