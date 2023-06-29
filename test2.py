import cv2
import face_recognition
import tempfile

# 加载图像
image = cv2.imread('yt_left.jpg')

# 使用 face_recognition 库检测人脸
face_locations = face_recognition.face_locations(image)

# 检测到人脸时进行处理
if len(face_locations) > 0:
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
        cv2.imwrite(f'output{i + 1}.jpg', saved_image)
else:
    print("未检测到人脸")
