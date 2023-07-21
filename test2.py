import cv2
import numpy as np
from cnocr import CnOcr

def show(image, window_name):
    cv2.namedWindow(window_name, 0)
    cv2.imshow(window_name, image)
    # 0任意键终止窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


ocr = CnOcr(model_name='densenet_lite_136-gru')

image = cv2.imread('/home/etsme/Pictures/cred/333.jpg')
show(image, "image")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show(gray, "gray")
blur = cv2.medianBlur(gray, 7)
show(blur, "blur")
threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
show(threshold, "threshold")
canny = cv2.Canny(threshold, 100, 150)
show(canny, "canny")
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=5)
show(dilate, "dilate")
binary, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
show(res, "res")
contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
image_copy = image.copy()
res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
show(res, "contours")
epsilon = 0.02 * cv2.arcLength(contours, True)
approx = cv2.approxPolyDP(contours, epsilon, True)
n = []
for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
    n.append((x, y))
n = sorted(n)
sort_point = []
n_point1 = n[:2]
n_point1.sort(key=lambda x: x[1])
sort_point.extend(n_point1)
n_point2 = n[2:4]
n_point2.sort(key=lambda x: x[1])
n_point2.reverse()
sort_point.extend(n_point2)
p1 = np.array(sort_point, dtype=np.float32)
h = sort_point[1][1] - sort_point[0][1]
w = sort_point[2][0] - sort_point[1][0]
pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

M = cv2.getPerspectiveTransform(p1, pts2)
dst = cv2.warpPerspective(image, M, (w, h))
# print(dst.shape)
show(dst, "dst")
if w < h:
    dst = np.rot90(dst)
resize = cv2.resize(dst, (1084, 669), interpolation=cv2.INTER_AREA)
show(resize, "resize")
temp_image = resize.copy()
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
show(gray, "gray")
threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
show(threshold, "threshold")
blur = cv2.medianBlur(threshold, 5)
show(blur, "blur")
kernel = np.ones((3, 3), np.uint8)
morph_open = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
show(morph_open, "morph_open")
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(morph_open, kernel, iterations=6)
show(dilate, "dilate")
binary, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
resize_copy = resize.copy()
res = cv2.drawContours(resize_copy, contours, -1, (255, 0, 0), 2)
show(res, "res")
labels = ['姓名', '性别', '民族', '出生年', '出生月', '出生日', '住址', '公民身份证号码']
positions = []
data_areas = {}
resize_copy = resize.copy()
for contour in contours:
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)
    if h > 50 and x < 670:
        res = cv2.rectangle(resize_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = gray[y:(y + h), x:(x + w)]
        blur = cv2.medianBlur(area, 3)
        data_area = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        positions.append((x, y))
        data_areas['{}-{}'.format(x, y)] = data_area

show(res, "res")

positions.sort(key=lambda p: p[1])
result = []
index = 0
while index < len(positions) - 1:
    if positions[index + 1][1] - positions[index][1] < 10:
        temp_list = [positions[index + 1], positions[index]]
        for i in range(index + 1, len(positions)):
            if positions[i + 1][1] - positions[i][1] < 10:
                temp_list.append(positions[i + 1])
            else:
                break
        temp_list.sort(key=lambda p: p[0])
        positions[index:(index + len(temp_list))] = temp_list
        index = index + len(temp_list) - 1
    else:
        index += 1
for index in range(len(positions)):
    position = positions[index]
    data_area = data_areas['{}-{}'.format(position[0], position[1])]
    ocr_data = ocr.ocr(data_area)
    ocr_result = ''.join([''.join(result[0]) for result in ocr_data]).replace(' ', '')
    # print('{}：{}'.format(labels[index], ocr_result))
    result.append('{}：{}'.format(labels[index], ocr_result))
    show(data_area, "data_area")

for item in result:
    print(item)
show(res, "res")

