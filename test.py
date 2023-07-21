import pytesseract
from PIL import Image
import pandas as pd
import re

result = pytesseract.image_to_string(Image.open('/home/etsme/Pictures/cred/shimingka.jpg'), lang='chi_sim')
print(result)

# 删除空行和长度小于等于1的行
lines = result.split('\n')  # 按行分割结果
filtered_lines = [line.strip() for line in lines if len(line.strip()) > 1]  # 去除空格并保留长度大于1的行

# 使用正则表达式筛选出以中文开头的行
filtered_lines = [line for line in filtered_lines if re.match("^[\\u4e00-\\u9fa5]+", line)]

textdf = pd.DataFrame({"text": filtered_lines})
textdf["textlen"] = textdf.text.apply(len)

## 去除长度<=1的行
textdf = textdf[textdf.textlen > 1].reset_index(drop = True)

ocr_list = []
# 打印处理后的结果
for line in textdf.text:
    print(line)
    ocr_list.append(line.split(" ")[0])

print(ocr_list)
