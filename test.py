import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageCaptioning

# 加载模型和特征提取器
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageCaptioning.from_pretrained(model_name)

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# 生成图像描述
def generate_caption(image_path):
    # 加载图像并进行预处理
    image = Image.open(image_path)
    image = preprocess(image)

    # 使用特征提取器提取图像特征
    inputs = feature_extractor(images=image.unsqueeze(0), return_tensors="pt")

    # 使用模型生成图像描述
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # 解码生成的描述文本
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption


# 测试图像路径
image_path = 'test_image.jpg'

# 生成图像描述
caption = generate_caption(image_path)
print(caption)
