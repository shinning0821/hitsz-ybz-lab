from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# 加载PyTorch模型
model = torch.load('your_model.pth')
model.eval()

# 图像预处理函数
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image.unsqueeze(0)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        if file:
            # 读取上传的图像文件
            image = Image.open(file)
            # 预处理图像
            input_image = preprocess_image(image)
            
            # 使用模型进行图像分割
            with torch.no_grad():
                output = model(input_image)
            
            # 将输出转换为图像
            output_image = transforms.ToPILImage()(output.squeeze())
            output_path = os.path.join("results", file.filename)
            output_image.save(output_path)
            
            # 返回图像的URL
            return jsonify({"image_url": request.url_root + "results/" + file.filename})
        else:
            return jsonify({"error": "No file uploaded"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/results/<filename>', methods=['GET'])
def send_image(filename):
    return send_from_directory("results", filename)

if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")
    app.run(host='0.0.0.0', port=5000)
