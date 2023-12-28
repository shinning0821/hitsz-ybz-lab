from flask import Flask, request, jsonify, send_from_directory
import os
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from flask_cors import CORS

import argparse
from fcos_core.config import cfg
from fcos_core.data.transforms import build_transforms
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.miscellaneous import mkdir
from alignment import gen_inst_map
from vizer.draw import draw_boxes

app = Flask(__name__)
CORS(app)
# 加载PyTorch模型
# model = torch.load('your_model.pth')
# model.eval()


parser = argparse.ArgumentParser(description="DSSD Demo.")
parser.add_argument(
    "--config-file",
    # default= "fcos_test.yaml",
    default="cryonusegtest.yaml",
    # default="retinanet_test.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--ckpt", type=str, default='model.pth', help="Trained weights.")
parser.add_argument("--score_threshold", type=float, default=0.4)
args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.freeze()


# 图像预处理函数
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image.unsqueeze(0)


@torch.no_grad()
def seg_single_img(cfg, ckpt, score_threshold,image):
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")

    model = build_detection_model(cfg)
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt)['model'])
    model.eval()

    transforms = build_transforms(is_train=False)
    images = transforms(image)[0].unsqueeze(0)
    results = model(images.to(device))
    boxes = results[0][0].bbox.to(cpu_device).numpy()
    labels = results[0][0].get_field("labels").to(cpu_device).numpy()
    scores = results[0][0].get_field("scores").to(cpu_device).numpy()

    indices = scores > score_threshold
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]

    color1 = (0, 255, 0)  # 这里是绿色
    color2=  (255, 0, 0)  # 这里是红色

    draw_img = image.copy()

    i=0
    for box in boxes:
        x1,y1,x2,y2 = box
        i+=1
        if(i%3==0):
            cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), color1, thickness=2)
        else:
            cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), color2, thickness=2)

        # label = "Your Label Here"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.5
        # font_thickness = 1
        # font_color = (255, 255, 255)  # 文本颜色

        # # 计算文本位置
        # text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        # text_x = int(x1)
        # text_y = int(y1) - 5  # 调整文本的位置

        # # 绘制文本
        # cv2.putText(draw_img, label, (text_x, text_y), font, font_scale, font_color, font_thickness)
        

    print("222")
    # maskpred = torch.sigmoid(results[1][0])
    # maskpred = F.interpolate(maskpred, scale_factor=2)
    # maskpred = torch.argmax(maskpred,dim=1)
    # maskpred = maskpred.float().squeeze(0).squeeze(0)
    # maskpred = maskpred.data.cpu().numpy().astype(np.uint8)
    # dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    # inst_map = gen_inst_map(dets,maskpred)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    return draw_img

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        if file:
            # 读取上传的图像文件
            input_image = np.array(Image.open(file).convert('RGB'))
            # 预处理图像
            # 将输出转换为图像
            mask_img = seg_single_img(cfg, ckpt=args.ckpt, score_threshold=0.4,image=input_image)

            output_image = np.zeros_like(input_image)

            # # 遍历实例分割结果图像
            # for i in range(len(np.unique(mask_img))):  # 假设实例ID的范围是0-9
            #     # 为当前实例ID生成随机颜色
            #     color = tuple(np.random.randint(0, 256, size=3).tolist())
            #     # color = (0,49,83)
            #     # 根据实例ID绘制对应的颜色
            #     output_image[mask_img == i] = color
            # output_image[mask_img == 0] = 0
            
            output_image = mask_img
            os.makedirs("results", exist_ok=True)
            output_path = os.path.join("results", file.filename)
            cv2.imwrite(output_path, output_image)
            
            # 返回图像的URL
            return jsonify({"image_url": request.url_root + "results/" + file.filename})
            
        else:
            return jsonify({"error": "No file uploaded"})
    except Exception as e:
        print("111")
        return jsonify({"error": str(e)})

@app.route('/results/<filename>', methods=['GET'])
def send_image(filename):
    return send_from_directory("results", filename)

if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs("results")
    app.run(host='0.0.0.0', port=5000)
