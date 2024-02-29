from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def main():# 不加这句就会报错
# 损失函数为Wiouv3
# Create a new YOLO model from scratch注意力机制和替换主干网络DCNv2
    model = YOLO('D:/YOLOv8/ultralytics/cfg/models/v8/yolov8-C2f_DCNv2.yaml')

# Load a pretrained YOLO model (recommended for training)
    model = YOLO('D:/YOLOv8/yolov8n.pt')

    results = model.train(data='D:/YOLOv8/ultralytics/cfg/datasets/mycoco128.yaml',device=0, epochs=180)

# Evaluate the model's performance on the validation set
    results = model.val()

# Perform object detection on an image using the model
    results = model('D:/YOLOv8/data/namidata/split/predict/')

# Export the model to ONNX format
# success = model.export(format='onnx')
if __name__ == '__main__': # 不加这句就会报错
    main()# 不加这句就会报错

