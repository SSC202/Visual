import os
import xml.etree.ElementTree as ET

# 定义自己的类别，自己数据集有几类就填写几类 Define the classes
classes = ["DA"]

# 定义自己的输出文件夹 Define the output directory
output_dir = "S:/STM32SSC/Visual/Yolov5/Code/yolov5/VOCData/labels"

# 定义自己的输入文件夹 Define the input directory
input_dir = "S:/STM32SSC/Visual/Yolov5/Code/yolov5/VOCData/Annotations"

# 把每一个输入文件夹里的VOC格式的xml文件转换为yolo格式
# Loop through each xml file in the input directory and convert to yolo format
for file in os.listdir(input_dir):
    if file.endswith(".xml"):
        file_path = os.path.join(input_dir, file)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 获取照片的尺寸，这是转换计算需要的参数
        # Get the image size
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # 创建yolo格式文件
        # Create the yolo format file
        out_file = open(os.path.join(output_dir, file.replace("xml", "txt")), "w")

        # 遍历每个对象并写入yolo格式文件
        # Iterate over each object and write to the yolo format file
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (
                int(xmlbox.find("xmin").text),
                int(xmlbox.find("ymin").text),
                int(xmlbox.find("xmax").text),
                int(xmlbox.find("ymax").text),
            )

            bbx_w = (b[2] - b[0]) / float(width)
            bbx_h = (b[3] - b[1]) / float(height)
            bbx_x = (b[0] + b[2]) / 2.0 / float(width)
            bbx_y = (b[1] + b[3]) / 2.0 / float(height)

            out_file.write(
                str(cls_id)
                + " "
                + str(bbx_x)
                + " "
                + str(bbx_y)
                + " "
                + str(bbx_w)
                + " "
                + str(bbx_h)
                + "\n"
            )
        out_file.close()
