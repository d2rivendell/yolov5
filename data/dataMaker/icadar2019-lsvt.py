import os
import json
from PIL import Image
"""
yolov5 format: class x_center y_center width height
"""

# 原始数据
images_paths = ["C:/Users/fander/Desktop/GitHub/datasets/ICDAR2019-LSVT/images/"]

label_path = "D:/ML/data/OCR/ICDAR2019-LSVT/train_full_labels.json"


# 生成数据

# 坐标
dst_label_path = "C:/Users/fander/Desktop/GitHub/datasets/ICDAR2019-LSVT/labels"

# 符合格式的可训练图片路径
train_img_path = "C:/Users/fander/Desktop/GitHub/datasets/ICDAR2019-LSVT/train_img"

def checkImgPath(path):
    return path.endswith(".jpg") or path.endswith(".JPG")

def creatPath(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_annotate(img_path, img_name, json):
    img = Image.open(img_path)
    m_w, m_h = img.size
    annotates = []
    infos = list(json[img_name])
    for info in infos:
        points = info['points']
        illegibility = bool(info['illegibility'])
        # 只获取不模糊且是四边形标注的图片
        if len(points) == 4 and illegibility == False:
            left_top = list(points[0])
            right_top = list(points[1])
            right_bottom = list(points[2])
            left_bottom = list(points[3])

            # 尽可能让文字包含在方形里面

            # 左上角的x、y需要最小
            left_top_x = min(left_top[0], left_bottom[0])
            left_top_y = min(left_top[1], right_top[1])

            # 右上角的x、y需要最大
            right_bottom_x = max(right_top[0], right_bottom[0])
            right_bottom_y = max(left_bottom[1], right_bottom[1])

            width = right_bottom_x - left_top_x
            height = right_bottom_y - left_top_y

            x = left_top_x + width / 2
            y = left_top_y + height / 2
            res = [x / m_w, y / m_h, width / m_w, height / m_h]
            if x < 0 or y < 0 or width < 0 or height < 0:
                continue
            res = list(map(lambda x: format(x, '.6f'), res))
            res.insert(0, '0')
            annotates.append(res)
    return  annotates


def main(roots, json):
    creatPath(dst_label_path)
    creatPath(train_img_path)
    train_paths = []
    fit_count = 0
    wrong_count = 0
    for root in roots:
        walk_path = os.path.normpath(root)
        for path, dir_list, file_list in os.walk(walk_path):
            for file_name in file_list:
                if checkImgPath(file_name):
                    img_path = os.path.join(path, file_name)
                    name = file_name.split('.')[0]
                    img_label_path = os.path.join(dst_label_path, name + ".txt")

                    # 移除存在的label
                    if os.path.exists(img_label_path):
                        os.remove(img_label_path)
                    # 获取图片中文本坐标
                    annotate = get_annotate(img_path, name, json)
                    if len(annotate) == 0:
                        wrong_count += 1
                        print("{} points标注不符合yolo格式！".format(img_path))
                        continue
                    fit_count += 1
                    train_paths.append(img_path)
                    # 坐标写入label
                    with open(img_label_path, 'w') as f:
                        for li in annotate:
                            res = ' '.join(li)
                            f.write(res)
                            f.write('\n')
    train_path = os.path.join(train_img_path, 'all.txt')
    with open(train_path, 'w') as f:
        for p in train_paths:
            f.write(p)
            f.write('\n')
    print("有{0}张图片符合yolo格式，有{1}张不符合！".format(fit_count, wrong_count))
    print("done!")

def make():
    """
    构造符合yolo格式的标注数据
    生产一个label文件目录，里面有对应同名的图片的标注数据
    生产一个图片列表文件，里面包含满足yolo标注格式的训练数据
    :return:
    """
    with open(label_path, 'r') as f:
         load_dict = json.load(f)
    main(images_paths, load_dict)

# for test
import cv2
import matplotlib.pyplot as plt
def box_image(labels, img_path):
    if len(labels) == 0:
        print("label is empty")
        return
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    red = (0, 0, 255)
    for label in labels:
      label = label.split()[1:]
      label = list(map(lambda x: float(x), label))
      x_center = label[0] * w
      y_center = label[1] * h
      width = label[2] * w
      height = label[3] * h
      left = (int(x_center - width / 2), int(y_center - height / 2))
      right = (int(x_center + width / 2), int(y_center + height / 2))
      cv2.rectangle(img, left, right, red)
    cv2.imshow('fff', img)
    cv2.waitKey()
      # plt.imshow(img)
      # plt.show()

def test(label_path, img_text_path):
    """
    测试生产的标注是否准确
    """
    with open(img_text_path, 'r') as f:
        for line in f.readlines():
            img_path = line.strip()
            dir, name = os.path.split(img_path)
            name = name.split('.')[0]
            img_label_path = os.path.join(dst_label_path, name + ".txt")
            labels = []
            with open(img_label_path, 'r') as lb_f:
                for w in lb_f.readlines():
                 label = w.strip()
                 labels.append(label)
            box_image(labels, img_path)

def split_image(train_img_text_path, train_path, test_path, test_size=0.1):
    """
    生产训练集和测试集，默认训练集：测试集 = 9:1
    """
    from sklearn.model_selection import train_test_split
    all_path = []
    with open(train_img_text_path, 'r') as f:
        for line in f.readlines():
            p = line.strip()
            all_path.append(p)
    train, test = train_test_split(all_path, test_size=test_size, random_state=1024)
    print('训练集有{0}，测试集有{1}'.format(len(train), len(test)))
    with open(train_path, 'w') as f:
        for t in train:
           f.write(t)
           f.write('\n')
    with open(test_path, 'w') as f:
        for t in test:
           f.write(t)
           f.write('\n')



if __name__ == '__main__':
    # 1
    # make()

    # 2
    # test(label_path, os.path.join(train_img_path, 'all.txt'))

    # 3
    yolo_train_path = "C:/Users/fander/Desktop/GitHub/datasets/ICDAR2019-LSVT/yolo_2019_lsvt_train.txt"
    yolo_test_path  = "C:/Users/fander/Desktop/GitHub/datasets/ICDAR2019-LSVT/yolo_2019_lsvt_test.txt"
    split_image(os.path.join(train_img_path, 'all.txt'), yolo_train_path, yolo_test_path)
