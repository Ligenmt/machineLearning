import numpy as np
from PIL import Image
import os


# 工具类，批量将图片数值化，转化为data.txt

# RGB 转换为灰度图、二值化图
def image_to_gray():
    filedir = 'E:\\reportcaptcha\\train\\'
    list = os.listdir(filedir)

    for i in range(0, len(list)):
        filename = list[i]
        full_filename = filedir + filename
        print(full_filename)
        I = Image.open(full_filename)
        L = I.convert('1')   #转化为二值化图
        L.save('E:\\reportcaptcha\\binary\\' + filename)
        im_array = np.array(L)
        (row, col) = im_array.shape
        for i in range(row):
            for j in range(col):
                v = im_array[i][j]
                if v:
                    print('1', end=' ')
                else:
                    print('0', end=' ')
            print()

# 图片数据化
import json
def image_to_data():
    filedir = 'E:\\reportcaptcha\\train_data3\\'
    list = os.listdir(filedir)
    data_list = []
    for i in range(0, len(list)):
        filename = list[i]
        full_filename = filedir + filename
        print(full_filename)
        I = Image.open(full_filename)
        L = I.convert('1')   #转化为二值化图
        img_binary = np.array(L)
        (row, col) = img_binary.shape
        img_binary_list = img_binary.tolist()
        for r in range(row):
            for c in range(col):
                value = img_binary_list[r][c]
                if value:
                    img_binary_list[r][c] = [1]
                else:
                    img_binary_list[r][c] = [0]
        data = {'captcha': list[i].replace('.gif', ''), 'data': img_binary_list}
        data_list.append(data)
    json_str = json.dumps(data_list)
    with open("E:\\reportcaptcha\\train_data3.txt", 'w') as f:
        f.write(json_str)


# 显示数字化图片
def show_data():
    with open("data.txt", 'r') as f:
        content = f.read()
        data_list = json.loads(content)
        img_binary_list = data_list[0]['data']
        (row, col) = 25, 100
        for i in range(row):
            for j in range(col):
                v = img_binary_list[i][j][0]
                if v:
                    print('1', end=' ')
                else:
                    print('0', end=' ')
            print()


def remove_image_file():
    filedir = 'E:\\reportcaptcha\\samples\\'
    list = os.listdir(filedir)
    for i in range(0, len(list)):
        filename = list[i]
        if len(filename) < 10:
            print(filename)
            os.remove(filedir + filename)


# remove_image_file()
image_to_data()