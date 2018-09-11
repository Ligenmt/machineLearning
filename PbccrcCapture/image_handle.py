import numpy as np
from PIL import Image
import os
# RGB 转换为灰度图、二值化图
filedir = 'E:\\reportcaptcha\\train\\'
list = os.listdir(filedir)

# for i in range(0, len(list)):
#     filename = list[i]
#     full_filename = filedir + filename
#     print(full_filename)
#     I = Image.open(full_filename)
#     L = I.convert('1')   #转化为二值化图
#     L.save('E:\\reportcaptcha\\binary\\' + filename)
    # im_array = np.array(L)
    # (row, col) = im_array.shape
    # for i in range(row):
    #     for j in range(col):
    #         v = im_array[i][j]
    #         if v:
    #             print('1', end=' ')
    #         else:
    #             print('0', end=' ')
    #     print()

# 图片数据化
import os
import json
fileurl = 'E:\\reportcaptcha\\train\\'
list = os.listdir(fileurl)
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
                img_binary_list[r][c] = 1
            else:
                img_binary_list[r][c] = 0
    data = {'captcha': list[i].replace('.gif', ''), 'data': img_binary_list}
    data_list.append(data)
json_str = json.dumps(data_list)
with open("E:\\reportcaptcha\\data.txt", 'w') as f:
    f.write(json_str)

# with open("E:\\reportcaptcha\\data.txt", 'r') as f:
#     content = f.read()
#     data_list = json.loads(content)
#     img_binary_list = data_list[0]['data']
#     (row, col) = 25, 100
#     for i in range(row):
#         for j in range(col):
#             v = img_binary_list[i][j]
#             if v:
#                 print('1', end=' ')
#             else:
#                 print('0', end=' ')
#         print()
