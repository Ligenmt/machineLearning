from PbccrcCaptcha import pbccrc_captcha_model
import os

#测试模型，使用图片文件进行测试

filedir = 'E:\\reportcaptcha\\samples\\'
list = os.listdir(filedir)
total = len(list)
right = 0
for i in range(0, total):
    X = pbccrc_captcha_model.image_to_input(path='E:\\reportcaptcha\\samples\\' + list[i])
    y_pred = pbccrc_captcha_model.model.predict(X)
    print('file:', list[i], 'captcha:', pbccrc_captcha_model.decode(y_pred), list[i][0:6] == pbccrc_captcha_model.decode(y_pred))
    if list[i][0:6] == pbccrc_captcha_model.decode(y_pred):
        right += 1
print(right, total)
