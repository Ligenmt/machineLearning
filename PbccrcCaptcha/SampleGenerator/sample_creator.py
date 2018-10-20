import captcha
import requests
import base64
import time
import os
from PbccrcCaptcha.SampleGenerator.captcha_recognizor import CaptchaRecognizor

path = 'E:\\pbccrc\\samples\\'
isExists = os.path.exists(path)
# 判断结果
if not isExists:
    os.makedirs(path)
    print('创建文件夹%s成功' % path)

cr = CaptchaRecognizor()
url = ''
for i in range(10000):
    try:
        res = requests.get(url)
        body = res.json()
        img_base64 = body['data']['captcha']
        img_base64 = img_base64.replace('data:image/gif;base64,', '')
        value = cr.get_captcha(img_base64)
        print(value)
        if len(value) == 6:
            imgdata = base64.b64decode(img_base64)
            file = open(path + value + '.gif', 'wb')
            file.write(imgdata)
            file.close()
            time.sleep(2)
    except Exception as e:
        print(e)
