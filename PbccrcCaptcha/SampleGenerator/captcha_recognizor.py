from PIL import Image
import numpy
import os
import base64
from io import BytesIO


class CaptchaRecognizor:

    def __init__(self):
        self.images = {}
        image_dir = 'images'

        list = os.listdir(image_dir)
        threshold = 50
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)

        for i in range(0, len(list)):
            path = os.path.join(image_dir, list[i])
            im = Image.open(path)
            imgry = im.convert('L')
            imgpoint = imgry.point(table, '1')
            key = list[i].replace('.png', '')
            self.images[key] = imgpoint
            # imgarray = imgry.load()
        print('CaptchaRecognizor inited')

    def get_captcha(self, img_base64):

        threshold = 127
        notok = True
        while notok:
            pfd = BytesIO(base64.b64decode(img_base64))
            img = Image.open(pfd)
            imgry = img.convert('L')

            table = []
            for i in range(256):
                if i < threshold:
                    table.append(0)
                else:
                    table.append(1)
            imgpoint = imgry.point(table, '1')
            (row, col) = numpy.array(imgpoint).shape
            if row != 25 or col != 100:
                return 'This is not a pbccrc img!'

            # imgpoint.show()
            # imgpoint.save('E:\\testreg.gif')
            x0, y0, x1, y1 = 0, 0, 0, 25

            result = {}
            for key in self.images:
                model = self.images[key]
                modelarray = model.load()
                (row, col) = numpy.array(model).shape
                # print(numpy.array(model).shape)

                for x in range(0, 100 - col):
                    # print("x", x)
                    region = (x, y0, x + col, y1)
                    cut = imgpoint.crop(region)
                    # print(numpy.array(cut).shape)
                    cutarray = cut.load()
                    yes = 0
                    no = 0
                    for i in range(col):

                        for j in range(row):
                            _j = 24 - j
                            pix_img = modelarray[i, j]
                            cut_img = cutarray[i, j]
                            if pix_img == cut_img:
                                yes += 1
                            else:
                                no += 1
                    if no < 4:
                        # print("%s at: %d, yes: %d, no: %d" % (key, x, yes, no))
                        # cut.show()
                        result[x] = key

            if (len(result) < 6):
                threshold += 5
            else:
                notok = False

            sortedkey = sorted(result.keys())
            values = ''
            for i in sortedkey:
                values += result[i]
            return values

# def init_images():
#     images = {}
#     image_dir = 'images'
#
#     list = os.listdir(image_dir)
#     threshold = 50
#     table = []
#     for i in range(256):
#         if i < threshold:
#             table.append(0)
#         else:
#             table.append(1)
#
#     for i in range(0, len(list)):
#         path = os.path.join(image_dir, list[i])
#         # print(path)
#         im = Image.open(path)
#         imgry = im.convert('L')
#         imgpoint = imgry.point(table, '1')
#         key = list[i].replace('.png', '')
#         images[key] = imgpoint
#         #imgarray = imgry.load()
#
#     return images
# images = init_images()



