from PbccrcCaptcha import pbccrc_captcha_model


X = pbccrc_captcha_model.image_to_input(path='E:\\reportcaptcha\\samples\\hfadg9.gif')
y_pred = pbccrc_captcha_model.model.predict(X)

print('captcha:', pbccrc_captcha_model.decode(y_pred))
