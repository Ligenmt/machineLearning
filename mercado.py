import hashlib
import hmac
import requests
import json
import urllib



# Constantes
MB_TAPI_ID = 'bbfaf520a0811d1d8d1b2dd4e336ee1f'
MB_TAPI_SECRET = '8a77cbc78a4544c5d7ccd9ba9edc3a09e413a86c60fc966caba8bc456ec1df8c'
REQUEST_HOST = 'www.mercadobitcoin.net'
REQUEST_PATH = '/tapi/v3/'

import time
tapi_nonce = str(int(time.time()))
# tapi_nonce = 1
print(tapi_nonce)
# Parâmetros
params = {
    'tapi_method': 'list_orders',
    'tapi_nonce': tapi_nonce,
    'coin_pair': 'BRLBTC'
}
params = urllib.parse.urlencode(params)

# Gerar MAC
params_string = REQUEST_PATH + '?' + params
print(params_string)
H = hmac.new(MB_TAPI_SECRET.encode('utf-8'), digestmod=hashlib.sha512)
H.update(params_string.encode('utf-8'))
tapi_mac = H.hexdigest()
print(tapi_mac)
# Gerar cabeçalho da requisição
headers = {
    'Content-type': 'application/x-www-form-urlencoded',
    'TAPI-ID': MB_TAPI_ID,
    'TAPI-MAC': tapi_mac
}

# Realizar requisição POST

res = requests.post(url='https://' + REQUEST_HOST + REQUEST_PATH, data=params, headers=headers)
print(res.json())
