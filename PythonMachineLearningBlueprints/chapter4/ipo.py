import numpy as np
import pandas as pd


ipos = pd.read_csv('ipo_data_0.csv', encoding='latin-1')
# print(ipos)
ipos = ipos.applymap(lambda x: x if not '$' in str(x) else x.replace('$', ''))
ipos = ipos.applymap(lambda x: x if not '%' in str(x) else x.replace('%', ''))
print(ipos.info())

ipos.replace(

)