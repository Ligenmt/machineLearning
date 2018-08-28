import pandas as pd
import os
import requests


# r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
# with open('iris.csv', 'w') as f:
#     f.write(r.text)

df = pd.read_csv('iris.csv', names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类别'])
# print(df)
# print(df['花萼长度'])

#选择前两列和前四行
# print(df.ix[:3, :2])

#只选择描述宽度的列
# print(df.ix[:3, [x for x in df.columns if '宽度' in x]])
#
# print(df['类别'].unique())
#
# print(df[df['类别'] == 'Iris-versicolor'])
#
# print(df.describe())
from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


clf = RandomForestClassifier(max_depth=5, n_estimators=10)

X = df.ix[:, :4]
y = df.ix[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
rf = pd.DataFrame(list(zip(y_pred, y_test)), columns=['predicted', 'actual'])
rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis=1)

print(rf)

ratio = rf['correct'].sum()/rf['correct'].count()
print(ratio)
