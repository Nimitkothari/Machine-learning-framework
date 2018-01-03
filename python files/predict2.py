import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import pickle
path = os.getcwd()
print(path)
path = os.getcwd() + '/data/ext.txt'
data = pd.read_csv(path,header=None, names=['Size', 'Bedrooms', 'Age', 'Bathrooms', 'Price'])
#print("data",data)
print("data.shape",data.shape)
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
#print("train_set",train_set)
#print("test_set",test_set)
#print("train_set.shape",train_set.shape)
#print("train_set.shape",test_set.shape)
df_copy = train_set.copy()
print("df_copy",df_copy)
test_set_full = test_set.copy()
print("test_set_full",test_set_full)
test_set = test_set.drop(["Bedrooms"], axis=1)
print("test_set",test_set)
train_labels = train_set["Bedrooms"]
print("train_lables",train_labels)
train_set_full = train_set.copy()
train_set = train_set.drop(["Bedrooms"], axis=1)
lin_reg = LinearRegression()
print("linear model imported")
lin_reg.fit(train_set, train_labels)
print("linear fit done")
pickle.dump(lin_reg, open('predict2.pkl', 'wb'))
