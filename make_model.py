import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import pickle

data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\IMAGE_GET\FINAL2022.csv")
data0 = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\IMAGE_GET\FINAL2023.csv")
data2 = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\IMAGE_GET\FINAL2021.csv")
data3 = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\IMAGE_GET\FINAL2020.csv")


c0 = np.array([data0.iloc[0]])
for i in range(91):
    x = np.array(data0.iloc[i+1])
    c0 = np.concatenate((c0,[x]),axis=0)


c1 = np.array([data.iloc[0]])
for i in range(data.shape[0]-1):
    x = np.array(data.iloc[i+1])
    c1 = np.concatenate((c1,[x]),axis=0)
    
c2 = np.array([data2.iloc[0]])
for i in range(199):
    x = np.array(data2.iloc[i+1])
    c2 = np.concatenate((c2,[x]),axis=0)
    
c3 = np.array([data3.iloc[0]])
for i in range(199):
    x = np.array(data3.iloc[i+1])
    c3 = np.concatenate((c3,[x]),axis=0)

print(len(c0))
print(len(c1))
print(len(c2))
print(len(c3))
    
averaged_data = np.zeros((200,17))
for i in range(183):
    for j in range(17):
        averaged_data[i,j] = (c1[i,j]*100/max(c1[:,j])+c2[i,j]*100/max(c2[:,j])+c3[i,j]*100/max(c3[:,j]))/3.0

# for i in range(92,184):
#     for j in range(5):
#         averaged_data[i,j] = (c1[i,j]*10/max(c1[:,j]) + c2[i,j]*10/max(c2[:,j]) + c3[i,j]*10/max(c3[:,j]))/3.0

for i in range(183, 200):
    for j in range(17):
        averaged_data[i,j] = (c2[i,j]*100/max(c2[:,j]) + c3[i,j]*100/max(c3[:,j]))/2.0

d = np.array([[1]])
for i in range(91):
    d = np.concatenate((d,[[i+1]]),axis=0)
    
sc = np.zeros((92,1))
for i in range(92):
    sc[i,0] = 0.3*(averaged_data[i,0]+averaged_data[i,1]+averaged_data[i,2]+averaged_data[i,3])+0.3*(averaged_data[i,4]+averaged_data[i,5]+averaged_data[i,6]+averaged_data[i,7])+0.2*(averaged_data[i,8]+averaged_data[i,9]+averaged_data[i,10]+averaged_data[i,11])+0.1*(averaged_data[i,12]+averaged_data[i,13]+averaged_data[i,14]+averaged_data[i,15])+0.1*averaged_data[i,16]

test = np.zeros((92,1))
c_test = c0[:92]
for i in range(92):
    test[i,0] = 0.3*(c_test[i,0]+c_test[i,1]+c_test[i,2]+c_test[i,3])+0.3*(c_test[i,4]+c_test[i,5]+c_test[i,6]+c_test[i,7])+0.2*(c_test[i,8]+c_test[i,9]+c_test[i,10]+c_test[i,11])+0.1*(c_test[i,12]+c_test[i,13]+c_test[i,14]+c_test[i,15])+0.1*c_test[i,16]

regr = linear_model.LinearRegression()
poly = PolynomialFeatures(10,interaction_only=False)
x = poly.fit_transform(sc)
regr.fit(x,d)

file_name = "rank_model.pkl"
file_name1="polynomial_transform.pkl"

pickle.dump(regr,open(file_name,"wb"))
pickle.dump(poly,open(file_name1,"wb"))