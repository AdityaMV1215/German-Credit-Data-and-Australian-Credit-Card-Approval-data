import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_german = pd.read_csv("german-credit-data.csv", header=None)
df_german = df_german.iloc[:,1:]

df_australian = pd.read_csv("australian-credit-approval.csv", header=None)
df_australian = df_australian.iloc[:,1:]

def scatter(df):
    for i in range(0,df.shape[1]-1):
        plt.scatter(df.iloc[:,-1], df.iloc[:,i])
        plt.title("Column {} vs last column".format(i))
        plt.show()
        plt.close()

def correlation(df):
    for i in range(0,df.shape[1]-1):
        print("Correlation of column {} with last column = {}".format(i, df.iloc[:,i].corr(df.iloc[:,-1])))

#correlation(df_australian)
#correlation(df_german)

a = np.array([1,2,3,4,5,6,7])
b = np.array([1,2,3,4,5,6,7])
c = np.append(a.reshape(-1,1),b.reshape(-1,1), axis=1)
print(c)
