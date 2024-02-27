import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd

from glob import glob
import os


for PID in os.listdir ("C:/Users/User/Documents/Interaction Lab/compiled data"):
    print (PID)
    
PID = 25
files = glob(f"C:/Users/User/Documents/Interaction Lab/compiled data/{PID}/*.csv")
files.sort
print(files)

data=pd.concat((pd.read_csv(file) for file in files))
print(data)

# query all the points that have a particular in a column
spontaneous_points = data.query('condition=="spontaneous" and y < .35')
print(len(spontaneous_points))

df = spontaneous_points

#make array of spontaneous points
arrayx = np.array(spontaneous_points[['x','y','z']].values)
arrayy = np.array(spontaneous_points[['side']].values)

#create training set
N = int(.8*len(arrayx))
trainx = arrayx[:N]
trainy = arrayy[:N]

#create testing set
testx = arrayx[N:]
testy = arrayy[N:]

print(arrayx)
print(arrayy)

#makes logistic regression model
model = LogisticRegression()


#call model
model = LogisticRegression().fit(trainx, trainy)


#show r^2
r_sq = model.score(testx, testy)
arrayr2 = np.array(f"coefficient of determination: {r_sq}")
print(arrayr2)

#predict response
y_pred = model.predict(testx)
print(f"predicted response:\n{y_pred}")


from joblib import dump, load
dump(model, 'Test Model.joblib')

m = load('Test Model.joblib')

r_sq = m.score(testx, testy)
print(f"coefficient of determination: {r_sq}")
