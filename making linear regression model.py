import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



ax = plt.axes(projection="3d")

#loads cvs file
df=pd.read_csv(r'C:/Users/User/Documents/Interaction Lab/2022-10-20.csv')

#
# query all the points that have a particular in a column
spontaneous_points = df.query('condition=="spontaneous" and y < .35')
print(len(spontaneous_points))
#print(spontaneous_points[['x','y','z']].values)


#makes colored scatterplot using specific columns
df = spontaneous_points

#fg = ax.scatter3D(df['x'], df['y'], df['z'],
#                  color=np.where(df['side']=="r", 'r','b')
#                  )

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

#makes linear regression model
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

#show array of probablities
proba = model.predict_proba(testx)

#labels for axises
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
#plt.show()


from joblib import dump, load
dump(model, 'Test Model.joblib')

m = load('Test Model.joblib')

r_sq = m.score(testx, testy)
print(f"coefficient of determination: {r_sq}")
