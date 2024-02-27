import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d


ax = plt.axes(projection="3d")

#loads cvs file
df=pd.read_csv(r'C:/Users/User/Documents/Interaction Lab/2022-10-20.csv')

#
# query all the points that have a particular in a column
spontaneous_points = df.query('condition=="spontaneous" and y < .35')
print(len(spontaneous_points))
print(spontaneous_points[['x','y','z']].values)


#makes colored scatterplot using specific columns
df = spontaneous_points

fg = ax.scatter3D(df['x'], df['y'], df['z'],
                  color=np.where(df['side']=="r", 'r','b')
                  )

#labels for axises
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()