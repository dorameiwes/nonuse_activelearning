import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d


ax = plt.axes(projection="3d")

#loads cvs file
df=pd.read_csv(r'C:/Users/User/Documents/Interaction Lab/2022-10-20.csv')

#makes scatterplot using specific columns
    #df2=df.loc[df['side'] == "r", 'x']
    #df3=df.loc[df['side'] == "r", 'y']
    #df4=df.loc[df['side'] == "r", 'z']
    #fg = ax.scatter3D(df['x'], df['y'], df['z'])

#plt.scatter(df['x'], df['y'], df['z'], 
#            color=np.where(df['side'], 'r','C0')
#           )

fg = ax.scatter3D(df['x'], df['y'], df['z'],
                  color=np.where(df['side']=="r", 'r','b')
                  )

#labels for axises
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()