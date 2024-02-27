import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import glob


from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



#loading multiple cvs files into single data frame
csv_files = glob.glob('Users/User/Documents/Interaction Lab/all_data/21*.csv')

print(csv_files)

combined_df = pd.DataFrame()

for csv_file in csv_files:
    df= pd.read_csv(csv_file,
                        #usecols=['x', 'y', 'z']
                        )
    combined_df=pd.concat([combined_df, df])

#print (combined_df)


