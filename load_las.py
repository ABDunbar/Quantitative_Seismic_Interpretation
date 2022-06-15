import glob
import os
import lasio as lasio
import numpy as np
import platform

my_os = platform.system()
if my_os == 'Windows':
    path = ".\data"
elif my_os == 'Linux':
    path = './data'

# get all paths and alphabetically ordered
paths = sorted(glob.glob(os.path.join(path, "*.las")))

well_df = [0] * len(paths)

for i in range(len(paths)):
  # read with lasio
  well = lasio.read(paths[i])
  # convert to dataframe
  df = well.df()
  # in this dataframe, depth is positioned as index, not as column
  # so better to change depth index to column
  well_df[i] = df.reset_index()
  well_df[i].rename(columns={'DEPT': 'DEPTH'}, inplace=True)
  well_df[i] = well_df[i].replace(-9999.0, np.NaN)

well1, well2, well3, well4, well5, well5_resist = well_df

# print(well1.head())
# print(well2.head())
print(well3.head())
# print(well4.head())
# print(well5.head())
# print(well5_resist.head())