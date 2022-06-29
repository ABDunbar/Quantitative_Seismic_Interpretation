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


def load():
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
        # replace null values (-9999.0) with NaN
        well_df[i] = well_df[i].replace(-9999.0, np.NaN)

    return well_df
