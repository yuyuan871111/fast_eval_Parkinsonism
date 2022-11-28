import pandas as pd
import re, os, pdb
import numpy as np

def supply_zero_value_by_former(x):
    # to replace the zero value by the previous valid keypoints
    new_x = x.copy()
    for each_kpt in x.columns:
        for zero_value_index in x[x[each_kpt] == 0][each_kpt].index:
            new_x[each_kpt][zero_value_index] = find_non_zero_value(x[each_kpt], zero_value_index)
    return new_x

def find_non_zero_value(array_x, target_index): 
    # to find the closet valid value in the keypoint in previous timeframe
    # array_x: (timeframes, one keypoints)
    # target_index: the timeframe (index) which require to find the previous valid value
    find_index = target_index
    found = False
    while (not found):
        if array_x[find_index] != 0:
            found = True
        elif find_index == 0:
            found = True
        else:
            find_index -= 1

    return array_x[find_index]


def thumb_length(x,y,z=None, kpt_method="mediapipe-pd"):
    if kpt_method == "mediapipe-pd":
        # 3D distances of (keypoint 4 and 8)
        # x: pd.Dataframe (index = timestep), (x_0~x_20)
        # y: pd.Dataframe (index = timestep), (y_0~y_20)
        # z: pd.Dataframe (index = timestep), (z_0~z_20)
        for finger_idx in range(4):
            if finger_idx == 0:
                d2_x = (x[f"x_{finger_idx}"] - x[f"x_{finger_idx+1}"]).pow(2) 
                d2_y = (y[f"y_{finger_idx}"] - y[f"y_{finger_idx+1}"]).pow(2) 
                d2_z = (z[f"z_{finger_idx}"] - z[f"z_{finger_idx+1}"]).pow(2)
                d2 = d2_x + d2_y + d2_z
                d = d2.pow(0.5)
            else:
                d2_x = (x[f"x_{finger_idx}"] - x[f"x_{finger_idx+1}"]).pow(2) 
                d2_y = (y[f"y_{finger_idx}"] - y[f"y_{finger_idx+1}"]).pow(2) 
                d2_z = (z[f"z_{finger_idx}"] - z[f"z_{finger_idx+1}"]).pow(2)
                d2 = d2_x + d2_y + d2_z
                d = d + d2.pow(0.5)
        
        d = d[~(d == 0)] # thumb would not be 0.
        
        return np.median(d)
    elif kpt_method == "open-pose":
        # 2D distances of (keypoint 4 and 8)
        # x: pd.Dataframe (index = timestep), (x_0~x_20)
        # y: pd.Dataframe (index = timestep), (y_0~y_20)
        for finger_idx in range(4):
            if finger_idx == 0:
                d2_x = (x[f"x_{finger_idx}"] - x[f"x_{finger_idx+1}"]).pow(2) 
                d2_y = (y[f"y_{finger_idx}"] - y[f"y_{finger_idx+1}"]).pow(2) 
                d2 = d2_x + d2_y
                d = d2.pow(0.5)
            else:
                d2_x = (x[f"x_{finger_idx}"] - x[f"x_{finger_idx+1}"]).pow(2) 
                d2_y = (y[f"y_{finger_idx}"] - y[f"y_{finger_idx+1}"]).pow(2) 
                d2 = d2_x + d2_y
                d = d + d2.pow(0.5)
        
        d = d[~(d == 0)] # thumb would not be 0.
        
        return np.median(d)
    else:
        raise NotImplementedError


def reaxis(df: pd.DataFrame):
    '''
    df.columns: [timestamp, x_0~x_20, y_0~y_20, z_0~z_20]
    '''
    new_df = df['timestamp'].copy()
    directions = ['x', 'y', 'z']
    for each_direction in directions:
        data = df.filter(regex=f'{each_direction}_*')
        
        # reference point = the median of position of non-empty frames
        df_0 = df[f'{each_direction}_0']
        ref_point = df_0[df_0 != 0].median()
        
        data_processed = data.to_numpy() - np.full((len(df),21), ref_point)
        data_processed = pd.DataFrame(data_processed)
        data_processed.columns = data.columns
        new_df = pd.concat([new_df, data_processed], axis=1)

    return new_df

def normalize_by_thumbs(df: pd.DataFrame):
    '''
    df.columns: [timestamp, x_0~x_20, y_0~y_20, z_0~z_20]
    '''
    x = df.filter(regex="x_*")
    y = df.filter(regex="y_*")
    z = df.filter(regex="z_*")
    
    thumb_len = thumb_length(x,y,z,kpt_method="mediapipe-pd")
    
    timestamps = df['timestamp'] #backup timestamp
    df = df / thumb_len #normalizae
    df['timestamp'] = timestamps #write timestamp

    return df