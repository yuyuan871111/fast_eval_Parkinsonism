import numpy as np
import pandas as pd


# hand keypoint extraction
def finger_tapping_distance(InputDataframe: pd.DataFrame, kpt_method="2D"):
    if kpt_method == "2D":
        # 2D distances of (keypoint 4 and keypoint 8)
        # x: pd.Dataframe (index = timestep), (x_0~x_20)
        # y: pd.Dataframe (index = timestep), (x_0~x_20)
        x = InputDataframe.filter(regex="x_")
        y = InputDataframe.filter(regex="y_")
        d_square = (x["x_4"] - x["x_8"]).pow(2) + (y["y_4"] - y["y_8"]).pow(2)

        # alternatives: distances of (keypoint 3 and keypoint 7)
        # d = (x["x_3"] - x["x_7"]).pow(2) + (y["y_3"]-y["y_7"]).pow(2)

        return np.sqrt(d_square)

    elif kpt_method == "3D":
        # 3D distances of (keypoint 4 and 8)
        # x: pd.Dataframe (index = timestep), (x_0~x_20)
        # y: pd.Dataframe (index = timestep), (y_0~y_20)
        # z: pd.Dataframe (index = timestep), (z_0~z_20)
        x = InputDataframe.filter(regex="x_")
        y = InputDataframe.filter(regex="y_")
        z = InputDataframe.filter(regex="z_")
        d_square = (x["x_4"] - x["x_8"]).pow(2) + (y["y_4"] - y["y_8"]).pow(2) + (z["z_4"] - z["z_8"]).pow(2)
        return np.sqrt(d_square)

    else:
        raise NotImplementedError
