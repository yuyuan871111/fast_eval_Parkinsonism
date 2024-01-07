import pandas as pd


def cal_error_frame_ratio(input_csv_path):

    # read csv
    data = pd.read_csv(input_csv_path)

    if len(data) == 0:  # empty case
        error_frame_ratio = 1

    elif data['timestamp'].values[-1] != (len(data) - 1):
        # file time length / expected timestamp
        error_frame_ratio = 1 - len(data) / data['timestamp'].values[-1]

    else:
        # 0: thumb-tip
        # 4: index-finger-tip
        # (x_0, x_4 == zero value) / expected timestamp
        error_frame_ratio = sum((data['x_0'] == 0) & (data['x_4'] == 0)) / len(data)

    return error_frame_ratio
