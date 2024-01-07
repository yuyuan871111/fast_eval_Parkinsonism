# import os
# import pdb
# import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


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


def thumb_length(x, y, z=None, kpt_method="mediapipe-pd"):
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

        d = d[~(d == 0)]  # thumb would not be 0.

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

        d = d[~(d == 0)]  # thumb would not be 0.

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

        data_processed = data.to_numpy() - np.full((len(df), 21), ref_point)
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

    thumb_len = thumb_length(x, y, z, kpt_method="mediapipe-pd")

    timestamps = df['timestamp']  # backup timestamp
    df = df / thumb_len  # normalizae
    df['timestamp'] = timestamps  # write timestamp

    return df


def plot_keypoints(x: pd.DataFrame, y: pd.DataFrame, c=0, n=8, factor=10, x_lim=[0.3, 1], y_lim=[0.55, 0.3]):
    '''
    x: pd.Dataframe (timestep, x0~x20)
    y: pd.Dataframe (timestep, y0~y20)
    c: plot from which frame
    n: the number of plotting frame
    factor: interval of plotting frame
    '''
    fig, axes = plt.subplots(n, 1, figsize=(10, 20))
    for i, ax in enumerate(axes):
        frame = i * factor + c
        ax.scatter(x.iloc[frame], y.iloc[frame])
        for j in range(21):
            ax.annotate(j, (x.iloc[frame][j], y.iloc[frame][j]))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)


def finger_tapping_distance(x, y, z=None, kpt_method="mediapipe-pd"):
    '''
    kpt_method: ["mediapipe", "mediapipe-pd"]
        mediapipe:
            3D distances of (keypoint 4 and 8)
            x: np.array (timestep, x0~x20)
            y: np.array (timestep, y0~y20)
            z: np.array (timestep, z0~z20)
        return d2, d2_x, d2_y, d2_z

        mediapipe-pd:
            3D distances of (keypoint 4 and 8)
            x: pd.Dataframe (index = timestep), (x_0~x_20)
            y: pd.Dataframe (index = timestep), (y_0~y_20)
            z: pd.Dataframe (index = timestep), (z_0~z_20)
        return d2
    '''
    if kpt_method == "mediapipe":

        if z is not None:
            d2_x = np.power((x[4] - x[8]), 2)
            d2_y = np.power((y[4] - y[8]), 2)
            d2_z = np.power((z[4] - z[8]), 2)
            d2 = d2_x + d2_y + d2_z

        else:
            d2_x = np.power((x[4] - x[8]), 2)
            d2_y = np.power((x[4] - x[8]), 2)
            d2_z = 0
            d2 = d2_x + d2_y + d2_z

        return d2, d2_x, d2_y, d2_z

    elif kpt_method == "mediapipe-pd":

        d2 = (x["x_4"] - x["x_8"]).pow(2) + (y["y_4"] - y["y_8"]).pow(2) + (z["z_4"] - z["z_8"]).pow(2)
        return d2

    else:
        raise NotImplementedError


def interest_region(timestamp: np.array, signals: np.array, t_lower_cutoff=2, t_higher_cutoff=-2):
    """
    timestamp: np.array (1 dimension)
    signals: np.array (1 dimension)
    t_lower_cutoff: int (sec)
    t_higher_cutoff: int (sec) (if negative: last few seconds)
    """
    assert len(timestamp) == len(signals), "Not same length in timestamp and signals."
    timestamp = np.array(timestamp)
    signals = np.array(signals)

    if t_higher_cutoff >= t_lower_cutoff:
        pass
    else:
        t_higher_cutoff = timestamp[-1] + t_higher_cutoff

    time_interest_idx = np.where((timestamp >= t_lower_cutoff) & (timestamp <= t_higher_cutoff))
    time_interest_idx = list(np.array(time_interest_idx).flatten())
    signals = signals[time_interest_idx]
    timestamp = timestamp[time_interest_idx]
    return timestamp, signals


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def stft_plot(sig, png_filepath=None, fs=60):
    amp = sig[(len(sig) // 3):(len(sig) // 2)].max()
    fig, axes = plt.subplots(3, 3, figsize=[16, 12])
    nperseg_list = [10, 30, 50, 70, 90, 110, 130, 150, 170]
    axes = axes.ravel()
    for idx, nperseg in enumerate(nperseg_list):
        f, t, Zxx = signal.stft(sig, fs, nperseg=nperseg, noverlap=nperseg * 2 // 3)
        # Zxx = np.log(Zxx)
        axes[idx].pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud', cmap="hsv")
        axes[idx].set_title(f'STFT Magnitude (nperseg={nperseg})')
        axes[idx].set_ylabel('Frequency [Hz]')
        axes[idx].set_xlabel('Time [sec]')
        axes[idx].set_ylim([0, 10])

    if png_filepath is None:
        fig.tight_layout()
        fig.show()
    else:
        fig.tight_layout()
        fig.patch.set_facecolor('white')
        fig.savefig(png_filepath, dpi=300)
    return None


def peakFreqInte_bySTFT(sig, fs=60, nperseg=150, noverlap=100, f_lower_cutoff=1, f_upper_cutoff=10):
    f, t, Zxx = signal.stft(sig, fs, nperseg=nperseg, noverlap=noverlap)
    Zxx = np.abs(Zxx)

    f_min_arg = np.argwhere(f > f_lower_cutoff)[0][0]
    f_max_arg = np.argwhere(f < f_upper_cutoff)[-1][0]
    f_range = f[f_min_arg:f_max_arg]

    max_intensity = Zxx[f_min_arg:f_max_arg].max(axis=0)
    max_freq = []
    for idx, _ in enumerate(max_intensity):
        max_freq.append(f_range[np.argmax(Zxx[f_min_arg:f_max_arg, idx])])

    return t, f, Zxx, max_freq, max_intensity


def sig_diff(signals, d2=True):
    signals_prev = list(signals.copy())
    signals_now = list(signals.copy())
    signals_prev.insert(-1, 0)
    signals_now.insert(0, 0)
    signals_diff = np.array(signals_now) - np.array(signals_prev)
    signals_diff = np.delete(signals_diff, -1)

    if d2:
        signals_diff = signals_diff**2
        signals_diff = np.sqrt(signals_diff)

    return signals_diff


def mergePlot_PeakInteRaw(
    t, sig, max_freq, max_intensity, error_frame_ratio=None,
    png_filepath=None, fs=60, inte_ylim_max=5e-2, freq_ylim_max=10
):
    fig, (ax1, ax3, ax4) = plt.subplots(3, 1, figsize=[10, 12])

    ax2 = ax1.twinx()
    ax1.plot(t, max_freq, 'g-')
    ax2.plot(t, max_intensity, 'b-')
    t_max = t.max() + 1

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Peak frequency (Hz)', color='g')
    ax1.set_ylim([0, freq_ylim_max])
    ax2.set_ylabel('Intensity', color='b')
    ax2.set_ylim([0, inte_ylim_max])
    if error_frame_ratio is not None:
        ax1.set_title(f"Error frame ratio: {error_frame_ratio:.3f}")
    ax1.set_xlim([-0.5, t_max])

    peaks, _ = signal.find_peaks(sig, prominence=0.1)  # 10% of thumb length
    ax3.plot(np.arange(len(sig)) / fs, sig)
    sig_upper = sig.max() * 1.1
    ax3.plot(peaks / fs, sig[peaks], "xr")
    ax3.set_ylim([0 - sig_upper / 20, sig_upper])
    ax3.set_ylabel('Tips distance/thumb length (-)')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim([-0.5, t_max])

    max_freq_diff = sig_diff(max_freq)
    ax4.plot(t, max_freq_diff)
    ax4.set_xlabel('Time (s)')
    ax4.set_xlim([-0.5, t_max])
    ax4.set_ylabel('Abosolute differences of peak frequency (Hz)')
    ax4.set_ylim([0, freq_ylim_max])

    if png_filepath is None:
        fig.tight_layout()
        fig.show()
    else:
        fig.tight_layout()
        fig.patch.set_facecolor('white')
        fig.savefig(png_filepath, dpi=300)

    return None
