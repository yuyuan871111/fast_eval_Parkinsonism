import json
import os

import cv2
import ffmpeg
import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from utils.hand.AD import cal_error_frame_ratio
from utils.hand.dataset import PDHandData
from utils.hand.keypoints import (finger_tapping_distance, moving_average,
                                  normalize_by_thumbs, peakFreqInte_bySTFT,
                                  reaxis)
from utils.hand.mediapipe_collect_hand_kpt import collect_hand_keypoints_pipe
from utils.hand.supp2emptytime import supp2emptytimestamp
from utils.seed import set_seed
# from utils.third_party.measurement import measurements
from utils.hand.util import get_model_by_name, parse_args_keypoint

# import pdb
# import warnings


def ffmpeg4format(video_path: str, output_path: str, overwrite_output: bool = True):
    '''
    video_path: your video path (ffmpeg available). e.g. '/your/path/test001.avi'
    output_path: your output path (mp4 preferred). e.g. '/your/path/output/test001.mp4
    overwrite_output: if existed files can be overwritten (default: True)
    '''
    os.makedirs('/'.join(output_path.split('/')[0:-1]), exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=60, round='up')
        .filter('scale', w=720, h=1280)
        .output(output_path)
        .run(overwrite_output=overwrite_output, quiet=True)
    )

    return None


def mp_kpts_generator(video_path: str, output_root_path: str, hand_query: str = 'Left', export_video: bool = False, logging: bool = False):
    '''
    video_path: your video path (.mp4 only). (video quality should be `fps=60, w=720, h=1280` or after `ffmpeg4format`) e.g. `/your/path/to/test0001.mp4`
    output_root_path: your output root path. e.g. `/your/path/to/folder`
    hand_query: please indicate which is the side of your hand. Choices = `Left` or `Right`
    export_video: whether to export the annotated video with hand keypoints
    '''
    assert '.mp4' in video_path, "Only mp4 video file is available."
    filename = video_path.split('/')[-1].replace('.mp4', '')
    hand_query = hand_query.capitalize()
    assert (hand_query == "Right") or (hand_query == "Left"), "Please indicate the handness"

    # skip existed files
    if os.path.isfile(f"{output_root_path}/{filename}_mp_hand_kpt.csv"):
        pass
    elif os.path.isfile(f"{output_root_path}/{filename}_mp_hand_kpt.thre0.csv"):
        pass
    elif os.path.isfile(f"{output_root_path}/{filename}_mp_hand_kpt.empty.csv"):
        pass
    else:
        video_output_path = f"{output_root_path}/{filename}_annot.mp4"
        try:
            try:
                # raise NotImplementedError # skip this part
                # main transformation: thershold=0.5
                kpt_output_path = f"{output_root_path}/{filename}_mp_hand_kpt.csv"
                annotated_images = collect_hand_keypoints_pipe(
                    video_path=video_path,
                    hand_query=hand_query,
                    output_path=kpt_output_path,
                    threshold=0.5,
                    logging=logging
                )
                if export_video:
                    frames2video(annotated_images=annotated_images, video_output_path=video_output_path)

            except Exception:
                # main transformation: thershold=0
                kpt_output_path = f"{output_root_path}/{filename}_mp_hand_kpt.thre0.csv"
                annotated_images = collect_hand_keypoints_pipe(
                    video_path=video_path,
                    hand_query=hand_query,
                    output_path=kpt_output_path,
                    threshold=0,
                    logging=logging
                )
                if export_video:
                    frames2video(annotated_images=annotated_images, video_output_path=video_output_path)

        except Exception as E:
            # merely return columns with no keypoints
            kpt_output_path = f"{output_root_path}/{filename}_mp_hand_kpt.empty.csv"

            with open(kpt_output_path, 'w') as ff:
                # write columns only (empty data)
                ff.write("timestamp,x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,x_18,x_19,x_20,y_0,y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_10,y_11,y_12,y_13,y_14,y_15,y_16,y_17,y_18,y_19,y_20,z_0,z_1,z_2,z_3,z_4,z_5,z_6,z_7,z_8,z_9,z_10,z_11,z_12,z_13,z_14,z_15,z_16,z_17,z_18,z_19,z_20")
                ff.write("\n")

            # annot video = original video (because it can't extract keypoints)
            if export_video:
                os.system(f"cp {video_path} {video_output_path}")

            # report to logs
            if logging:
                with open("run_mediapipe_exception_threshold_0.log", "a") as ff:
                    ff.write(str(E))
                    ff.write("\n")
                    ff.write(f"return empty csv: {kpt_output_path}")
                    ff.write("\n")
                    ff.write("-" * 50)
                    ff.write("\n")

    return None


def mp_kpts_preprocessing(input_filepath: str, output_filepath: str, logging=False):
    '''
    input_filepath: input mediapipe csv. e.g. `/your/path/to/a.csv`
    output_filepath: output filepath. e.g. `/your/path/to/output/a.csv`
    logging: True for logging
    '''
    assert '.csv' in input_filepath, "Only mediapipe csv file is available."
    filename = input_filepath.split('/')[-1].replace('.csv', '')

    # run
    try:
        # check error frame ratio
        error_frame_ratio = cal_error_frame_ratio(input_filepath)
        data = supp2emptytimestamp(input_filepath, mode='prev_frame', logging=logging)

        # reaxis the position
        data = reaxis(data)

        # normalized by thumbs
        data = normalize_by_thumbs(data)

        # save
        data.to_csv(output_filepath, index=None)

    except Exception as e:
        print(f"Processing: {filename}")
        print(e)
        raise

    return error_frame_ratio


def model_pred_severity(
    wkdir_path: str,
    test_data_path: str,
    test_map_path: str,
    hand_LR: str = "Left",
    hand_pos: int = 1,
    random_rotat_3d: bool = True,
    seed: int = 42,
):
    '''
    `test_data_path`: testing data root path
    `test_map_path`:
        testing csvname [col 0]
        label [col 1] (if existed, or just set 0 to all rows)
        error frame ratio map [col 2]
    `hand_LR`: 'Left' or 'Right'
    `hand_pos`: ('1': finger tapping, '2': open/close, '3': supination/pronation)
    '''
    # create model prefixs
    if hand_pos == 1:
        model_prefixs = [f'{hand_LR}_FG_{idx+1}' for idx in range(3)]
    elif hand_pos == 2:
        # TO DO: models required to be trained
        raise NotImplementedError
    elif hand_pos == 3:
        # TO DO: models required to be trained
        raise NotImplementedError
    else:
        raise NotImplementedError

    # prediction for 0/1+, 1-/2+, 2-/3+
    dfs = []
    for each_model_prefix in model_prefixs:
        model_path = f'{wkdir_path}/utils/saved_models/DrGuo_3d_rotat_val_pick/{each_model_prefix}/best.pth'
        args_path = f'{wkdir_path}/utils/saved_models/DrGuo_3d_rotat_val_pick/{each_model_prefix}/args.txt'
        _, df, _ = hand_pos_inference(
            test_data_path=test_data_path,
            test_map_path=test_map_path,
            model_path=model_path,
            args_path=args_path,
            multiple_sampling_num=5,
            random_rotat_3d=random_rotat_3d,
            class_num=2,
            seed=seed,
            device='cpu',
            logging=False,
            gau_samp=False
        )
        df.columns = [f"{each}_{each_model_prefix}" for each in df.columns]
        dfs.append(df)

    # merge results
    test_data_df_raw = pd.read_csv(test_map_path, header=None)  # read
    df = pd.concat(dfs, axis=1)
    df = df.filter(regex="predict")
    df['predict_overall'] = df.sum(axis=1, skipna=False)
    df = pd.concat([test_data_df_raw, df], axis=1)
    df.columns = ['csvname', 'label', 'error_frame_ratio'] + list(df.columns[3:])

    return df


def hand_pos_inference(
    test_data_path: str,
    test_map_path: str,
    model_path: str = "./utils/saved_models/test/best_R.pth",
    args_path: str = "./utils/saved_models/test/args_R.txt",
    output_folder: str = None,        # output folder
    device: str = 'cuda:0',
    seed: int = 42,
    balance_dataset: bool = False,
    class_num: int = None,
    multiple_sampling_num: int = 4,   # the number (at least) of multiple sampling from each video (data post-processing)
    random_rotat_3d: bool = False,
    gau_samp: bool = False,           # Use Gaussian sampling method for randomly cropping to avoid the head and tail missing.
    logging: bool = False,

):
    '''
    `test_data_path`: testing data root path
    `test_map_path`:
        testing csvname [col 0]
        label [col 1] (if existed, or just set 0 to all rows)
        error frame ratio map [col 2]
    `model_path`: model path, e.g. "./utils/saved_models/best_R.pth",
    `args_path`: model's args path, e.g. "./utils/saved_models/args_R.txt",
    `output_folder`: output path [str] or None (return cfg and results`pd.DataFrame`)
    `device`: use gpu or not 'cuda:0', 'cpu'
    `seed`: set seed for multiple sampling
    `balance_dataset`: set true to balance dataset
    `class_num`: set your class number or automatically counted from labels
    `multiple_sampling_num`: the number (at least) of multiple sampling from each video (data post-processing)
    `random_rotat_3d`: whether to use random rotate 3d
    `gau_samp`: set true to use Gaussian sampling method for randomly cropping to avoid the head and tail missing.
    `logging`: set true to show logging
    '''
    set_seed(seed)  # set seed

    # load previous (training setting)
    with open(args_path, 'r') as ff:
        cfg = json.load(ff)

    # setting
    cfg['test_data_path'] = test_data_path
    cfg['test_map_path'] = test_map_path
    cfg['model_path'] = model_path
    cfg['args_path'] = args_path
    cfg['output_folder'] = output_folder
    cfg['device'] = device
    cfg['seed'] = seed
    cfg['balance_dataset'] = balance_dataset
    cfg['multiple_sampling_type'] = 'random-crop'  # no other choice
    cfg['multiple_sampling_num'] = multiple_sampling_num
    cfg['random_rotat_3d'] = random_rotat_3d
    cfg['gau_samp'] = gau_samp
    cfg['logging'] = logging

    # it is no need to balance dataset when prediction only
    cfg['balance_dataset_method'] = 'None' if (cfg['balance_dataset'] == False) else 'random-crop'

    # read and preprocessing data map
    test_data_df_raw = pd.read_csv(test_map_path, header=None)  # read
    test_data_df = test_data_df_raw[~test_data_df_raw[0].str.contains("empty")]  # filter out empty
    if not cfg['low_confid_accept']:
        test_data_df = test_data_df[~test_data_df[0].str.contains("lowconfid")]  # filter out low confidence data
    test_data_df.reset_index(inplace=True, drop=True)  # reset index

    # enhanced features & keypoint selection
    # (Priority: enhanced features > keypoint selection)
    if cfg['enhanced_feat']:
        enhanced_type = cfg['category'].split("_")[-1]
        all_channels, channels_num = parse_args_keypoint(cfg['keypoint'])
        all_channels = all_channels + ["enhanced_feat"]
        channels_num = len(all_channels)
    else:
        enhanced_type = 0
        all_channels, channels_num = parse_args_keypoint(cfg['keypoint'])

    cfg['enhanced_type'] = enhanced_type
    cfg['channels_num'] = channels_num

    # classification class
    class_map = {each_class: i for i, each_class in enumerate(sorted(test_data_df[1].unique()))}
    if class_num is None:
        class_num = len(class_map.keys())

    # Check Acceptability domain: error_frame_ratio
    if cfg['logging']:
        print("\nChecking the error frame ratio...")
    origin_files_num = len(test_data_df)
    test_data_mask = test_data_df[2] <= cfg['error_frame_thres']  # set: error frame ratio <= threshold
    file_remained_ratio = sum(test_data_mask) / origin_files_num
    test_data_df = test_data_df[test_data_mask]
    if cfg['logging']:
        print(f"Files remains: {len(test_data_df)}/{origin_files_num} ({file_remained_ratio:.3f})")

    # check whether the file have been all filtered.
    if file_remained_ratio == 0:
        pred_list_test = []
        label_list_test = []
        csvname_list_test = []
    else:
        # dataset proccessing
        test_dataset = PDHandData(filename_label_df=test_data_df,
                                  input_channels=all_channels,
                                  data_root=test_data_path,
                                  seed=cfg['seed'],
                                  mk_balanced_dataset=cfg['balance_dataset'],
                                  mk_balanced_type=cfg['balance_dataset_method'],
                                  multi_sample_type=cfg['multiple_sampling_type'],
                                  multi_sample_num=cfg['multiple_sampling_num'],
                                  enhanced_type=cfg['enhanced_type'],
                                  random_rotat_3d=cfg['random_rotat_3d'],
                                  group_map=class_map,
                                  gaussian_sampling=cfg['gau_samp'],
                                  crop_len=cfg['crop_len'],
                                  return_name=True)

        test_loader = DataLoader(test_dataset,
                                 batch_size=cfg['batch_size'],
                                 shuffle=False,
                                 num_workers=1)

        # read model
        model = get_model_by_name(model_name=cfg['model'],
                                  n_classes=class_num,
                                  in_channels=cfg['channels_num'],
                                  crop_len=cfg['crop_len'],
                                  device=cfg['device'])
        model.load_state_dict(torch.load(cfg['model_path'], map_location=cfg['device']))
        model.to(cfg['device'])

        # prediction
        model.eval()
        with torch.no_grad():
            correct_cum = 0
            data_num = 0
            pred_list_test = []
            label_list_test = []
            csvname_list_test = []
            for i, (signals, labels, csvname) in enumerate(test_loader):
                if cfg['logging']:
                    print(
                        f"Val Processing... {i+1}/{len(test_loader)}            ",
                        end="\r")
                signals = signals.float().to(cfg['device'])
                labels = labels.long().to(cfg['device'])
                out = model(signals)
                _, preds = torch.max(out, 1)
                correct = torch.sum(preds == labels.data).cpu().numpy()
                correct_cum += correct
                data_num += signals.shape[0]

                pred_list_test += list(preds.cpu().numpy())
                label_list_test += list(labels.data.cpu().numpy())
                csvname_list_test += list(csvname)

            test_acc_epoch = correct_cum / data_num
            test_f1_epoch = f1_score(label_list_test, pred_list_test, average="weighted")
            test_recall_epoch = recall_score(label_list_test, pred_list_test, average="weighted", zero_division=0)

            if cfg['logging']:
                print(f"Raw test Acc {test_acc_epoch:.4f} F1 {test_f1_epoch:.4f} Recall {test_recall_epoch:.4f}")

    # post-processing
    results_df = pd.DataFrame(
        {"csvname": csvname_list_test,
         "label": label_list_test,
         "predict": pred_list_test})
    results_df = results_df.sort_values(by="csvname", ascending=True)
    results_df.reset_index(inplace=True, drop=True)

    clean_results = []
    for idx in range(len(test_data_df_raw)):
        each_row = test_data_df_raw.iloc[idx]
        each_csvname = each_row[0]      # 'csvname'
        each_label = each_row[1]        # 'label'
        each_errF_ratio = each_row[2]   # 'error_frame_ratio

        if 'empty' in each_csvname:
            clean_results.append([each_csvname, each_label, np.nan, 'empty (no keypoint extracted)'])

        elif (not cfg['low_confid_accept']) and ('lowconfid' in each_csvname):
            clean_results.append([each_csvname, each_label, np.nan, 'low-confid keypoints are not in applicability domain.'])

        elif each_errF_ratio > cfg['error_frame_thres']:
            clean_results.append([each_csvname, each_label, np.nan, f'error frame ratio is too high ({each_errF_ratio:.4f}).'])

        else:
            predict_list = results_df[results_df['csvname'] == each_csvname]['predict'].values
            final_predict = int(predict_list.mean().round())  # conservative

            clean_results.append([each_csvname, each_label, final_predict, predict_list])

    clean_results_df = pd.DataFrame(clean_results, columns=["csvname", "label", "predict", "note or details"])
    clean_results_df_woNA = clean_results_df.dropna(axis=0, how='any')

    # export results (either to file or return)
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        # write results
        clean_results_df.to_csv(f"{output_folder}/results.csv", index=None)

        # write cfg
        with open(f"{output_folder}/cfg.txt", 'w') as f:
            json.dump(cfg, f, indent=2)

        return None

    else:
        return cfg, clean_results_df, clean_results_df_woNA


def hand_rotation(data_test, rotat_axis="xyz", rotat_angle=[0, 90, 0]):
    r = R.from_euler(rotat_axis, rotat_angle, degrees=True)

    axis_num = pd.Series([each.split("_")[0] for each in data_test.columns if "_" in each]).unique()
    kpts_num = pd.Series([each.split("_")[-1] for each in data_test.columns if "_" in each]).unique()
    assert len(axis_num) == 3, "please include 3D keypoints"

    new_data_list = []
    for num in kpts_num:
        data = data_test[[f'x_{num}', f'y_{num}', f'z_{num}']]
        new_data = np.dot(r.as_matrix(), data.values.T)
        new_data_list.append(new_data)
    new_data = np.array(new_data_list)

    # new_data: (kpts_num, axis_num[x,y,z], timeframe)
    new_data = new_data.transpose((1, 0, 2))
    # new_data: (axis_num[x,y,z], kpts_num, timeframe)
    new_data = new_data.reshape((len(axis_num) * len(kpts_num), len(data_test)))
    # new_data: ([x1,x2,x3,...,y1,y2,y3,...,z1,z2,z3,...], timeframe)
    new_data = new_data.T
    # new_data: (timeframe, all_kpts)
    new_data = np.insert(new_data, 0, data_test['timestamp'].values, axis=1)
    # new_data: (timeframe, timestamp+all_kpts)
    new_data = pd.DataFrame(new_data, columns=data_test.columns)

    return new_data


def csv2json(csv_input_path: str, json_output_path: str, header=None):
    '''
    `csv_input_path`: the path of csv
    `json_output_path`: the path of json output
    '''
    data_input = pd.read_csv(csv_input_path, header=header)
    data_input.to_json(json_output_path, indent=2, orient='index')
    return None


def read_json(json_input_path: str):
    '''
    `json_input_path`: the path of json output
    '''
    data = pd.read_json(json_input_path, orient="index")
    data.columns = data.iloc[0].values
    data.drop(0, axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def hand_parameters(data_input: pd.DataFrame, hand_pos: int = 1):
    '''
    `data_input`: (timeframe, x0~x20, y0~20, z0~20)
    `hand_pos`: ('1': finger tapping, '2': open/close, '3': supination/pronation)
    return "results":
        distance-thumb-ratio: finger tapping
        frequency-interval: frequency interval of short-time Fourrier Transformation
        stft: short-time Fourrier Transformation
            time: timestamp
            freq: the frequency with max intensity in each timestamp
            intensity: the max intensity of the frequency in each timestamp
        freq-mean: mean of stft['freq']
        freq-std: standard deviation of stft['freq']
        freq-median: median of stft['freq']
        intensity-mean: mean of stft['intensity']
        intensity-std: standard deviation of stft['intensity']
        intensity-median: median of stft['intensity']
    '''
    results = {
        'stft': {},
        'peaks': {}
    }
    x = data_input.filter(regex="x_")
    y = data_input.filter(regex="y_")
    z = data_input.filter(regex="z_")

    # calcualte the distance between thumbs and index finger
    if hand_pos == 1:
        d2 = finger_tapping_distance(x, y, z, kpt_method='mediapipe-pd')
    elif hand_pos == 2:
        # TO DO: required to be calculated for some key parameters (e.g. area of open and close)
        raise NotImplementedError
    elif hand_pos == 3:
        # TO DO: required to be calculated for some key parameters
        raise NotImplementedError
    else:
        raise NotImplementedError

    d = d2.pow(0.5)
    d = moving_average(d, 5)  # average for each 5 frames (0.083s under 60fs)

    # short-time Fourier transform
    t, f, _, max_freq, max_intensity = peakFreqInte_bySTFT(
        d, fs=60, nperseg=150, noverlap=145,
        f_lower_cutoff=0.5, f_upper_cutoff=10
    )
    IF_value = max_intensity * max_freq

    # find peaks
    peaks, _ = signal.find_peaks(d, prominence=0.1)  # 10% of thumb length

    results['distance-thumb-ratio'] = d.tolist()
    results['frequency-interval'] = f.tolist()
    results['stft']['time'] = t.tolist()
    results['stft']['freq'] = max_freq
    results['stft']['intensity'] = max_intensity.tolist()
    results['stft']['inte-freq'] = IF_value.tolist()
    results['freq-mean'] = np.mean(max_freq)
    results['freq-std'] = np.std(max_freq)
    results['freq-median'] = np.median(max_freq)
    results['intensity-mean'] = np.mean(max_intensity)
    results['intensity-std'] = np.std(max_intensity)
    results['intensity-median'] = np.median(max_intensity)
    results['inte-freq-mean'] = np.mean(IF_value)
    results['inte-freq-std'] = np.std(IF_value)
    results['inte-freq-median'] = np.median(IF_value)
    results['peaks']['time'] = (peaks / 60).tolist()  # timeframe/60fps
    results['peaks']['peak_h'] = [] if results['peaks']['time'] == [] else d[peaks].tolist()
    results['peaks-mean'] = None if results['peaks']['time'] == [] else np.mean(d[peaks])
    results['peaks-std'] = None if results['peaks']['time'] == [] else np.std(d[peaks])
    results['peaks-median'] = None if results['peaks']['time'] == [] else np.median(d[peaks])

    return results


# export video
def frames2video(annotated_images, video_output_path):
    """
    export video
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{video_output_path}', fourcc, 60.0, (720, 1280))
    for annotated_image in annotated_images:
        out.write(cv2.flip(annotated_image, 1))
    out.release()

    return None
