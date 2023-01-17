import os, pandas as pd, numpy as np, json, pdb, argparse


from utils.hand.api import ffmpeg4format, mp_kpts_generator, mp_kpts_preprocessing, hand_pos_inference, model_pred_severity
from utils.hand.api import hand_parameters
from utils.seed import set_seed
from utils.hand.keypoints import stft_plot, mergePlot_PeakInteRaw

def main_batch(cfg):
    raise NotImplementedError


def main_single(cfg):
    # config
    filename = cfg['filename'] # no extension, name only
    ext = cfg['ext']
    input_root_path = cfg['input_root_path']
    output_root_path = cfg['output_root_path']
    hand_LR = cfg['hand_LR']
    hand_pos = cfg['hand_pos']
    wkdir_path = cfg['wkdir_path']

    video_path = f"{input_root_path}/{filename}.{ext}"
    assert os.path.exists(video_path), "Please check your video path."

    # Step 0
    try:
        video_output_path = f'{output_root_path}/{filename}.mp4'
        ffmpeg4format(video_path=video_path, output_path=video_output_path)
    except Exception as e:
        print(filename, ": ", e)

    # Step 1: mediapipe keypoint generation
    mp_kpts_generator(
        video_path=video_output_path, 
        output_root_path=output_root_path,
        hand_query=hand_LR, export_video=True, logging=False
    )
    ffmpeg4format(video_path=video_output_path.replace(".mp4", "_annot.mp4"), 
                  output_path=video_output_path.replace(".mp4", "_annot_.mp4"))
                  # encode to h264

    # Step 2: data preprocessing
    df_map_list = []
    try:
        try:
            csv_input_path = f"{output_root_path}/{filename}_mp_hand_kpt.csv"
            csv_output_path = f"{output_root_path}/{filename}_mp_hand_kpt_processed.csv"
            error_frame_ratio = mp_kpts_preprocessing(csv_input_path, csv_output_path, logging=False)

        except:
            csv_input_path = f"{output_root_path}/{filename}_mp_hand_kpt.thre0.csv"
            csv_output_path = f"{output_root_path}/{filename}_mp_hand_kpt_processed.thre0.csv"
            error_frame_ratio = mp_kpts_preprocessing(csv_input_path, csv_output_path, logging=False)

    except Exception as e:
        print(filename, ": ", e, ": Not predictable due to no keypoint extracted.")
        error_frame_ratio = 1

    df_map_list.append([csv_output_path.split("/")[-1], int(0), error_frame_ratio])
    df_map = pd.DataFrame(df_map_list)
    df_map.to_csv(f"{output_root_path}/{filename}_map.csv", index=None, header=None)

    # Step 3: prediction
    df_predict = model_pred_severity(
        wkdir_path=wkdir_path,
        test_data_path=output_root_path,
        test_map_path=f"{output_root_path}/{filename}_map.csv",
        hand_LR=hand_LR,
        hand_pos=hand_pos,
        random_rotat_3d=True,
        seed=42
    )
    df_predict.drop(["label"], inplace =True, axis=1)
    #df_predict.to_csv(f"{output_root_path}/{filename}_UPDRS_prediction.csv", index=None)
    df_predict.iloc[0].to_json(f"{output_root_path}/{filename}_UPDRS_prediction.json", indent=2)

    # Step 4: clean uneccasary files
    os.remove(f"{output_root_path}/{filename}_map.csv")
    os.rename(video_output_path.replace(".mp4", "_annot_.mp4"), video_output_path.replace(".mp4", "_annot.mp4"))

    # Step 5: traditional parameters
    results = {}
    csv_input_path = f'{csv_output_path}' # csv_output => generated by deep learning script
    json_output_path = f'{csv_output_path.replace(".csv", "")}_handparams.json'
    data_input = pd.read_csv(csv_input_path)

    try:
        results = hand_parameters(data_input=data_input)
        with open(json_output_path, "w") as ff:
            json.dump(results, ff, indent=2)

    except Exception as e:
        print(filename, ": ", e)

    # Step 6: plot for traditional parameters
    stft_plot(np.array(results["distance-thumb-ratio"]), png_filepath=f"{output_root_path}/{filename}_stft.png")
    mergePlot_PeakInteRaw(
        np.array(results["stft"]["time"]), 
        np.array(results["distance-thumb-ratio"]), 
        max_freq=np.array(results["stft"]["freq"]), max_intensity=np.array(results["stft"]["intensity"]),
        inte_ylim_max=0.5, png_filepath=f"{output_root_path}/{filename}_merge.png"
    )
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--wkdir_path',
                        type=str,
                        default='.',
                        help='where is the root path of this script')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='seed for the splitting')

    # data
    parser.add_argument('--filename',
                        type=str,
                        default="sample_fast",
                        help='the name of the video')
    parser.add_argument('--ext', 
                        type=str,
                        default='mp4',
                        help='the extension of the file')
    parser.add_argument('--hand_LR',
                        type=str,
                        default="Left",
                        choices=["Right", "Left"],
                        help="Which hand would you like to transform: [Left or Right]")
    parser.add_argument('--hand_pos',
                        type=int,
                        default=1,
                        choices=[1, 2, 3],
                        help="Which hand feature would you like to transform: [1: finger tapping, 2: open/close, 3: supination/pronation]")
    parser.add_argument('--input_root_path',
                        type=str,
                        default="./",
                        help="the root path of the input")
    parser.add_argument('--output_root_path',
                        type=str,
                        default="./sample_output",
                        help="the root path of the input")                    
    parser.add_argument('--mode',
                        type=str,
                        default="single",
                        choices=["single", "batch"], 
                        help="Single mode or batch mode to transform your data.")
    # other
    parser.add_argument('--logging',
                        action="store_true",
                        help='if logging or not')
    args = parser.parse_args()
    

    # Initialization: 
    cfg = args.__dict__ # args -> cfg
    set_seed(cfg['seed'])

    if cfg['mode'] == 'single':
        main_single(cfg)
    elif cfg['mode'] == 'batch':
        main_batch(cfg)
    else:
        raise NotImplementedError