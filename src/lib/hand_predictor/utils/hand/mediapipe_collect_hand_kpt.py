import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def collect_hand_keypoints_pipe(video_path: str, hand_query, output_path: str=None, threshold=0.5, logging=False):
    '''
    video_path: [str] your video path
    hand_query: ["Left", "Right"] which hand you want to extract
    output_path: [str] your output csv
    threshold: [float, 0~1] confidence ratio you want to fix
    '''
    frames = read_video_data(video_path, logging=logging)
    time_frame, keypoints_list, _, annotated_images = collect_hand_keypoints(frames, hand_query, threshold=threshold, logging=logging, create_annotated_img=True) # hand_query: Right or Left
    t = pd.DataFrame(time_frame)
    x = pd.DataFrame(keypoints_list[0].T)
    y = pd.DataFrame(keypoints_list[1].T)
    z = pd.DataFrame(keypoints_list[2].T)

    keypoints_data = pd.concat([t,x,y,z], axis=1)
    keypoints_data.columns = ["timestamp"]+[f"x_{idx}" for idx in range(21)]+[f"y_{idx}" for idx in range(21)]+[f"z_{idx}" for idx in range(21)]
    if output_path is not None: keypoints_data.to_csv(output_path, index=False)
    if logging: print(f"{video_path}: DONE.")

    return annotated_images

# read video
def read_video_data(video_path: str, logging: bool=False):
    assert os.path.isfile(video_path), "Files doesn't exist."
    
    cap = cv2.VideoCapture(video_path)
    counter = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            if logging: print(f"Total frames: {counter}. Can't receive more frame (stream end?). Exiting ...")
            break
        counter += 1
        frames.append(frame)

    cap.release()
    
    return frames

# collect hand keypoints by mediapipe
def collect_hand_keypoints(frames, hand_query = "Right", create_annotated_img=False, threshold=0.5, logging=False):
    keypoints_list = []
    images_series = []
    fail_to_detect_hand = []
    time_frame = []

    with mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=threshold
        ) as hands:

        pbar = tqdm(frames) if logging else frames
        for idx, frame in enumerate(pbar):
            keypoints = []
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(frame, 1)
            #image_height, image_width, _ = image.shape

            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw hand world landmarks. (unit: meter)
            if not results.multi_hand_world_landmarks:
                fail_to_detect_hand.append(image)
                images_series.append(image)
                continue

            # drawing annotated image
            if create_annotated_img:
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                images_series.append(annotated_image)
                
            # Draw hand world landmarks. (unit: meter)
            for jdx, hand_world_landmarks in enumerate(results.multi_hand_world_landmarks):
                # Handness
                handedness = results.multi_handedness[jdx].classification[0].label
                
                if handedness == hand_query:
                    for point in mp_hands.HandLandmark:
                        keypoint = hand_world_landmarks.landmark[point]
                        keypoints.append([keypoint.x, keypoint.y, keypoint.z])
                    
                    keypoints_data = np.array(keypoints).T
                    keypoints_list.append(keypoints_data)
                    time_frame.append(idx)
                    
                    break

    keypoints_list = np.array(keypoints_list)
    keypoints_list = np.moveaxis(keypoints_list, 0, -1)
    # dimension of keypoints: (xyz, keypoints, timeframe) = (3, 21, timeframe)

    if create_annotated_img:
        return time_frame, keypoints_list, fail_to_detect_hand, images_series
    else:
        return time_frame, keypoints_list, fail_to_detect_hand

