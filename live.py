import os
import cv2
import torch
import argparse
import numpy as np

from collections import deque

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor_live import CoTrackerPredictor

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

def preprocess_frame(frame):
    return np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture_device",
        type=int,
        default=0,
        help="device to capture video from",
    )
    # parser.add_argument(
    #     "--checkpoint",
    #     default="./checkpoints/cotracker_stride_4_wind_8.pth",
    #     help="cotracker model",
    # )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=0,
        help="Regular grid size"
    )

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.capture_device)
    if not cap.isOpened():
        print("Failed to open capture device")
        exit(1)
    
    S = 8

    model = CoTrackerPredictor(
        device=DEFAULT_DEVICE,
        checkpoint=args.checkpoint,
        grid_size=args.grid_size)

    model = model.to(DEFAULT_DEVICE)

    new_frame_counter = 0
    new_frame_req = S
    frame_buffer = deque(maxlen=S)

    vis = Visualizer(pad_value=120, linewidth=3)
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream closed")
            break

        frame = preprocess_frame(frame)
        frame_buffer.append(frame)

        new_frame_counter = (new_frame_counter + 1) % new_frame_req
        if new_frame_counter == 0:
            # Initially we have to wait for #window_size frames. Afterwards we only need the next #stride_size frames.
            new_frame_req = S // 2

            frames = np.stack(frame_buffer)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
            frames = frames.to(DEFAULT_DEVICE)

            pred_tracks, pred_visibility = model(frames)

            res_video = vis.visualize(frames[:,:S//2], pred_tracks[:,:S//2], pred_visibility[:,:S//2], save_video=False, query_frame=0).squeeze(0).permute(0, 2, 3, 1)

            # Convert to numpy and convert color
            res_video = np.array(res_video)
            res_video = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in res_video]

            for res_frame in res_video:
                cv2.imshow('Capture', res_frame)
                cv2.waitKey(40) # artificially fake a 25 fps frame rate

        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
