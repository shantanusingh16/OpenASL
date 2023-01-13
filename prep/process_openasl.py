import os
import json
import math
import cv2
import sys
import glob
import subprocess
import shutil
import tempfile
import argparse
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from p_tqdm import p_map
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


def crop_resize(imgs, bbox, target_size):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    if x1-x0<y1-y0:
        exp = (y1-y0-(x1-x0))/2
        x0, x1 = x0-exp, x1+exp
    else:
        exp = (x1-x0-(y1-y0))/2
        y0, y1 = y0-exp, y1+exp
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    left_expand = -x0 if x0 < 0 else 0
    up_expand = -y0 if y0 < 0 else 0
    right_expand = x1-imgs[0].shape[1]+1 if x1 > imgs[0].shape[1]-1 else 0
    down_expand = y1-imgs[0].shape[0]+1 if y1 > imgs[0].shape[0]-1 else 0
    rois = []
    for img in imgs:
        expand_img = cv2.copyMakeBorder(img, up_expand, down_expand, left_expand, right_expand, cv2.BORDER_CONSTANT, (0, 0, 0))
        roi = expand_img[y0+up_expand: y1+up_expand, x0+left_expand: x1+left_expand]
        roi = cv2.resize(roi, (target_size, target_size))
        rois.append(roi)
    return rois

def save_rois(rois, target_path):
    os.makedirs(target_path, exist_ok=True)
    decimals = 10
    write_params = [int(cv2.IMWRITE_PNG_COMPRESSION),9]
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(target_path, str(i_roi).zfill(decimals)+'.png'), roi, write_params)
        

def helper_fn(items, input_video_dir, output_dir, target_size, ffmpeg):
    # input_video_dir = '/scratch/shantanu/openasl/openas19' #'/ssd_scratch/users/mounika.k/openas01/openasl/'
    # output_dir = '/scratch/shantanu/openasl_cropped/openas19' #'/ssd_scratch/users/mounika.k/processed_openasl/'
    # target_size = 224
    # ffmpeg = 'ffmpeg' #'/home/shantanu.singh/ffmpeg-git-20220108-amd64-static/ffmpeg'

    vid, yid, start_time, end_time, bbox = items
    input_video_whole, output_dir = os.path.join(input_video_dir, yid+'.mp4'), os.path.join(output_dir, vid)
    if not os.path.exists(input_video_whole):
        return 0
    tmp_dir = tempfile.mkdtemp()
    input_video_clip = os.path.join(tmp_dir, 'tmp.mp4')
    cmd = [ffmpeg, '-ss', start_time, '-to', end_time, '-i', input_video_whole, '-c:v', 'libx264', '-crf', '20', input_video_clip]
    print(' '.join(cmd))
    subprocess.call(cmd)
    cap = cv2.VideoCapture(input_video_clip)
    frames_origin = []
    print(f"Reading video clip: {input_video_clip}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_origin.append(frame)
    if len(frames_origin) == 0:
        return -1  # Input does not exist or is corrupted.

    shutil.rmtree(tmp_dir)

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) == len(frames_origin):
        return 0 # Already exists
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    x0, y0, x1, y1 = bbox
    W, H = frames_origin[0].shape[1], frames_origin[0].shape[0]
    bbox = [int(x0*W), int(y0*H), int(x1*W), int(y1*H)]
    print(bbox, frames_origin[0].shape, target_size)
    rois = crop_resize(frames_origin, bbox, target_size)
    print(f"Saving ROIs to {output_dir}")
    save_rois(rois, output_dir)

    ## EXTRACT POSE USING MEDIAPIPE
    #extract_keypts(rois, vid)

    return 1 # Wrote successfully

def get_clip(input_video_dir, output_video_dir, tsv_fn, bbox_fn, target_size=224, ffmpeg=None):
    os.makedirs(output_video_dir, exist_ok=True)
    df = pd.read_csv(tsv_fn, sep='\t')
    vid2bbox = json.load(open(bbox_fn))
    items = []
    for vid, yid, start, end in zip(df['vid'], df['yid'], df['start'], df['end']):
        if vid not in vid2bbox:
            continue
        bbox = vid2bbox[vid]
        items.append([vid, yid, start, end, bbox])
    
    for item in items:
        helper_fn(item, input_video_dir, output_video_dir, target_size, ffmpeg)
    #p_map(helper_fn, items, num_cpus=10)

    # Download using process pool
    # futures = {}
    # with ProcessPoolExecutor(max_workers=5) as executor:
    #     with tqdm(total=len(items)) as progress_bar:
    #         futures = dict()
    #         for idx, (vid, yid, start_time, end_time, bbox) in enumerate(items):
    #             future = executor.submit(helper_fn, input_video_dir, output_video_dir, 
    #                 vid, yid, start_time, end_time, bbox, 
    #                 target_size, ffmpeg)
    #             futures[future] = idx

    #         results = [None] * len(items) # pre_allocate slots
    #         for future in as_completed(futures):
    #             idx = futures[future] # order of submission
    #             vid, _, _, _, _ = items[idx]
    #             results[idx] = {'vid': vid, 'result': future.result()}
    #             progress_bar.update(1) # advance by 1

    #         print(results)
    #         with open('/scratch/shantanu/crop_video_results.pickle', 'wb') as f:
    #             pickle.dump(results, f)

    return


def main():
    parser = argparse.ArgumentParser(description='download video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, help='data tsv file')
    parser.add_argument('--bbox', type=str, help='bbox json file')
    parser.add_argument('--raw', type=str, help='raw video dir')
    parser.add_argument('--output', type=str, help='output dir')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='path to ffmpeg')
    parser.add_argument('--target-size', type=int, default=224, help='image size')
    
    args = parser.parse_args()
    get_clip(args.raw, args.output, args.tsv, args.bbox, target_size=args.target_size, ffmpeg=args.ffmpeg)
   

if __name__ == '__main__':
    main()
