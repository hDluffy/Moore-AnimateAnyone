from src.dwpose import DWposeDetector
import os
from tqdm import tqdm
from pathlib import Path
from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np
from PIL import Image
import argparse
import json

def process_video(video_path, output_video_path, dst_size):
    fps = get_fps(video_path)
    vr = read_frames(video_path)

    first_frame = vr[0]
    height = first_frame.height
    width = first_frame.width
    
    dst_ratio=dst_size[1]/(dst_size[0]*1.0)
    src_ratio=height/(width*1.0)
    if src_ratio>dst_ratio:
        left=0
        upper=(height-width*dst_ratio)*0.5
        right=width
        lower=height-(height-width*dst_ratio)*0.5
        height = width*dst_ratio
    else:
        left=(width-height/dst_ratio)*0.5
        upper=0
        right=width-(width-height/dst_ratio)*0.5
        lower=height
        width = height/dst_ratio
    size = (int(width), int(height))
    print("video_path:",video_path)
    print("fps:",fps)
    print("size:",size)
    
    frames_crop=[]
    for idx in tqdm(range(len(vr)), desc=f"Processing {os.path.basename(video_path)}"):
        frame = vr[idx]
        frame_crop=frame.crop((left,upper,right,lower))
        frames_crop.append(frame_crop)

    save_videos_from_pil(frames_crop, output_video_path, fps=fps)    

def process_dwpose(dwprocessor, video_path, output_dwpose_video_path):
    fps = get_fps(video_path)
    vr = read_frames(video_path)

    first_frame = vr[0]
    height = first_frame.height
    width = first_frame.width
    
    size = (int(width), int(height))
    print("video_path:",video_path)
    print("fps:",fps)
    print("size:",size)
    
    dwpose_results=[]
    for idx in tqdm(range(len(vr)), desc=f"Processing {os.path.basename(video_path)}"):
        frame = vr[idx]
        detected_pose, _ = dwprocessor(frame)
        dwpose_results.append(detected_pose)

    save_videos_from_pil(dwpose_results, output_dwpose_video_path, fps=fps)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, default="")
    parser.add_argument("--output_path",  type=str, default="")
    parser.add_argument("--dst_w", type=int, default=512)
    parser.add_argument("--dst_h", type=int, default=768)
    parser.add_argument("--do_crop", default=False, action='store_true')
    parser.add_argument("--do_dwpose", default=False, action='store_true')
    args = parser.parse_args()

    src_path = args.input_path
    output_folder = args.output_path

    if args.do_crop:
        output_path = os.path.join(output_folder, "train")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        video_num=0
        for video_name in tqdm(os.listdir(src_path), desc=f"Processing"):
            video_path = os.path.join(src_path, video_name)
            output_video_path = os.path.join(output_path, '{:05d}'.format(video_num) + '.mp4')
            process_video(video_path, output_video_path, [args.dst_w,args.dst_h])
            video_num = video_num+1

        src_path = output_path
    

    if args.do_dwpose:
        dwprocessor = DWposeDetector()
        dwprocessor = dwprocessor.to(f"cuda")
        output_dwpose_path = os.path.join(output_folder, 'train_dwpose')
        if not os.path.exists(output_dwpose_path):
            os.makedirs(output_dwpose_path)
        
        meta_list=[]
        for video_name in tqdm(os.listdir(src_path), desc=f"Processing"):
            video_path = os.path.join(src_path, video_name)
            output_dwpose_video_path = os.path.join(output_dwpose_path, video_name)
            process_dwpose(dwprocessor, video_path, output_dwpose_video_path)

            meta_list.append({"video_path":video_path, "kps_path":output_dwpose_video_path})

        file_path = os.path.join("./data", output_folder.split("/")[-1]+".json")
        with open(file_path, 'w') as f:
            json.dump(meta_list, f,indent=4)
            

    