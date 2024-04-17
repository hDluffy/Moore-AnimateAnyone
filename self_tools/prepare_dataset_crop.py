from src.dwpose import DWposeDetector
import os
from tqdm import tqdm
from src.utils.util import get_fps, read_frames, save_videos_from_pil
import argparse
import json
import imageio
from self_tools.face_lmk import interface as lmk_inter
from .crop_halfbody import crop_video_dwpose

crop_type="half"
if crop_type=="half":
    crop_ratio=[ 1.5, -3.5, 0.5 ]
    stan_w=512
    stan_h=768
elif crop_type=="whole":
    crop_ratio=[ 2, -9.8, 0.5 ]
    stan_w=576
    stan_h=1024

def save_videos_imageio(videos: list, path: str, fps=25):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, videos, fps=float(fps))

def crop_whole2half_date():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_meta_path",  type=str, default="./data/video_train_data_3_5_wholebody_meta.json")
    parser.add_argument("--output_path",  type=str, default="")
    args = parser.parse_args()

    lmk_handle=lmk_inter.init_lmk()

    vid_meta = []
    vid_meta.extend(json.load(open(args.data_meta_path, "r")))
    
    new_meta_list=[]
    for video_paths in tqdm(vid_meta, desc="Processing "):
        train_video_path = video_paths["video_path"]
        train_dwpose_video_path = video_paths["kps_path"]
        
        video_name = train_video_path.split("/")[-1]
        video_subdir = train_video_path.split("/")[-2]
        dwpose_subdir = train_dwpose_video_path.split("/")[-2]

        output_video_subdir=os.path.join(args.output_path,video_subdir)
        output_dwpose_subdir=os.path.join(args.output_path,dwpose_subdir)
        if not os.path.exists(output_video_subdir):
            os.makedirs(output_video_subdir)
        if not os.path.exists(output_dwpose_subdir):
            os.makedirs(output_dwpose_subdir)

        video_list = read_frames(train_video_path)
        video_fps = get_fps(train_video_path)

        dwpose_list = read_frames(train_dwpose_video_path)
        dwpose_fps = get_fps(train_dwpose_video_path)

        video_rect, face_num=lmk_inter.get_faceRect(lmk_handle,video_list[0])
        if face_num!=1:
            continue
        # idx=1
        # while (face_num!=1):
        #     video_rect, face_num=lmk_inter.get_faceRect(lmk_handle,video_list[idx])
        #     idx=idx+1
        crop_frames_list,_=crop_video_dwpose(None,video_list,video_rect,crop_ratio,stan_w,stan_h)
        crop_dwpose_list,_=crop_video_dwpose(None,dwpose_list,video_rect,crop_ratio,stan_w,stan_h)
        save_videos_imageio(crop_frames_list,os.path.join(output_video_subdir,video_name),fps=video_fps)
        save_videos_imageio(crop_dwpose_list,os.path.join(output_dwpose_subdir,video_name),fps=dwpose_fps)

        new_meta_list.append({"video_path":os.path.join(output_video_subdir,video_name), "kps_path":os.path.join(output_dwpose_subdir,video_name)})
    
    file_path = os.path.join("./data", args.output_path.split("/")[-1]+".json")
    with open(file_path, 'w') as f:
        json.dump(new_meta_list, f,indent=4)

def crop_raw_date():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, default="")
    parser.add_argument("--output_path",  type=str, default="")
    args = parser.parse_args()

    lmk_handle=lmk_inter.init_lmk()
    detector = DWposeDetector()
    detector = detector.to(f"cuda")

    raw_video_path=args.data_path
    video_subdir = raw_video_path.split("/")[-1]
    dwpose_subdir = raw_video_path.split("/")[-1]+"_dwpose"

    output_video_subdir=os.path.join(args.output_path,video_subdir)
    output_dwpose_subdir=os.path.join(args.output_path,dwpose_subdir)
    if not os.path.exists(output_video_subdir):
        os.makedirs(output_video_subdir)
    if not os.path.exists(output_dwpose_subdir):
        os.makedirs(output_dwpose_subdir)
    
    new_meta_list=[]
    for video_name in tqdm(os.listdir(raw_video_path), desc=f"Processing"):
        video_path = os.path.join(raw_video_path, video_name)

        video_list = read_frames(video_path)
        video_fps = get_fps(video_path)

        video_rect, face_num=lmk_inter.get_faceRect(lmk_handle,video_list[0])
        if face_num!=1:
            continue
        # idx=1
        # while (face_num!=1):
        #     video_rect, face_num=lmk_inter.get_faceRect(lmk_handle,video_list[idx])
        #     idx=idx+1
        crop_frames_list,crop_dwpose_list=crop_video_dwpose(detector,video_list,video_rect,crop_ratio,stan_w,stan_h)

        save_videos_imageio(crop_frames_list,os.path.join(output_video_subdir,video_name),fps=video_fps)
        save_videos_imageio(crop_dwpose_list,os.path.join(output_dwpose_subdir,video_name),fps=video_fps)

        new_meta_list.append({"video_path":os.path.join(output_video_subdir,video_name), "kps_path":os.path.join(output_dwpose_subdir,video_name)})
    file_path = os.path.join("./data", args.output_path.split("/")[-1]+".json")
    with open(file_path, 'w') as f:
        json.dump(new_meta_list, f,indent=4)

if __name__ == "__main__":
    #crop_whole2half_date()
    crop_raw_date()