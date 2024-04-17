import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from PIL import Image
from src.dwpose import DWposeDetector
from .video_dwpose_align_image import process_align_video
from .crop_halfbody import crop_video_dwpose,crop_standard_img
from .pose2video_pipeline import load_pipeline, pose2vidpipeline
from src.utils.util import get_fps, read_frames, save_videos_from_pil
from tools.faceswap_ctypes import interface as fs_inter
from self_tools.face_lmk import interface as lmk_inter
import torch
import torchvision
import imageio
from einops import repeat,rearrange
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="./self_tools/animation.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args

def save_videos_grid_imageio(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=25):
    
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=float(fps))

def save_result_video(video, fps, ref_name,pose_name):
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_videos_grid_imageio(
            video,
            f"{save_dir}/{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_one.mp4",
            n_rows=3,
            fps=fps,
        )

def faceswap_video(fs_handle,video):
    c,fn,h,w=video[0].shape
    video_=video[0].transpose(0,1) #fn,c,h,w
    for idx in tqdm(range(fn), desc=f"Processing faceswap"):
        frame=torchvision.transforms.functional.to_pil_image(video_[idx])
        res_arry=fs_inter.faceswap_process(fs_handle,ref_image_pil,frame)
        video_[idx]=torch.tensor(res_arry/255.0).permute(2,0,1)
    video[0]=video_.transpose(0,1)
    return video

if __name__ == "__main__":
    args=parse_args()
    pose_detector = DWposeDetector()
    pose_detector = pose_detector.to(f"cuda")
    pipe=load_pipeline(args.config)
    fs_handle=fs_inter.faceswap_init()
    lmk_handle=lmk_inter.init_lmk()
    ## whole or half
    align_type="whole"
    #img_list=["image1.jpg","image2.jpg","image3.jpg","image4.jpg","image5.jpg","image6.jpg"]
    img_list=["image1.jpg"]
    input_dir="./configs/inference/ref_images/self"
    video_dir="./configs/inference/km3"
    video_path=os.path.join(video_dir,"src70_250.mp4")
    for img_name in img_list:
        ref_image_path=os.path.join(input_dir,img_name)
        dwpose_save_path=os.path.join(video_dir,img_name.replace(".jpg",".mp4"))
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        #1.视频和图片对齐及dwpose提取
        if align_type=="half":
            crop_ratio=[ 1.5, -3.5, 0.5 ]
            stan_w=512
            stan_h=768
            ref_rect, _=lmk_inter.get_faceRect(lmk_handle,ref_image_pil)
            ref_image_pil,_,_,_,_=crop_standard_img(ref_image_pil,ref_rect,crop_ratio,stan_w,stan_h)
        if os.path.exists(dwpose_save_path):
            pose_map_list = read_frames(dwpose_save_path)
            src_fps = get_fps(dwpose_save_path)
            print(f"pose video{dwpose_save_path} is exist and has {len(pose_map_list)} frames, with {src_fps} fps")
        else:
            video_list = read_frames(video_path)[70:250]
            src_fps = get_fps(video_path)
            if align_type=="whole":
                pose_map_list=process_align_video(pose_detector,video_list,ref_image_path)
            elif align_type=="half":
                video_rect, _=lmk_inter.get_faceRect(lmk_handle,video_list[0])
                _,pose_map_list=crop_video_dwpose(pose_detector,video_list,video_rect,crop_ratio,stan_w,stan_h)
            save_videos_from_pil(pose_map_list, dwpose_save_path, fps=src_fps)

        #2.视频生成
        video=pose2vidpipeline(args,pipe,ref_image_pil,pose_map_list)
        #3.人脸融合
        fs_inter.faceswap_set_src(fs_handle,ref_image_pil)
        video=faceswap_video(fs_handle,video)

        ##save
        fps=(src_fps if args.fps is None else args.fps)
        save_result_video(video,fps,img_name,video_dir.split('/')[-1])
    
    fs_inter.release_handle(fs_handle)
    lmk_inter.release_lmk(lmk_handle)

    