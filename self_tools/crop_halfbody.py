import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm



def crop_standard_img(raw_img:Image, face_rect, crop_ratio = [ 1.5, -3.5, 0.5 ],stan_w=512,stan_h=768):
    top_ratio=crop_ratio[0]
    bottom_ratio=crop_ratio[1]
    lelf_ratio=crop_ratio[2]

    top = face_rect[1] - (face_rect[3] - face_rect[1])*top_ratio
    bottom = face_rect[3] - (face_rect[3] - face_rect[1])*bottom_ratio
    crop_h = (bottom - top)
    crop_w = (float(stan_w)/stan_h) *crop_h
    loffset = (crop_w - (face_rect[2] - face_rect[0])) *lelf_ratio
    roffset = (crop_w - (face_rect[2] - face_rect[0])) *(1-lelf_ratio)
    left = face_rect[0] - loffset
    right = face_rect[2] + roffset

    left=int(left)
    top=int(top)
    right=int(right)
    bottom=int(bottom)

    raw_img_crop = raw_img.crop((left, top, right, bottom))
    res_image=raw_img_crop.resize((stan_w,stan_h))

    return res_image, left, top, right, bottom

def crop_video_dwpose(pose_detector,video_list, face_rect, crop_ratio = [ 1.5, -3.5, 0.5 ],stan_w=512,stan_h=768):

    res_image, left, top, right, bottom = crop_standard_img(video_list[0],face_rect,crop_ratio,stan_w,stan_h)

    crop_frames_list=[]
    pose_map_list=[]
    for idx in tqdm(range(len(video_list)), desc=f"Processing crop video"):
        frame = video_list[idx]
        frame_crop = frame.crop((left, top, right, bottom))
        frame_resize=frame_crop.resize((stan_w,stan_h))
        
        if pose_detector!=None:
            detected_map,pose,score=pose_detector(frame_resize)
            pose_map_list.append(detected_map)
        crop_frames_list.append(frame_resize)

    return crop_frames_list, pose_map_list


if __name__ == "__main__":
    from self_tools.face_lmk import interface as lmk_inter
    ref_image_path="./configs/inference/ref_images/self/image1.jpg"

    crop_type="half"
    if crop_type=="half":
        crop_ratio=[ 1.5, -3.5, 0.5 ]
        stan_w=512
        stan_h=768
        save_path=ref_image_path[:-4]+"_half.jpg"
    elif crop_type=="whole":
        crop_ratio=[ 2, -9.8, 0.5 ]
        stan_w=576
        stan_h=1024
        save_path=ref_image_path[:-4]+"_whole.jpg"

    lmk_handle=lmk_inter.init_lmk()
    
    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    ref_rect, _=lmk_inter.get_faceRect(lmk_handle,ref_image_pil)
    ref_image_pil,_,_,_,_=crop_standard_img(ref_image_pil,ref_rect,crop_ratio,stan_w,stan_h)
    ref_image_pil.save(save_path)