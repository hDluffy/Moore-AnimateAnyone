import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
import math
import pdb


def move_faces(pose_dst,pose_src):
    faces_src = pose_src["faces"]
    faces_dst = pose_dst["faces"]

    bodies_src=pose_src["bodies"]
    bodies_dst=pose_dst["bodies"]
    candidate_src=bodies_src["candidate"]
    candidate_dst=bodies_dst["candidate"]

    if 0:
        ##仿射变换到模板图上
        # 三个原始点
        pts1 = np.float32([candidate_dst[1], candidate_dst[14], candidate_dst[15]])
        # 对应的变换后的点
        pts2 = np.float32([candidate_src[1], candidate_src[14], candidate_src[15]])
        M = cv2.getAffineTransform(pts1, pts2)

        dst_pts = np.array(candidate_dst)
        transformed_pts = cv2.transform(dst_pts.reshape(-1, 1, 2), M)
        # 将结果转换回正常的形状
        transformed_pts = transformed_pts.reshape(-1, 2)
        bodies_src["candidate"][1-1]=transformed_pts[1-1]
        bodies_src["candidate"][15-1]=transformed_pts[15-1]
        bodies_src["candidate"][16-1]=transformed_pts[16-1]
        bodies_src["candidate"][17-1]=transformed_pts[17-1]
        bodies_src["candidate"][18-1]=transformed_pts[18-1]

        dst_pts = np.array(faces_dst)
        transformed_pts = cv2.transform(dst_pts.reshape(-1, 1, 2), M)
        # 将结果转换回正常的形状
        transformed_pts = transformed_pts.reshape(-1, 2)
        for idx in range(68):
            faces_src[0][idx][1]=transformed_pts[idx][1]
            faces_src[0][idx][0]=transformed_pts[idx][0]

    else:
        x_=candidate_src[2-1][1]-candidate_dst[2-1][1]
        y_=candidate_src[2-1][0]-candidate_dst[2-1][0]

        bodies_src["candidate"][1-1]=candidate_dst[1-1]+[y_,x_]
        bodies_src["candidate"][15-1]=candidate_dst[15-1]+[y_,x_]
        bodies_src["candidate"][16-1]=candidate_dst[16-1]+[y_,x_]
        bodies_src["candidate"][17-1]=candidate_dst[17-1]+[y_,x_]
        bodies_src["candidate"][18-1]=candidate_dst[18-1]+[y_,x_]
        
        for idx in range(68):
            faces_src[0][idx][1]=faces_dst[0][idx][1]+x_
            faces_src[0][idx][0]=faces_dst[0][idx][0]+y_
            
    return dict(bodies=bodies_src, hands=pose_src["hands"], faces=faces_src)

def dwpose_align_dst(pose_dst,dst_size,pose_src,src_size):
    candidate_src=pose_src["bodies"]["candidate"]
    candidate_dst=pose_dst["bodies"]["candidate"]
    
    l_src=math.sqrt(math.pow((candidate_src[15-1][0]-candidate_src[11-1][0])*src_size[0],2)+math.pow((candidate_src[15-1][1]-candidate_src[11-1][1])*src_size[1],2))
    l_dst=math.sqrt(math.pow((candidate_dst[15-1][0]-candidate_dst[11-1][0])*dst_size[0],2)+math.pow((candidate_dst[15-1][1]-candidate_dst[11-1][1])*dst_size[1],2))

    resize_ratio=l_dst/l_src
    reize_w=src_size[0]*resize_ratio
    reize_h=src_size[1]*resize_ratio

    off_dst_2=candidate_dst[2-1][0]*dst_size[0]
    left=candidate_src[2-1][0]*reize_w-off_dst_2
    right=left + dst_size[0]
  
    off_dst_11_14=dst_size[1]-(max(candidate_dst[11-1][1],candidate_dst[14-1][1]))*dst_size[1]
    lower=off_dst_11_14+(max(candidate_src[11-1][1],candidate_src[14-1][1]))*reize_h    
    upper=lower - dst_size[1]

    print(left,right,lower,upper)
    return reize_w,reize_h,left,right,lower,upper

def process_align_video(pose_estimation, video_list, dst_img):
    dst_img = Image.open(dst_img).convert("RGB")
    _,pose_dst,_=pose_estimation(dst_img)
    W,H=dst_img.size
    dst_size=[W,H]

    first_frame = video_list[0]
    _,pose_src,_=pose_estimation(first_frame)
    W,H=first_frame.size
    src_size=[W,H]
    
    reize_w,reize_h,left,right,lower,upper=dwpose_align_dst(pose_dst,dst_size,pose_src,src_size)
    
    #frames_crops=[]
    pose_map_list=[]
    for idx in tqdm(range(len(video_list)), desc=f"Processing align video"):
        frame = video_list[idx]
        frame_resize=frame.resize((int(reize_w),int(reize_h)))
        frame_crop=frame_resize.crop((int(left),int(upper),int(right),int(lower)))
        # if idx==0:
        #     frame_resize.save("./frame_resize_ts.jpg")
        #     frame_crop.save("./frame_crop_ts.jpg")
        detected_map,_,_=pose_estimation(frame_crop)
        # pose=move_faces(pose_dst,pose)
        
        pose_map_list.append(detected_map)
        #frames_crops.append(frame_crop)

    return pose_map_list

    