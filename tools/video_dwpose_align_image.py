import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from src.dwpose import util
from src.dwpose.wholebody import Wholebody

from src.utils.util import get_fps, read_frames, save_videos_from_pil
from tqdm import tqdm
import math
import pdb

#边界为18时不包含脚，24包含脚
body_edge=18

def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        # [3, 17],
        # [6, 18],
        [11, 24],
        [11, 23],
        [11, 22],
        [14, 21],
        [14, 20],
        [14, 19],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        [255, 128, 0], 
        [255, 0, 128],
        [0, 255, 128],
        [128, 255, 0],
        [0, 128, 255],
        [128, 0, 255],

    ]

    for i in range(body_edge-1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(body_edge):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas

def get_dwpose(pose_estimation,input_image):
    input_image = cv2.cvtColor(
        np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
    )
    input_image = HWC3(input_image)
    H, W, C = input_image.shape
    with torch.no_grad():
        candidate, subset = pose_estimation(input_image)
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        score = subset[:, :body_edge]
        max_ind = np.mean(score, axis=-1).argmax(axis=0)
        score = score[[max_ind]]
        body = candidate[:, :body_edge].copy()
        body = body[[max_ind]]
        nums = 1
        body = body.reshape(nums * body_edge, locs)
        body_score = copy.deepcopy(score)
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(body_edge * i + j)
                else:
                    score[i][j] = -1

        un_visible = subset < 0.3
        candidate[un_visible] = -1

        foot = candidate[:, 18:24]

        faces = candidate[[max_ind], 24:92]

        hands = candidate[[max_ind], 92:113]
        hands = np.vstack([hands, candidate[[max_ind], 113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)
    return pose,H,W


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

def process_video(pose_estimation, video_path, dst_img, output_video_path, start_f=0, end_f=256):
    dst_img=cv2.imread(dst_img)
    pose_dst,H,W=get_dwpose(pose_estimation,dst_img)
    dst_size=[W,H]

    fps = get_fps(video_path)
    print("fps:",fps)
    vr = read_frames(video_path)
    first_frame = vr[start_f]
    src_img=cv2.cvtColor(np.array(first_frame),cv2.COLOR_RGB2BGR)
    pose_src,H,W=get_dwpose(pose_estimation,src_img)
    src_size=[W,H]
    
    candidate_src=pose_src["bodies"]["candidate"]
    candidate_dst=pose_dst["bodies"]["candidate"]
    
    l_src=math.sqrt(math.pow((candidate_src[15-1][0]-candidate_src[11-1][0])*src_size[0],2)+math.pow((candidate_src[15-1][1]-candidate_src[11-1][1])*src_size[1],2))
    l_dst=math.sqrt(math.pow((candidate_dst[15-1][0]-candidate_dst[11-1][0])*dst_size[0],2)+math.pow((candidate_dst[15-1][1]-candidate_dst[11-1][1])*dst_size[1],2))

    resize_ratio=l_dst/l_src
    reize_w=src_size[0]*resize_ratio
    reize_h=src_size[1]*resize_ratio

    # left=(reize_w - dst_size[0])/2.0
    # right=left + dst_size[0]
    # c_src_9_12=(candidate_src[9-1]+candidate_src[12-1])/2.0
    # c_dst_9_12=(candidate_dst[9-1]+candidate_dst[12-1])/2.0
    # off_dst_9_12=c_dst_9_12[0]*dst_size[0]
    off_dst_2=candidate_dst[2-1][0]*dst_size[0]
    left=candidate_src[2-1][0]*reize_w-off_dst_2
    right=left + dst_size[0]
  
    off_dst_11_14=dst_size[1]-(max(candidate_dst[11-1][1],candidate_dst[14-1][1]))*dst_size[1]
    lower=off_dst_11_14+(max(candidate_src[11-1][1],candidate_src[14-1][1]))*reize_h    
    upper=lower - dst_size[1]


    print(left,right,lower,upper)
    
    frames_crops=[]
    pose_map_list=[]
    for idx in tqdm(range(len(vr)), desc=f"Processing {os.path.basename(video_path)}"):
        frame = vr[idx]
        frame_resize=frame.resize((int(reize_w),int(reize_h)))
        frame_crop=frame_resize.crop((int(left),int(upper),int(right),int(lower)))
        if idx==0:
            frame_resize.save("./frame_resize_ts.jpg")
            frame_crop.save("./frame_crop_ts.jpg")
        if idx < start_f:
            continue
        if idx >= end_f:
            break
        img_crop=cv2.cvtColor(np.array(frame_crop),cv2.COLOR_RGB2BGR)
        pose,H,W=get_dwpose(pose_estimation,img_crop)
        # pose=move_faces(pose_dst,pose)
        detected_map = draw_pose(pose, H, W)
        if idx==start_f:
            merge_img=dst_img*0.5+detected_map*0.5
            cv2.imwrite("./merge_img.jpg",merge_img)
        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR
        )
        detected_map = Image.fromarray(detected_map)
        
        pose_map_list.append(detected_map)
        frames_crops.append(frame_crop)

    save_videos_from_pil(pose_map_list, output_video_path, fps=fps)  

def image_test(pose_estimation):
    input_image1="./configs/inference/ref_images/self/image1.jpg"
    input_image1=cv2.imread(input_image1)
    pose1,H,W=get_dwpose(pose_estimation,input_image1)

    input_tmp="./configs/inference/ref_images/self/image2.jpg"
    #input_tmp="./configs/inference/km3/tmp01.mp4"
    if input_tmp.endswith(".mp4"):
        fps = get_fps(input_tmp)
        print("fps:",fps)
        vr = read_frames(input_tmp)
        first_frame = vr[70]
        input_image2=cv2.cvtColor(np.array(first_frame),cv2.COLOR_RGB2BGR)
    else:
        input_image2=cv2.imread(input_tmp)
    pose2,H,W=get_dwpose(pose_estimation,input_image2)
    pose2=move_faces(pose1,pose2)
    detected_map = draw_pose(pose2, H, W)
    detected_map = HWC3(detected_map)
    detected_map = cv2.resize(
        detected_map, (W, H), interpolation=cv2.INTER_LINEAR
    )

    detected_map = Image.fromarray(detected_map)
    detected_map.save("./dwpose_ts.jpg")

if __name__ == "__main__":
    pose_estimation = Wholebody("cpu")
    # image_test(pose_estimation)

    img_list=["image1.jpg","image2.jpg","image3.jpg","image4.jpg","image5.jpg","image6.jpg"]
    #img_list=["image5.jpg"]
    input_dir="./configs/inference/ref_images/self"
    video_dir="./configs/inference/km3"
    video_path=os.path.join(video_dir,"科目三.mp4")
    for img_name in img_list:
        input_image=os.path.join(input_dir,img_name)
        save_path=os.path.join(video_dir,img_name.replace(".jpg",".mp4"))
        process_video(pose_estimation,video_path,input_image,save_path,70,250)

    