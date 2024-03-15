import os
from tqdm import tqdm
import argparse
import json  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, default="")
    parser.add_argument("--video_dir_list",  nargs='+', help='List of items')
    parser.add_argument("--dwpose_dir_list",  nargs='+', help='List of items')
    parser.add_argument("--json_name",  type=str, default="output.json")
    args = parser.parse_args()
    
    meta_list=[]
    for index, video_dir in enumerate(args.video_dir_list):
        src_path = os.path.join(args.input_path, args.video_dir_list[index])
        dwpose_path = os.path.join(args.input_path, args.dwpose_dir_list[index])

        for video_name in tqdm(os.listdir(src_path), desc=f"Processing"):
            video_file=os.path.join(src_path, video_name)
            dwpose_file=os.path.join(dwpose_path, video_name)

            meta_list.append({"video_path":video_file, "kps_path":dwpose_file})

    file_path = args.json_name
    with open(file_path, 'w') as f:
        json.dump(meta_list, f,indent=4)
            

    