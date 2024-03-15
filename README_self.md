## data
- 训练视频数据处理，进行中心crop和提取dwpose，详见tools/prepare_dataset，数据形式在output_path下会生成train和train_dwpose文件夹，当提取dwpose后会生成video_meta.json
    ```bash
    # --do_crop 对原视频开启中心crop，--do_dwpose 提取对应的dwpose，并生成video_meta.json用于训练索引
    python -m tools.prepare_dataset --input_path /home/date1/hjq/DataSets/video_dataset/Self_dataset/full_body --output_path /home/date1/hjq/DataSets/video_dataset/Self_dataset --dst_w 512 --dst_h 768 --do_crop --do_dwpose
    ```
- 推理时模板视频前处理，详见tools/video_dwpose_align_image.py</br>
    思路：先分别提取视频中关键帧和输入图的人物dwpose，通过两者身体线段的长度比来作为视频每帧的缩放比，通过输入图中人物dwpose中身体中心点和脚踝点到边界的长度作为视频每帧缩放后的crop偏移；再逐帧进行resize和crop后，进行dwpose提取，并将用户faces点位及body头部几个点位做仿射变换对齐到模板点位，并移动替换视频帧dwpose的相关点位，从而保证人脸区域不畸变。
    ```bash
    python -m tools.video_dwpose_align_image 
    ```
  
## Training
源码中的问题:1.dtype未对应；2.ReferenceAttentionControl中的do_classifier_free_guidance被重置
- 配置accelerator启动deepspeed,在对应目录下会生成default_config.yaml
```bash
accelerate config

#deepspeed按如下参数设置，注offload_optimizer_device和offload_param_device要设为none，选cpu可能会报错
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
```

- training stage1
```bash
accelerate launch --config_file /path/default_config.yaml train_stage_1.py
```

- training stage2
```bash
accelerate launch --config_file /path/default_config.yaml train_stage_2.py
```

- test stage2
调整./configs/prompts/animation_ts.yaml中的模型路径，执行下面指令
```bash
python -m scripts.pose2vid --config ./configs/prompts/animation_ts.yaml -W 512 -H 784 -L 141
```


## 一些有效优化简记

### 模板视频dwpose对齐到用户输入图
- 以身高比作为缩放比例
- 以用户喉部点位作为左右crop的中心，以用户最低脚部点位作为高低偏移量
- 将用户faces点位及body头部几个点位做仿射变换对齐到模板点位，并移动替换视频帧dwpose的相关点位

### 训练trick
- stage2保持与stage1相同尺寸，能更好抑制生成视频的抖动，整体更稳定