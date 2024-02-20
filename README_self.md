## data
- 视频数据处理，进行中心crop和提取dwpose，详见tools/prepare_dataset，数据形式在output_path下会生成train和train_dwpose文件夹，当提取dwpose后会生成video_meta.json
    ```bash
    # --do_crop 对原视频开启中心crop，--do_dwpose 提取对应的dwpose，并生成video_meta.json用于训练索引
    python -m tools.prepare_dataset --input_path /home/date1/hjq/DataSets/video_dataset/Self_dataset/full_body --output_path /home/date1/hjq/DataSets/video_dataset/Self_dataset --dst_w 512 --dst_h 768 --do_crop --do_dwpose
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