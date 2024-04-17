## data
- 训练视频数据处理，通过人脸rect来获取crop框进行crop
  ```bash
  ## 详见脚本实现，有两个接口：1.通过json对video和dwpose进行crop,2.对raw_video进行crop并提取dwpose
  python -m self_tools.prepare_dataset_crop [option]
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

## 测试
- 调整./self_tools/animation.yaml中的模型路径，执行下面指令
  ```bash
  ##修改脚本中的测试数据路径及crop参数
  python -m self_tools.test_run -W 512 -H 768 -L 200
  ```

## 一些有效优化简记

### 模板视频dwpose对齐到用户输入图
#### 方法1
- 以身高比作为缩放比例
- 以用户喉部点位作为左右crop的中心，以用户最低脚部点位作为高低偏移量
- 将用户faces点位及body头部几个点位做仿射变换对齐到模板点位，并移动替换视频帧dwpose的相关点位
#### 方法2
- 将body的关键点分为上半身和下半身分别与用户图进行仿射变换对齐
- 对人脸及人头区域关键点进行相应比例偏移

### 训练trick
- stage2保持与stage1相同尺寸，能更好抑制生成视频的抖动，整体更稳定;
- 加上脚部关键点，能更好控制脚的生成，保证着地感;【注：加入脚部关键点后，模板视频需要确保脚部完整可见】