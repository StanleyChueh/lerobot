<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="https://cdn-uploads.huggingface.co/production/uploads/631ce4b244503b72277fc89f/MNkMdnJqyPvOAEg20Mafg.png" width="100%">
  <br/>
  <br/>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)

<!-- [![Coverage](https://codecov.io/gh/huggingface/lerobot/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/huggingface/lerobot) -->

</div>

<br/>

<h2 align="center">
    <p><a href="https://huggingface.co/docs/lerobot/so101">
        Test your robot with LeRobot!</a></p>
</h2>

## Model testing

Activate conda env
```bash
cd ~/CSL/lerobot_nn/
conda activate lerobot
```

### Koch Robot:

#### TODO:

In this dev branch, we use the latest LeRobot code to test performance

Code version:commit- **6d0d65a** -2025-12-28

1. Test GR00T N1.5 ⏹️

2. Test PI0.5 ⏹️

3. Test XVLA ⏹️

#### GR00T N1.5

##### Record

```bash
lerobot-record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm     --display_data=true     --dataset.repo_id=ethanCSL/multi_block_picking_new_lerobot_gr00t     --dataset.num_episodes=25          --dataset.episode_time_s=30     --dataset.reset_time_s=5     --dataset.single_task="pick up the green block and put into the box"
```

> **Note**
> You don't have to specify policy type in recording, this is just for separating different type of datasets

##### Train

```bash
 accelerate launch   $(which lerobot-train)   --output_dir=outputs/train_groot   --save_checkpoint=true   --batch_size=16   --steps=20000   --save_freq=20000   --log_freq=200   --policy.type=groot   --policy.repo_id=multi_block_picking_new_lerobot_gr00t   --policy.tune_diffusion_model=false   --dataset.repo_id=ethanCSL/multi_block_picking_new_lerobot_gr00t   --dataset.video_backend=pyav   --wandb.enable=false   --wandb.disable_artifact=true   --job_name=groot
```
> **Note**
> RTX5090 LeRobot installation solution
> 
> https://docs.google.com/document/d/1a7i0UfWbSUTbJk_9MFXW-8Dd742hih3A2z61CJjXPG4/edit?usp=sharing
>
> In this command, it needs at least 20GB GPU VRAM to start it, it still needs ~12GB of VRAM to run if lower batch size to 2 or 4


#### Task1: GR00T multi-block pick and place task

<img width="979" height="553" alt="image" src="https://github.com/user-attachments/assets/504f5afa-6bcf-4795-b2fc-a5dd405f6d83" />

```bash
lerobot-record   --robot.type=koch_follower  --robot.port=/dev/ttyUSB_follower    --robot.id=my_awesome_follower_arm  --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm   --robot.cameras='{ 
    front: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30},
  }'   --display_data=true   --dataset.repo_id=ethanCSL/eval_multi_block_picking_new_lerobot_gr00t   --dataset.num_episodes=10   --dataset.single_task="pick up the green block and put into the box" --policy.path=/home/bruce/CSL/lerobot_nn/outputs/train/multi_block_picking_new_lerobot_gr00t/020000/pretrained_model
```

> **Note**
> In this testing, the GPU usage for evaluation is still high, 7.4GB of VRAM in this case
>
> Dataset visualization:
> 
> https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FethanCSL%2Fmulti_block_picking_new_lerobot_gr00t%2Fepisode_0
