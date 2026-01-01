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

##### Train

```bash
 accelerate launch   $(which lerobot-train)   --output_dir=outputs/train_groot   --save_checkpoint=true   --batch_size=16   --steps=20000   --save_freq=20000   --log_freq=200   --policy.type=groot   --policy.repo_id=multi_block_picking_new_lerobot_gr00t   --policy.tune_diffusion_model=false   --dataset.repo_id=ethanCSL/multi_block_picking_new_lerobot_gr00t   --dataset.video_backend=pyav   --wandb.enable=false   --wandb.disable_artifact=true   --job_name=groot
```

> **Note**
> You don't have to specify policy type in recording, this is just for separating different type of datasets

#### Task1: SmolVLA multi-blocks picking

> **Dataset**Dataset: https://huggingface.co/ethanCSL/svla_color_test_green
> 
> Dataset visualization:https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FethanCSL%2Fcolor_test_green%2Fepisode_0
> <img width="975" height="387" alt="image" src="https://github.com/user-attachments/assets/2d58a1f2-bb79-4ce8-8170-b111a4cf0f9c" />
> <img width="970" height="396" alt="image" src="https://github.com/user-attachments/assets/12e391cd-cb61-41c9-bc69-8839ad1916a0" />

Pick green block:
```bash
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"   --dataset.single_task="grip green block and put into box"   --dataset.repo_id=ethanCSL/eval_svla_multi_blocks_picking   --dataset.episode_time_s=5000   --dataset.num_episodes=10   --policy.path=/home/bruce/CSL/lerobot_nn/model_test/koch/svla_multi_blocks_picking/checkpoints/020000/pretrained_model
```

Pick white block:
```bash
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"   --dataset.single_task="grip white block and put into box"   --dataset.repo_id=ethanCSL/eval_svla_multi_blocks_picking   --dataset.episode_time_s=5000   --dataset.num_episodes=10   --policy.path=/home/bruce/CSL/lerobot_nn/model_test/koch/svla_multi_blocks_picking/checkpoints/020000/pretrained_model
```
#### Task2: Complex environment picking
<img width="973" height="383" alt="image" src="https://github.com/user-attachments/assets/ace726d2-6c41-4196-b575-e2ee1955a49c" />

> **Dataset**Dataset: [https://huggingface.co/ethanCSL/color_complex](https://huggingface.co/ethanCSL/svla_color_complex)
> 
> Dataset visualization:https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FethanCSL%2Fcolor_complex%2Fepisode_0

Pick white block
```bash
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"   --dataset.single_task="grip white block and put into box"   --dataset.repo_id=ethanCSL/eval_svla_multi_blocks_picking   --dataset.episode_time_s=5000   --dataset.num_episodes=10   --policy.path=/home/bruce/CSL/lerobot_nn/model_test/koch/svla_color_complex/checkpoints/020000/pretrained_model
```

<h2 align="center">
    <p><a href="https://huggingface.co/docs/lerobot/so101">
        Record -> Train -> Inference!</a></p>
</h2>

### Koch

ACT and SmolVLA shares the same recording command,but SmolVLA needs to modify --dataset.single_task for prompt

#### Activate conda env
```bash
conda activate lerobot_nn
```

#### Record episode
##### ACT

```bash
python -m lerobot.record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm     --display_data=true     --dataset.repo_id=ethanCSL/test     --dataset.num_episodes=25          --dataset.episode_time_s=10     --dataset.reset_time_s=5     --dataset.single_task="test" 
```

##### SmolVLA
```
python -m lerobot.record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm     --display_data=true     --dataset.repo_id=ethanCSL/test     --dataset.num_episodes=25          --dataset.episode_time_s=10     --dataset.reset_time_s=5     --dataset.single_task="grip the green block and put into the box" 
```

Resume

Add this argument in the back of the command to resume recording
```
--resume=True
```
> **NOTE:**
> In SmolVLA multi-task setup, you have to record one task first(e.g. "grip the green block and put into the box", and then resume recording to record another task(e.g. "grip the white block and put into the box"
> You can see dataset in ~/.cache/huggingface/lerobot/ethanCSL
>
>  Check video index before recording, make sure top and front camera is correct
#### Train
##### ACT

```
python -m lerobot.scripts.train --policy.type=act --dataset.repo_id=user_name/repo_name --output_dir=outputs/train/your_task_name
```

##### SmolVLA
```
python train.py   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/smolvla_multiblock   --batch_size=16   --steps=20000   --output_dir=outputs/train/svla_multiblock   --job_name=my_smolvla_training   --policy.device=cuda   --wandb.enable=false --policy.repo_id=svla_multiblock
```
