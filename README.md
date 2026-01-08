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
conda activate lerobot_nn
```

### Koch Robot:

> **NOTE:**
>
> Model test with reset and teleop arm
> 
> lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}"   --dataset.single_task="grip green block and put into box"   --dataset.repo_id=ethanCSL/eval_svla_multi_blocks_picking   --dataset.episode_time_s=10000   --dataset.num_episodes=10000   --policy.path=/home/bruce/CSL/lerobot_nn/outputs/train/Ting_grip_box_svla/checkpoints/020000/pretrained_model --display_data=True  --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm  --dataset.reset_time_s=5


#### Task1: SmolVLA multi-blocks picking

> **Dataset**Dataset: https://huggingface.co/ethanCSL/svla_color_test_green
> 
> Dataset visualization:https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FethanCSL%2Fcolor_test_green%2Fepisode_0
> <img width="975" height="387" alt="image" src="https://github.com/user-attachments/assets/2d58a1f2-bb79-4ce8-8170-b111a4cf0f9c" />
> <img width="970" height="396" alt="image" src="https://github.com/user-attachments/assets/12e391cd-cb61-41c9-bc69-8839ad1916a0" />

Pick green block:
```bash
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"   --dataset.single_task="grip green block and put into box"   --dataset.repo_id=ethanCSL/eval_svla_multi_blocks_picking   --dataset.episode_time_s=5000   --dataset.num_episodes=10   --policy.path=/home/bruce/CSL/lerobot_nn/model_test/koch/svla_multi_blocks_picking/checkpoints/020000/pretrained_model --display_data=True  --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm  --dataset.reset_time_s=5
```

Pick white block:
```bash
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"   --dataset.single_task="grip white block and put into box"   --dataset.repo_id=ethanCSL/eval_svla_multi_blocks_picking   --dataset.episode_time_s=5000   --dataset.num_episodes=10   --policy.path=/home/bruce/CSL/lerobot_nn/model_test/koch/svla_multi_blocks_picking/checkpoints/020000/ --display_data=True  --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm  --dataset.reset_time_s=5
```

#### Task2: Complex environment picking
<img width="973" height="383" alt="image" src="https://github.com/user-attachments/assets/ace726d2-6c41-4196-b575-e2ee1955a49c" />

> **Dataset**Dataset: [https://huggingface.co/ethanCSL/color_complex](https://huggingface.co/ethanCSL/svla_color_complex)
> 
> Dataset visualization:https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FethanCSL%2Fcolor_complex%2Fepisode_0

Pick white block
```bash
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"   --dataset.single_task="grip white block and put into box"   --dataset.repo_id=ethanCSL/eval_svla_multi_blocks_picking   --dataset.episode_time_s=5000   --dataset.num_episodes=10   --policy.path=/home/bruce/CSL/lerobot_nn/model_test/koch/svla_color_complex/checkpoints/020000/pretrained_model --display_data=True  --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm  --dataset.reset_time_s=5
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

Right key to save episode, left key to discard episode

```bash
python -m lerobot.record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm     --display_data=true     --dataset.repo_id=ethanCSL/test     --dataset.num_episodes=25          --dataset.episode_time_s=10000     --dataset.reset_time_s=5     --dataset.single_task="test" 
```

##### SmolVLA

Right key to save episode, left key to discard episode

```
 python -m lerobot.record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm     --display_data=true     --dataset.repo_id=ethanCSL/test     --dataset.num_episodes=25          --dataset.episode_time_s=100000     --dataset.reset_time_s=5     --dataset.single_task="grip the green block and put into the box" 
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
python -m lerobot.scripts.train  --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/smolvla_multiblock   --batch_size=16   --steps=20000   --output_dir=outputs/train/svla_multiblock   --job_name=my_smolvla_training   --policy.device=cuda   --wandb.enable=false --policy.repo_id=svla_multiblock
```

### Franka emika panda

> **NOTE:**
> Remember to set franka_ros environment if you have not.
> https://github.com/frankarobotics/franka_ros

#### Control PC(Client)

Setting ethernet
```bash
cd franka_ws/
python connect_franka.py
```

#### Record 

Start joint impedance control
```bash
cd franka_ws/
source devel/setup.bash
roslaunch franka_example_controllers joint_impedance_example_controller.launch robot_ip:=172.16.0.2 load_gripper:=true 
```

Run Leading arm control node
```bash
cd gello_franka/
python3 small_arm_to_franka.py
```

> **NOTE:**
> You have to wait for a bit, it needs to initialize first before teleoperation.

Run recording node
```bash
cd franka_record/
python record_small_arm_three_cam_save_RAM.py --repo_id ethanCSL/test --single_task test
```
> **NOTE:**
> right key to save, left key to discard episode.

#### Model testing(Inference):

Launch franka_ros cartesian impedance control
```bash
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=172.16.0.2 load_gripper:=true launch_rviz:=false
```

Set to initial pose
```bash
cd Control_PC/franka_ws/
python franka_ros.py 
```

> **NOTE:**
> If you see Switched to cartesian impedance controller,you can shut this down.
> You will see something like below
> [INFO] [1766474178.275051]: Switched to cartesian impedance controller.


Publish camera topic
```bash
cd franka_record/
python3 image_publisher_SA.py
```

##### Action Chunking Transformer(ACT)
-------------------------------------------------

Run server node(Inference PC)
```bash
cd Inference_PC/lerobot_franka_inference/
python3 franka_socket_test_stability_test.py --ckpt-path /home/bruce/CSL/model_test/act_pick_n_place_100_top_view/pretrained_model
```
> **NOTE:**
> Please make sure the server has been opened successfully before running the client node!

Run client node(Control PC)
```bash
cd franka_record/tools/
python evaluation.py
```

##### SmolVLA(Still testing)
-------------------------------------------------

Run server node(Inference PC)
```bash
cd Inference_PC/lerobot_franka_inference/
python3 franka_socket_test_stability_test_smolvla.py --ckpt-path /home/bruce/CSL/model_test/pick_n_place_100_smolvla_40000/pretrained_model --eval-freq 10 --task "pick up the red cube and place it"
```

Run client node(Control PC)
```bash
cd franka_record/tools/
python evaluation.py
```

## Installation

LeRobot works with Python 3.10+ and PyTorch 2.2+.

### Environment Setup

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):

```bash
conda create -y -n lerobot_nn python=3.10
conda activate lerobot_nn
```

When using `miniconda`, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>
> - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>
> ```bash
> conda install ffmpeg=7.1.1 -c conda-forge
> ```
>
> - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

### Install LeRobot 🤗

#### From Source

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Then, install the library in editable mode. This is useful if you plan to contribute to the code.

```bash
pip install -e .
```

> **NOTE:** If you encounter build errors, you may need to install additional dependencies (`cmake`, `build-essential`, and `ffmpeg libs`). On Linux, run:
> `sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev`. For other systems, see: [Compiling PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

For simulations, 🤗 LeRobot comes with gymnasium environments that can be installed as extras:

- [aloha](https://github.com/huggingface/gym-aloha)
- [xarm](https://github.com/huggingface/gym-xarm)
- [pusht](https://github.com/huggingface/gym-pusht)

For instance, to install 🤗 LeRobot with aloha and pusht, use:

```bash
pip install -e ".[aloha, pusht]"
```

## Citation

If you want, you can cite this work with:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
