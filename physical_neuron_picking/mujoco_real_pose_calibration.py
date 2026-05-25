# Run gripper rest calibration
'''
python mujoco_leader_guided_calibration.py \
  --xml follower.xml \
  --pose_name rest \
  --follower_port /dev/ttyUSB_follower \
  --follower_id my_awesome_follower_arm \
  --leader_port /dev/ttyUSB_leader \
  --leader_id my_awesome_leader_arm
'''

# Run gripper reach calibration
'''
python mujoco_leader_guided_calibration.py \
  --xml follower.xml \
  --pose_name reach \
  --follower_port /dev/ttyUSB_follower \
  --follower_id my_awesome_follower_arm \
  --leader_port /dev/ttyUSB_leader \
  --leader_id my_awesome_leader_arm
'''

# Run gripper raise calibration
'''
python mujoco_leader_guided_calibration.py \
  --xml follower.xml \
  --pose_name raise \
  --follower_port /dev/ttyUSB_follower \
  --follower_id my_awesome_follower_arm \
  --leader_port /dev/ttyUSB_leader \
  --leader_id my_awesome_leader_arm
'''

# Run gripper opened calibration
'''
python mujoco_leader_guided_calibration.py \
  --xml follower.xml \
  --pose_name open \
  --follower_port /dev/ttyUSB_follower \
  --follower_id my_awesome_follower_arm \
  --leader_port /dev/ttyUSB_leader \
  --leader_id my_awesome_leader_arm
'''
# Run gripper closed calibration
'''
python mujoco_leader_guided_calibration.py \
  --xml follower.xml \
  --pose_name closed \
  --follower_port /dev/ttyUSB_follower \
  --follower_id my_awesome_follower_arm \
  --leader_port /dev/ttyUSB_leader \
  --leader_id my_awesome_leader_arm
'''
import argparse
import json
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.processor import make_default_processors


JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

OBS_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def extract_state_vector(obs):
    if "observation.state" in obs:
        v = to_numpy(obs["observation.state"])
        if v is not None and v.size >= 6:
            return v[:6].copy(), "observation.state"

    if "state" in obs:
        v = to_numpy(obs["state"])
        if v is not None and v.size >= 6:
            return v[:6].copy(), "state"

    if all(k in obs for k in OBS_JOINT_KEYS):
        values = []
        for k in OBS_JOINT_KEYS:
            value = obs[k]
            if hasattr(value, "item"):
                value = value.item()
            values.append(float(value))
        return np.asarray(values, dtype=np.float64), "direct_joint_keys"

    print("Available observation keys:")
    for k in obs.keys():
        print("  ", k)
    raise KeyError("Could not extract follower state vector.")


def find_body(root, body_name):
    for elem in root.iter("body"):
        if elem.attrib.get("name") == body_name:
            return elem
    return None


def site_exists(body_elem, site_name):
    for child in body_elem:
        if child.tag == "site" and child.attrib.get("name") == site_name:
            return True
    return False


def add_debug_site(body_elem, name, pos, rgba, size="0.035"):
    if site_exists(body_elem, name):
        return
    ET.SubElement(
        body_elem,
        "site",
        {
            "name": name,
            "type": "sphere",
            "pos": pos,
            "size": size,
            "rgba": rgba,
        },
    )


def make_debug_xml(xml_path, offwidth=640, offheight=480):
    xml_path = Path(xml_path).resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    global_elem = visual.find("global")
    if global_elem is None:
        global_elem = ET.SubElement(visual, "global")

    global_elem.set("offwidth", str(offwidth))
    global_elem.set("offheight", str(offheight))

    link5 = find_body(root, "link_5")
    link6 = find_body(root, "link_6")

    if link5 is None:
        raise ValueError("Cannot find body link_5 in XML.")
    if link6 is None:
        raise ValueError("Cannot find body link_6 in XML.")

    add_debug_site(
        link5,
        "DEBUG_RED_current_FK_EEF_used",
        "-0.06429 0.00327 0.0011",
        "1 0 0 1",
        size="0.040",
    )

    add_debug_site(
        link6,
        "DEBUG_GREEN_link6_origin",
        "0 0 0",
        "0 1 0 1",
        size="0.030",
    )

    add_debug_site(
        link6,
        "DEBUG_YELLOW_link6_tip_guess",
        "-0.080 0 0",
        "1 1 0 1",
        size="0.035",
    )

    debug_xml_path = xml_path.with_name(xml_path.stem + "_LEADER_CALIB_DEBUG.xml")
    tree.write(debug_xml_path, encoding="utf-8", xml_declaration=True)
    return debug_xml_path


def set_mujoco_qpos_from_state(model, data, state_vec, unit="deg", sign=None, offset=None):
    q = np.asarray(state_vec, dtype=np.float64).reshape(-1)[:6].copy()

    if sign is None:
        sign = np.ones(6, dtype=np.float64)
    else:
        sign = np.asarray(sign, dtype=np.float64)

    if offset is None:
        offset = np.zeros(6, dtype=np.float64)
    else:
        offset = np.asarray(offset, dtype=np.float64)

    q = sign * q + offset

    if unit == "deg":
        q = np.deg2rad(q)
    elif unit == "rad":
        pass
    else:
        raise ValueError(unit)

    mujoco.mj_resetData(model, data)

    for i, joint_name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in XML: {joint_name}")

        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = float(q[i])

    mujoco.mj_forward(model, data)
    return q


def get_site_z(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return np.nan
    return float(data.site_xpos[site_id][2])


def make_camera():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.65
    cam.azimuth = 135
    cam.elevation = -25
    cam.lookat[:] = np.array([0.0, 0.0, 0.16])
    return cam


def draw_overlay(img, lines):
    y = 28
    for line in lines:
        cv2.putText(
            img,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24


def render_pose(xml_path, state_vec, pose_name, out_dir, unit="deg"):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    width = 640
    height = 480

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = make_camera()

    q_rad = set_mujoco_qpos_from_state(
        model,
        data,
        state_vec,
        unit=unit,
    )

    red_z = get_site_z(model, data, "DEBUG_RED_current_FK_EEF_used")
    yellow_z = get_site_z(model, data, "DEBUG_YELLOW_link6_tip_guess")
    original_z = get_site_z(model, data, "end_effector_site")

    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    lines = [
        f"pose={pose_name} | unit={unit}",
        "RED=current XML end_effector_site | YELLOW=link6 tip guess | GREEN=link6 origin",
        f"RED z={red_z:.6f} | original_eef z={original_z:.6f} | YELLOW z={yellow_z:.6f}",
        f"raw follower state={np.array2string(np.asarray(state_vec), precision=3)}",
    ]
    draw_overlay(bgr, lines)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_img = out_dir / f"{pose_name}_mujoco_calibration.png"
    cv2.imwrite(str(out_img), bgr)

    out_json = out_dir / f"{pose_name}_state.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "pose_name": pose_name,
                "state_vec": [float(x) for x in state_vec],
                "unit": unit,
                "q_rad": [float(x) for x in q_rad],
                "red_current_eef_z": red_z,
                "yellow_tip_guess_z": yellow_z,
                "original_end_effector_site_z": original_z,
            },
            f,
            indent=2,
        )

    renderer.close()

    print(f"[+] Wrote image: {out_img}")
    print(f"[+] Wrote state: {out_json}")


def teleop_loop(stop_event, follower_robot, leader_teleop, fps):
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    print("[*] Leader-guided teleop loop started.")
    print("[*] Move the leader arm. The follower should follow.")
    print("[*] Press ENTER in the main terminal when the follower reaches the calibration pose.")

    while not stop_event.is_set():
        start_t = time.perf_counter()

        obs = follower_robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        raw_action = leader_teleop.get_action()
        action_from_leader = teleop_action_processor((raw_action, obs))

        robot_action_to_send = robot_action_processor((action_from_leader, obs))
        follower_robot.send_action(robot_action_to_send)

        dt = time.perf_counter() - start_t
        sleep_t = max(0.0, 1.0 / fps - dt)
        time.sleep(sleep_t)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--xml", default="follower.xml")
    argp.add_argument(
        "--pose_name",
        required=True,
        choices=["rest", "reach", "raise", "open", "closed"],
    )
    argp.add_argument("--out_dir", default="real_pose_calibration_outputs")
    argp.add_argument("--unit", choices=["deg", "rad"], default="deg")

    argp.add_argument("--follower_type", default="koch_follower")
    argp.add_argument("--follower_port", required=True)
    argp.add_argument("--follower_id", default="my_awesome_follower_arm")

    argp.add_argument("--leader_type", default="koch_leader")
    argp.add_argument("--leader_port", required=True)
    argp.add_argument("--leader_id", default="my_awesome_leader_arm")

    argp.add_argument("--fps", type=int, default=30)

    args = argp.parse_args()

    debug_xml = make_debug_xml(args.xml)

    # These config imports may need adjustment depending on your LeRobot version.
    # If import fails, copy the exact config classes used by your normal record script.
    from lerobot.robots.koch_follower import KochFollowerConfig
    from lerobot.teleoperators.koch_leader import KochLeaderConfig

    follower_cfg = KochFollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
    )

    leader_cfg = KochLeaderConfig(
        port=args.leader_port,
        id=args.leader_id,
    )

    follower_robot = make_robot_from_config(follower_cfg)
    leader_teleop = make_teleoperator_from_config(leader_cfg)

    stop_event = threading.Event()

    try:
        follower_robot.connect()
        leader_teleop.connect()

        worker = threading.Thread(
            target=teleop_loop,
            args=(stop_event, follower_robot, leader_teleop, args.fps),
            daemon=True,
        )
        worker.start()

        print("\n" + "=" * 90)
        print(f"Leader-guide the follower to pose: {args.pose_name}")
        print("When the follower is physically in the desired pose, press ENTER.")
        print("=" * 90)
        input()

        stop_event.set()
        worker.join(timeout=2.0)

        obs = follower_robot.get_observation()
        state_vec, state_key = extract_state_vector(obs)

        print(f"[+] Captured follower state from key: {state_key}")
        for i, name in enumerate(OBS_JOINT_KEYS):
            print(f"    state[{i}] {name:<18} = {state_vec[i]: .6f}")

    finally:
        stop_event.set()
        if follower_robot.is_connected:
            follower_robot.disconnect()
        if leader_teleop.is_connected:
            leader_teleop.disconnect()

    render_pose(
        xml_path=debug_xml,
        state_vec=state_vec,
        pose_name=args.pose_name,
        out_dir=args.out_dir,
        unit=args.unit,
    )


if __name__ == "__main__":
    main()