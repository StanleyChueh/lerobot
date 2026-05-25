'''
python lerobot_replay_in_mujoco.py   --repo_id ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2   --xml follower.xml   --rest_json real_pose_calibration_outputs/rest_state.json   --episode 21   --stride 5   --width 960   --height 720

'''
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Calibration data
CALIB_REST_STATE_DEG = np.asarray([
    -11.208791208791212,
    97.00520833333331,
    17.25596857056513,
    100.0,
    -9.010989010989007,
    47.38863287250384,
], dtype=np.float64)

JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]


CALIB_TARGET_REST_DEG = np.asarray([
    91.7,
    15.0,
    40.0,
    65.0,
    0.0,
    -30.0,
], dtype=np.float64)

CALIB_ORDER = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
CALIB_SCALE = np.asarray([1.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
CALIB_SIGN = np.asarray([1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float64)

GRIPPER_A = -1.424134513274
GRIPPER_B = 8.690973451327


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def load_rest_state(rest_json):
    with open(rest_json, "r") as f:
        data = json.load(f)
    return np.asarray(data["state_vec"], dtype=np.float64)


def get_episode_indices(dataset):
    for obj in [dataset, dataset.meta]:
        if hasattr(obj, "episode_data_index"):
            return getattr(obj, "episode_data_index")
    if hasattr(dataset.meta, "episodes"):
        return dataset.meta.episodes
    return None


def _to_int(x):
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def get_episode_frame_range(ep_index, ep_idx):
    if hasattr(ep_index, "column_names"):
        columns = set(ep_index.column_names)

        if "dataset_from_index" in columns and "dataset_to_index" in columns:
            row = ep_index[ep_idx]
            return _to_int(row["dataset_from_index"]), _to_int(row["dataset_to_index"])

        if "from" in columns and "to" in columns:
            row = ep_index[ep_idx]
            return _to_int(row["from"]), _to_int(row["to"])

        if "length" in columns:
            start_frame = 0
            for i in range(ep_idx):
                start_frame += _to_int(ep_index[i]["length"])
            end_frame = start_frame + _to_int(ep_index[ep_idx]["length"])
            return start_frame, end_frame

    if isinstance(ep_index, dict) and "from" in ep_index and "to" in ep_index:
        return _to_int(ep_index["from"][ep_idx]), _to_int(ep_index["to"][ep_idx])

    if isinstance(ep_index, list) and isinstance(ep_index[ep_idx], dict):
        ep = ep_index[ep_idx]
        if "from" in ep and "to" in ep:
            return _to_int(ep["from"]), _to_int(ep["to"])
        if "start" in ep and "end" in ep:
            return _to_int(ep["start"]), _to_int(ep["end"])
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            return _to_int(ep["dataset_from_index"]), _to_int(ep["dataset_to_index"])

    raise TypeError("Unrecognized episode index format.")


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


def add_debug_site(body_elem, name, pos, rgba, size):
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


def geom_exists(body_elem, geom_name):
    for child in body_elem:
        if child.tag == "geom" and child.attrib.get("name") == geom_name:
            return True
    return False


def add_debug_geom_sphere(body_elem, name, pos, rgba, size="0.055"):
    """
    Add a visible colored sphere attached to a body.
    This is easier to see than a MuJoCo site in rendered video.
    """
    if geom_exists(body_elem, name):
        return

    ET.SubElement(
        body_elem,
        "geom",
        {
            "name": name,
            "type": "sphere",
            "pos": pos,
            "size": size,
            "rgba": rgba,
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )

def color_body_geoms(body_elem, rgba):
    """
    Color all visible/collision geoms inside a body.
    This helps identify which physical part a body corresponds to.
    """
    for child in body_elem:
        if child.tag != "geom":
            continue

        name = child.attrib.get("name", "")
        if name.startswith("DEBUG_"):
            continue

        child.set("rgba", rgba)
        child.attrib.pop("material", None)

def make_body_visual_geoms_transparent(body_elem, alpha="0.35"):
    """
    Make visual geoms in a body semi-transparent so internal markers can be seen.
    """
    for child in body_elem:
        if child.tag == "geom":
            # Only modify visible mesh geoms; leave debug spheres as-is.
            name = child.attrib.get("name", "")
            if name.startswith("DEBUG_"):
                continue

            if child.attrib.get("class") == "visual" or "mesh" in child.attrib:
                child.set("rgba", f"0.7 0.7 0.7 {alpha}")

def make_debug_xml(xml_path, width=960, height=720):
    xml_path = Path(xml_path).resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    global_elem = visual.find("global")
    if global_elem is None:
        global_elem = ET.SubElement(visual, "global")

    global_elem.set("offwidth", str(width))
    global_elem.set("offheight", str(height))

    link5 = find_body(root, "link_5")
    link6 = find_body(root, "link_6")

    # Tint the parent link of the used EEF site.
    # Current end_effector_site is attached to link_5.
    color_body_geoms(link5, "1 0.2 0.2 0.65")   # reddish transparent link_5

    # Tint the next distal link/gripper part.
    color_body_geoms(link6, "1 1 0 0.65")       # yellow transparent link_6

    make_body_visual_geoms_transparent(link5, alpha="1.0")
    make_body_visual_geoms_transparent(link6, alpha="1.0")

    if link5 is None:
        raise ValueError("Cannot find body link_5.")
    if link6 is None:
        raise ValueError("Cannot find body link_6.")

    # ------------------------------------------------------------------
    # RED = exact point used by EEF height calculation.
    # Same local position as XML end_effector_site on link_5.
    # ------------------------------------------------------------------
    add_debug_site(
        link5,
        "DEBUG_RED_USED_EEF_SITE",
        "-0.06429 0.00327 0.0011",
        "1 0 0 1",
        "0.055",
    )

    add_debug_geom_sphere(
        link5,
        "DEBUG_RED_USED_EEF_GEOM",
        "-0.06429 0.00327 0.0011",
        "1 0 0 1",
        size="0.008",
    )

    # # ------------------------------------------------------------------
    # # GREEN = link_6 origin.
    # # ------------------------------------------------------------------
    # add_debug_site(
    #     link6,
    #     "DEBUG_GREEN_LINK6_ORIGIN",
    #     "0 0 0",
    #     "0 1 0 1",
    #     "0.045",
    # )

    # add_debug_geom_sphere(
    #     link6,
    #     "DEBUG_GREEN_LINK6_ORIGIN_GEOM",
    #     "0 0 0",
    #     "0 1 0 1",
    #     size="0.008",
    # )

    # # ------------------------------------------------------------------
    # # YELLOW = guessed distal gripper / fingertip candidate.
    # # Tune this point if needed.
    # # ------------------------------------------------------------------
    # add_debug_site(
    #     link6,
    #     "DEBUG_YELLOW_TIP_GUESS",
    #     "-0.080 0 0",
    #     "1 1 0 1",
    #     "0.055",
    # )

    # add_debug_geom_sphere(
    #     link6,
    #     "DEBUG_YELLOW_TIP_GUESS_GEOM",
    #     "-0.080 0 0",
    #     "1 1 0 1",
    #     size="0.008",
    # )

    debug_xml_path = xml_path.with_name(xml_path.stem + "_SHOW_USED_EEF.xml")
    tree.write(debug_xml_path, encoding="utf-8", xml_declaration=True)
    return debug_xml_path


def state_to_q_rad_calibrated(state_vec, rest_state_deg):
    state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1)

    if state_vec.size < 6:
        raise ValueError(f"State dimensionality must be >= 6, got {state_vec.shape}")

    raw_deg = state_vec[:6]
    raw_delta = raw_deg[CALIB_ORDER] - rest_state_deg[CALIB_ORDER]

    q_deg = CALIB_TARGET_REST_DEG + CALIB_SIGN * CALIB_SCALE * raw_delta

    # Override gripper joint with separately calibrated gripper mapping.
    # raw_deg[5] is gripper.pos from dataset / robot state.
    q_deg[5] = GRIPPER_A * raw_deg[5] + GRIPPER_B

    # XML joint_6 range is approximately [-140.4, 1.8] degrees.
    q_deg[5] = np.clip(q_deg[5], -140.4, 1.8)

    q_rad = np.deg2rad(q_deg)

    return q_deg, q_rad


def set_qpos_from_state(model, data, state_vec, rest_state_deg):
    q_deg, q_rad = state_to_q_rad_calibrated(state_vec, rest_state_deg)

    mujoco.mj_resetData(model, data)

    for i, joint_name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")

        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = float(q_rad[i])

    mujoco.mj_forward(model, data)

    return q_deg


def get_site_xyz(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site not found: {site_name}")
    return data.site_xpos[site_id].copy()


def get_body_xyz(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    return data.xpos[body_id].copy()


def print_site_parent_info(model, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site not found: {site_name}")

    body_id = int(model.site_bodyid[site_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    local_pos = model.site_pos[site_id].copy()

    print("\n" + "=" * 90)
    print("EEF SITE USED BY HEIGHT CALCULATION")
    print("=" * 90)
    print(f"site name       : {site_name}")
    print(f"parent body     : {body_name}")
    print(f"local site pos  : {local_pos}")
    print("height used     : mj_data.site_xpos[site_id][2]")
    print("visual marker   : RED sphere")
    print("=" * 90)


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
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--xml", default="follower.xml")
    parser.add_argument("--rest_json", required=None)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--out_dir", default="show_used_eef_outputs")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    if args.rest_json is not None:
        rest_state_deg = load_rest_state(args.rest_json)
    else:
        rest_state_deg = CALIB_REST_STATE_DEG

    debug_xml = make_debug_xml(args.xml, width=args.width, height=args.height)
    print(f"[+] Wrote debug XML: {debug_xml}")

    model = mujoco.MjModel.from_xml_path(str(debug_xml))
    data = mujoco.MjData(model)

    print_site_parent_info(model, "end_effector_site")

    dataset = LeRobotDataset(args.repo_id, video_backend="pyav")
    ep_index = get_episode_indices(dataset)
    start_frame, end_frame = get_episode_frame_range(ep_index, args.episode)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / f"episode_{args.episode:03d}_USED_EEF_marker.mp4"
    csv_path = out_dir / f"episode_{args.episode:03d}_USED_EEF_marker.csv"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )

    rows = []

    for local_i, frame_idx in enumerate(range(start_frame, end_frame, args.stride)):
        frame = dataset[frame_idx]
        state_vec = to_numpy(frame.get("observation.state"))

        if state_vec is None or state_vec.size < 6:
            continue

        q_deg = set_qpos_from_state(model, data, state_vec, rest_state_deg)

        used_eef_xyz = get_site_xyz(model, data, "end_effector_site")
        red_xyz = get_site_xyz(model, data, "DEBUG_RED_USED_EEF_SITE")
        #green_xyz = get_site_xyz(model, data, "DEBUG_GREEN_LINK6_ORIGIN")
        #yellow_xyz = get_site_xyz(model, data, "DEBUG_YELLOW_TIP_GUESS")
        link5_xyz = get_body_xyz(model, data, "link_5")
        link6_xyz = get_body_xyz(model, data, "link_6")

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lines = [
            f"Episode {args.episode:03d} | local frame={local_i} | global frame={frame_idx}",
            "RED = actual end_effector_site used for EEF height",
           # "GREEN = link_6 origin | YELLOW = guessed gripper tip",
            #f"USED EEF z={used_eef_xyz[2]:.6f} m | RED z={red_xyz[2]:.6f} m | YELLOW z={yellow_xyz[2]:.6f} m",
            f"q_deg={np.array2string(q_deg, precision=1, suppress_small=True)}",
        ]
        draw_overlay(bgr, lines)

        writer.write(bgr)

        rows.append({
            "episode": args.episode,
            "local_i": local_i,
            "global_frame": frame_idx,
            "used_eef_x": float(used_eef_xyz[0]),
            "used_eef_y": float(used_eef_xyz[1]),
            "used_eef_z": float(used_eef_xyz[2]),
            "red_debug_z": float(red_xyz[2]),
            #"green_link6_origin_z": float(green_xyz[2]),
            #"yellow_tip_guess_z": float(yellow_xyz[2]),
            "link5_body_z": float(link5_xyz[2]),
            "link6_body_z": float(link6_xyz[2]),
            "q0_deg": float(q_deg[0]),
            "q1_deg": float(q_deg[1]),
            "q2_deg": float(q_deg[2]),
            "q3_deg": float(q_deg[3]),
            "q4_deg": float(q_deg[4]),
            "q5_deg": float(q_deg[5]),
        })

    writer.release()
    renderer.close()

    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"[+] Wrote video: {video_path}")
    print(f"[+] Wrote csv  : {csv_path}")


if __name__ == "__main__":
    main()
