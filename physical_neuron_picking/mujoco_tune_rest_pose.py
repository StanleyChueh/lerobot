import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np


JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]


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


def make_debug_xml(xml_path, offwidth=960, offheight=720):
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
        raise ValueError("Cannot find body link_5.")
    if link6 is None:
        raise ValueError("Cannot find body link_6.")

    add_debug_site(
        link5,
        "DEBUG_RED_current_FK_EEF_used",
        "-0.06429 0.00327 0.0011",
        "1 0 0 1",
        size="0.040",
    )

    add_debug_site(
        link6,
        "DEBUG_YELLOW_link6_tip_guess",
        "-0.080 0 0",
        "1 1 0 1",
        size="0.040",
    )

    debug_xml_path = xml_path.with_name(xml_path.stem + "_TUNE_REST.xml")
    tree.write(debug_xml_path, encoding="utf-8", xml_declaration=True)
    return debug_xml_path


def set_qpos_deg(model, data, q_deg):
    q_deg = np.asarray(q_deg, dtype=np.float64)
    q_rad = np.deg2rad(q_deg)

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


def get_site_z(model, data, site_name):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
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


def draw_overlay(img, q_deg, step_deg, red_z, yellow_z):
    lines = [
        "Tune MuJoCo REST pose to match real robot photo",
        "Keys: q/a=j1, w/s=j2, e/d=j3, r/f=j4, t/g=j5, y/h=j6",
        "Keys: +/- change step, c save, ESC quit",
        f"step_deg = {step_deg}",
        f"q_deg = {np.array2string(q_deg, precision=2, suppress_small=True)}",
        f"RED current EEF z={red_z:.4f} | YELLOW tip guess z={yellow_z:.4f}",
    ]

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


def save_result(q_deg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "target_rest_qpos_deg.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "target_rest_deg": [float(x) for x in q_deg],
                "target_rest_rad": [float(x) for x in np.deg2rad(q_deg)],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("[SAVED TARGET REST QPOS]")
    print("=" * 80)
    print(f"target_rest_deg = {np.array2string(q_deg, precision=6, suppress_small=True)}")
    print(f"target_rest_rad = {np.array2string(np.deg2rad(q_deg), precision=6, suppress_small=True)}")
    print(f"JSON saved to: {out_json}")
    print("\nUse this in mujoco_calibrate_mapping.py:")
    print(
        "  --target_rest_deg "
        + " ".join(f"{x:.6f}" for x in q_deg)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="follower.xml")
    parser.add_argument(
        "--init_q_deg",
        nargs=6,
        type=float,
        default=[91.7, 0, 0, 0, 0, 0],
    )
    parser.add_argument("--out_dir", default="mujoco_rest_tuning")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    debug_xml = make_debug_xml(
        args.xml,
        offwidth=args.width,
        offheight=args.height,
    )

    model = mujoco.MjModel.from_xml_path(str(debug_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    q_deg = np.asarray(args.init_q_deg, dtype=np.float64)
    step_deg = 5.0

    key_map = {
        ord("q"): (0, +1),
        ord("a"): (0, -1),
        ord("w"): (1, +1),
        ord("s"): (1, -1),
        ord("e"): (2, +1),
        ord("d"): (2, -1),
        ord("r"): (3, +1),
        ord("f"): (3, -1),
        ord("t"): (4, +1),
        ord("g"): (4, -1),
        ord("y"): (5, +1),
        ord("h"): (5, -1),
    }

    print("\n" + "=" * 80)
    print("Interactive MuJoCo REST pose tuner")
    print("=" * 80)
    print("Goal: tune q_deg until MuJoCo rest pose visually matches real rest photo.")
    print("q/a: joint1 +/-")
    print("w/s: joint2 +/-")
    print("e/d: joint3 +/-")
    print("r/f: joint4 +/-")
    print("t/g: joint5 +/-")
    print("y/h: joint6 +/-")
    print("+/-: change step")
    print("c: save")
    print("ESC: quit")

    while True:
        set_qpos_deg(model, data, q_deg)

        red_z = get_site_z(model, data, "DEBUG_RED_current_FK_EEF_used")
        yellow_z = get_site_z(model, data, "DEBUG_YELLOW_link6_tip_guess")

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        draw_overlay(bgr, q_deg, step_deg, red_z, yellow_z)

        cv2.imshow("Tune REST qpos", bgr)
        key = cv2.waitKey(0)

        if key == 27:
            break

        if key == ord("c"):
            save_result(q_deg, args.out_dir)
            out_png = Path(args.out_dir) / "target_rest_pose.png"
            cv2.imwrite(str(out_png), bgr)
            print(f"Image saved to: {out_png}")
            continue

        if key == ord("+") or key == ord("="):
            step_deg *= 2.0
            print(f"step_deg = {step_deg}")
            continue

        if key == ord("-") or key == ord("_"):
            step_deg = max(0.25, step_deg / 2.0)
            print(f"step_deg = {step_deg}")
            continue

        if key in key_map:
            joint_idx, direction = key_map[key]
            q_deg[joint_idx] += direction * step_deg
            print(f"q_deg = {np.array2string(q_deg, precision=3, suppress_small=True)}")

    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
