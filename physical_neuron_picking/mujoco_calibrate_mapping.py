'''
python mujoco_calibrate_mapping.py \
  --xml follower.xml \
  --rest_json real_pose_calibration_outputs/rest_state.json \
  --reach_json real_pose_calibration_outputs/reach_state.json \
  --raise_json real_pose_calibration_outputs/raise_state.json \
  --out_dir mujoco_mapping_calibration_search_delta_order_scale \
  --target_rest_deg 91.7 15.0 40.0 65.0 0.0 -30.0 \
  --vary_joints 1 2 3
'''
import argparse
import itertools
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


def load_state_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.asarray(data["state_vec"], dtype=np.float64)


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

    debug_xml_path = xml_path.with_name(xml_path.stem + "_CALIB_SEARCH.xml")
    tree.write(debug_xml_path, encoding="utf-8", xml_declaration=True)
    return debug_xml_path


def set_qpos_delta_calibrated(
    model,
    data,
    raw_state_deg,
    rest_state_deg,
    target_rest_deg,
    sign,
    order=None,
    scale=None,
):
    raw_state_deg = np.asarray(raw_state_deg, dtype=np.float64)[:6]
    rest_state_deg = np.asarray(rest_state_deg, dtype=np.float64)[:6]
    target_rest_deg = np.asarray(target_rest_deg, dtype=np.float64)[:6]

    sign = np.asarray(sign, dtype=np.float64)[:6]

    if order is None:
        order = np.arange(6)
    else:
        order = np.asarray(order, dtype=int)

    if scale is None:
        scale = np.ones(6, dtype=np.float64)
    else:
        scale = np.asarray(scale, dtype=np.float64)[:6]

    # Important:
    # Use rest-relative movement, not absolute raw joint value.
    # This keeps REST fixed and tests whether REACH/RAISE deltas map correctly.
    raw_delta = raw_state_deg[order] - rest_state_deg[order]
    q_deg = target_rest_deg + sign * scale * raw_delta

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
    return q_deg, q_rad


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


def draw_overlay(img, lines):
    y = 28
    for line in lines:
        cv2.putText(
            img,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22


def render_pose(
    model,
    data,
    renderer,
    cam,
    pose_name,
    raw_state,
    rest_state,
    target_rest_deg,
    sign,
    order,
    scale,
):
    q_deg, _ = set_qpos_delta_calibrated(
        model=model,
        data=data,
        raw_state_deg=raw_state,
        rest_state_deg=rest_state,
        target_rest_deg=target_rest_deg,
        sign=sign,
        order=order,
        scale=scale,
    )

    red_z = get_site_z(model, data, "DEBUG_RED_current_FK_EEF_used")
    yellow_z = get_site_z(model, data, "DEBUG_YELLOW_link6_tip_guess")

    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    lines = [
        f"pose={pose_name}",
        f"RED current EEF z={red_z:.4f} | YELLOW tip guess z={yellow_z:.4f}",
        f"q_deg={np.array2string(q_deg, precision=1, suppress_small=True)}",
        f"sign={sign.tolist()}",
        f"order={order.tolist()}",
        f"scale={scale.tolist()}",
    ]
    draw_overlay(bgr, lines)
    return bgr


def make_contact_sheet(images, labels):
    labeled = []

    for img, label in zip(images, labels):
        copy_img = img.copy()
        cv2.putText(
            copy_img,
            label,
            (15, copy_img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled.append(copy_img)

    return np.concatenate(labeled, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="follower.xml")
    parser.add_argument("--rest_json", required=True)
    parser.add_argument("--reach_json", required=True)
    parser.add_argument("--raise_json", required=True)
    parser.add_argument("--out_dir", default="mujoco_mapping_calibration_search")
    parser.add_argument(
        "--target_rest_deg",
        nargs=6,
        type=float,
        default=[91.673, 0.0, 0.0, 0.0, 0.0, 0.0],
        help=(
            "MuJoCo qpos target for real rest pose, in degrees. "
            "Default is XML keyframe home qpos = [1.6,0,0,0,0,0] rad converted to degrees."
        ),
    )
    parser.add_argument(
        "--vary_joints",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Only enumerate sign flips for these joints. Gripper/wrist_roll usually do not affect arm pose much.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rest_state = load_state_json(args.rest_json)
    reach_state = load_state_json(args.reach_json)
    raise_state = load_state_json(args.raise_json)

    target_rest_deg = np.asarray(args.target_rest_deg, dtype=np.float64)

    
    joint_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    print("\n" + "=" * 100)
    print("REAL ROBOT STATE DELTAS")
    print("=" * 100)
    for i, name in enumerate(joint_names):
        print(
            f"{i} {name:<18} | "
            f"rest={rest_state[i]: .3f} | "
            f"reach={reach_state[i]: .3f} | "
            f"raise={raise_state[i]: .3f} | "
            f"reach-rest={reach_state[i] - rest_state[i]: .3f} | "
            f"raise-rest={raise_state[i] - rest_state[i]: .3f}"
        )

    debug_xml = make_debug_xml(
        args.xml,
        offwidth=args.width,
        offheight=args.height,
    )

    model = mujoco.MjModel.from_xml_path(str(debug_xml))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    base_sign = np.ones(6, dtype=np.float64)

    # Try possible joint-order mappings.
    # The most suspicious joints are:
    #   state[1] shoulder_lift
    #   state[2] elbow_flex
    #   state[3] wrist_flex
    #
    # We keep pan, wrist_roll, gripper fixed first.
    orders_to_test = [
        [0, 1, 2, 3, 4, 5],  # original
        [0, 2, 1, 3, 4, 5],  # swap shoulder_lift / elbow
        [0, 1, 3, 2, 4, 5],  # swap elbow / wrist_flex
        [0, 3, 2, 1, 4, 5],  # reverse lift/wrist relation
        [0, 2, 3, 1, 4, 5],
        [0, 3, 1, 2, 4, 5],
    ]

    # Try simple scale first.
    # If sign/order is correct but movement magnitude is too large/small,
    # scale will help diagnose that.
    scales_to_test = [
        [1, 1, 1, 1, 1, 1],
        [1, 0.5, 1, 1, 1, 1],
        [1, 1, 0.5, 1, 1, 1],
        [1, 1, 1, 0.5, 1, 1],
        [1, 0.5, 0.5, 1, 1, 1],
        [1, 1, 0.5, 0.5, 1, 1],
        [1, 1.5, 1, 1, 1, 1],
        [1, 1, 1.5, 1, 1, 1],
        [1, 1, 1, 1.5, 1, 1],
    ]

    variants = []

    for order in orders_to_test:
        order = np.asarray(order, dtype=int)

        for scale in scales_to_test:
            scale = np.asarray(scale, dtype=np.float64)

            for signs_for_varied in itertools.product([-1.0, 1.0], repeat=len(args.vary_joints)):
                sign = base_sign.copy()
                for joint_idx, s in zip(args.vary_joints, signs_for_varied):
                    sign[joint_idx] = s

                variant_name = (
                    "order_"
                    + "_".join(str(x) for x in order.tolist())
                    + "__scale_"
                    + "_".join(f"{x:.1f}" for x in scale.tolist())
                    + "__sign_"
                    + "_".join(
                        f"j{j}{'p' if sign[j] > 0 else 'n'}"
                        for j in range(6)
                    )
                )

                variants.append((variant_name, sign, order, scale))

    print("\n" + "=" * 100)
    print("CALIBRATION SEARCH")
    print("=" * 100)
    print(f"rest_state      = {rest_state}")
    print(f"target_rest_deg = {target_rest_deg}")
    print(f"num_variants    = {len(variants)}")
    print(f"output dir      = {out_dir}")

    summary_rows = []

    for variant_idx, (variant_name, sign, order, scale) in enumerate(variants):
        rest_img = render_pose(
            model,
            data,
            renderer,
            cam,
            "rest",
            rest_state,
            rest_state,
            target_rest_deg,
            sign,
            order,
            scale,
        )

        reach_img = render_pose(
            model,
            data,
            renderer,
            cam,
            "reach",
            reach_state,
            rest_state,
            target_rest_deg,
            sign,
            order,
            scale,
        )

        raise_img = render_pose(
            model,
            data,
            renderer,
            cam,
            "raise",
            raise_state,
            rest_state,
            target_rest_deg,
            sign,
            order,
            scale,
        )

        sheet = make_contact_sheet(
            [rest_img, reach_img, raise_img],
            ["REST", "REACH", "RAISE"],
        )

        out_path = out_dir / f"{variant_idx:03d}_{variant_name}.png"
        cv2.imwrite(str(out_path), sheet)

        summary_rows.append(
            {
                "variant_idx": variant_idx,
                "variant_name": variant_name,
                "sign": sign.tolist(),
                "order": order.tolist(),
                "scale": scale.tolist(),
                "target_rest_deg": target_rest_deg.tolist(),
                "image": str(out_path),
            }
        )

        print(
            f"[{variant_idx:03d}] {variant_name} | "
            f"sign={sign.tolist()} | "
            f"order={order.tolist()} | "
            f"scale={scale.tolist()} | "
            f"image={out_path}"
        )

    renderer.close()

    summary_path = out_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n[+] Wrote summary:", summary_path)
    print("\nOpen the PNG files and pick the variant where reach and raise match the real robot photos.")


if __name__ == "__main__":
    main()
