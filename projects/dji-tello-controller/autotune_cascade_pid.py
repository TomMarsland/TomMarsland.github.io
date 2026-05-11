import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from src.cascade_pid_controller import CascadePIDRuntime, DEFAULT_GAINS
from src.tello_controller import TelloController


PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
TARGETS_PATH = PROJECT_ROOT / "targets.csv"
GAINS_PATH = PROJECT_ROOT / "src" / "cascade_pid_gains.json"
LOGS_DIR = PROJECT_ROOT / "logs"
AUTOTUNE_CSV = LOGS_DIR / "autotune_results.csv"

M = 0.088
L = 0.06
KF = 0.566e-5
KM = 0.762e-7
TM = 0.0163
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2], dtype=np.float64)


SEARCH_SPACE = {
    "ascend_xy_gain": (0.20, 0.90),
    "ascend_xy_damp": (0.30, 1.60),
    "ascend_xy_limit_high": (0.08, 0.35),
    "ascend_xy_limit_low": (0.16, 0.55),
    "ascend_limit_switch_z": (0.20, 0.80),
    "ascend_z_gain": (0.70, 1.80),
    "ascend_z_damp": (0.40, 1.60),
    "ascend_z_cap": (0.45, 1.00),
    "cruise_speed_far": (0.45, 1.00),
    "cruise_speed_mid": (0.30, 0.90),
    "cruise_speed_near": (0.15, 0.60),
    "cruise_far_radius": (0.80, 1.80),
    "cruise_mid_radius": (0.35, 1.00),
    "cruise_xy_damp": (0.10, 1.10),
    "cruise_xy_cap": (0.40, 1.00),
    "cruise_z_gain": (0.40, 1.40),
    "cruise_z_damp": (0.30, 1.40),
    "cruise_z_cap": (0.20, 0.80),
    "settle_xy_gain": (0.30, 1.60),
    "settle_xy_damp": (0.60, 2.40),
    "settle_xy_cap": (0.08, 0.45),
    "settle_z_gain": (0.30, 1.60),
    "settle_z_damp": (0.60, 2.40),
    "settle_z_cap": (0.08, 0.40),
    "brake_radius": (0.25, 0.90),
    "brake_gain": (0.40, 1.80),
    "brake_damp": (0.80, 2.60),
    "brake_cap": (0.15, 0.65),
    "ascend_to_cruise_z": (0.05, 0.35),
    "cruise_to_ascend_z": (0.10, 0.45),
    "cruise_to_settle_xy": (0.10, 0.50),
    "cruise_to_settle_z": (0.05, 0.30),
    "settle_to_cruise_xy": (0.20, 0.70),
    "settle_to_ascend_z": (0.10, 0.45),
    "yaw_kp_far": (2.0, 5.0),
    "yaw_kp_near": (1.2, 3.5),
    "yaw_ki": (0.0, 0.15),
    "yaw_kd_far": (0.05, 0.40),
    "yaw_kd_near": (0.03, 0.30),
    "yaw_switch": (0.05, 0.30),
    "yaw_integral_band": (0.08, 0.35),
    "cmd_slew_xy": (0.6, 3.0),
    "cmd_slew_z": (0.8, 4.0),
    "cmd_slew_yaw": (0.8, 4.0),
}

ACTIVE_TUNING_KEYS = [
    "ascend_xy_limit_high",
    "ascend_xy_limit_low",
    "ascend_z_gain",
    "ascend_z_damp",
    "cruise_speed_far",
    "cruise_speed_mid",
    "cruise_xy_damp",
    "cruise_z_gain",
    "cruise_z_damp",
    "settle_xy_gain",
    "settle_xy_damp",
    "settle_xy_cap",
    "settle_z_gain",
    "settle_z_damp",
    "settle_z_cap",
    "brake_radius",
    "brake_gain",
    "brake_damp",
    "brake_cap",
    "cruise_to_settle_xy",
    "cruise_to_settle_z",
    "settle_to_cruise_xy",
    "cmd_slew_xy",
    "cmd_slew_z",
]


def load_targets():
    targets = []
    with TARGETS_PATH.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 4:
                targets.append(tuple(float(v) for v in row))
    return targets


def motor_model(desired_rpm, current_rpm, dt):
    rpm_derivative = (desired_rpm - current_rpm) / TM
    return current_rpm + rpm_derivative * dt


def compute_dynamics(rpm_values, lin_vel_world, quat):
    rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    omega = rpm_values * (2 * np.pi / 60)
    omega_squared = omega**2
    motor_forces = omega_squared * KF
    thrust = np.array([0, 0, np.sum(motor_forces)], dtype=np.float64)
    vel_body = np.dot(rotation.T, lin_vel_world)
    drag_body = -K_TRANS * vel_body
    force = drag_body + thrust
    z_torques = omega_squared * KM
    z_torque = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
    x_torque = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * L
    y_torque = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * L
    return force, np.array([x_torque, y_torque, z_torque], dtype=np.float64)


def spin_motors(drone_id, rpm, timestep):
    for joint_index in range(4):
        rad_s = rpm[joint_index] * (2.0 * np.pi / 60.0)
        current_angle = p.getJointState(drone_id, joint_index)[0]
        new_angle = current_angle + rad_s * timestep
        p.resetJointState(drone_id, joint_index, new_angle)


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def load_json_gains():
    gains = deepcopy(DEFAULT_GAINS)
    if GAINS_PATH.exists():
        with GAINS_PATH.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for key, value in loaded.items():
            if key in gains:
                gains[key] = float(value)
    return gains


def write_gains(gains):
    with GAINS_PATH.open("w", encoding="utf-8") as f:
        json.dump(gains, f, indent=2)
        f.write("\n")


def append_result(row):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    exists = AUTOTUNE_CSV.exists()
    with AUTOTUNE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "iteration",
                    "score",
                    "final_pos_mean",
                    "final_yaw_mean",
                    "mean_pos_mean",
                    "tail_pos_mean",
                    "tail_yaw_mean",
                    "success_rate",
                    "time_to_20_mean",
                ]
            )
        writer.writerow(row)


def clamp_gain(key, value):
    lo, hi = SEARCH_SPACE[key]
    return float(np.clip(value, lo, hi))


def normalize_gains(gains):
    gains = deepcopy(gains)
    gains["ascend_xy_limit_low"] = max(gains["ascend_xy_limit_low"], gains["ascend_xy_limit_high"] + 0.02)
    gains["cruise_speed_mid"] = min(gains["cruise_speed_mid"], gains["cruise_speed_far"] - 0.05)
    gains["cruise_speed_near"] = min(gains["cruise_speed_near"], gains["cruise_speed_mid"] - 0.05)
    gains["cruise_mid_radius"] = min(gains["cruise_mid_radius"], gains["cruise_far_radius"] - 0.10)
    gains["cruise_to_ascend_z"] = max(gains["cruise_to_ascend_z"], gains["ascend_to_cruise_z"] + 0.03)
    gains["settle_to_cruise_xy"] = max(gains["settle_to_cruise_xy"], gains["cruise_to_settle_xy"] + 0.05)
    gains["settle_to_ascend_z"] = max(gains["settle_to_ascend_z"], gains["cruise_to_settle_z"] + 0.05)
    gains["yaw_kp_far"] = max(gains["yaw_kp_far"], gains["yaw_kp_near"] + 0.20)
    gains["yaw_kd_far"] = max(gains["yaw_kd_far"], gains["yaw_kd_near"] + 0.02)
    gains["cmd_slew_z"] = max(gains["cmd_slew_z"], gains["cmd_slew_xy"])
    gains["brake_radius"] = max(gains["brake_radius"], gains["cruise_to_settle_xy"] + 0.10)
    return gains


def evaluate_gains(gains, targets, horizon_seconds=10.0):
    gains = normalize_gains(gains)
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(str(RESOURCES_DIR))
    timestep = 1.0 / 1000.0
    pos_control_timestep = 1.0 / 50.0
    steps_between = int(round(pos_control_timestep / timestep))
    total_steps = int(horizon_seconds / timestep)
    controller = CascadePIDRuntime(gains=gains)

    try:
        episode_metrics = []
        for target in targets:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.loadURDF(str(Path(pybullet_data.getDataPath()) / "plane.urdf"))
            drone_id = p.loadURDF(
                str(RESOURCES_DIR / "tello.urdf"),
                [0, 0, 1],
                p.getQuaternionFromEuler([0, 0, 0]),
            )
            tello = TelloController(9.81, M, L, 0.70, KF, KM)
            controller.reset()
            prev_rpm = np.zeros(4, dtype=np.float64)
            desired_vel = np.zeros(3, dtype=np.float64)
            yaw_rate_setpoint = 0.0
            loop_counter = 0
            reached_time = None
            first_under_20 = None
            pos_errors = []
            yaw_errors = []
            settle_failures = 0

            for step in range(total_steps):
                loop_counter += 1
                pos, quat = p.getBasePositionAndOrientation(drone_id)
                lin_vel_world, ang_vel_world = p.getBaseVelocity(drone_id)
                roll, pitch, yaw = p.getEulerFromQuaternion(quat)
                yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
                _, inverted_quat = p.invertTransform([0, 0, 0], quat)
                _, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
                lin_vel = np.array(p.rotateVector(inverted_quat_yaw, lin_vel_world))
                ang_vel = np.array(p.rotateVector(inverted_quat, ang_vel_world))

                if loop_counter >= steps_between:
                    loop_counter = 0
                    state = np.concatenate((pos, [roll, pitch, yaw]))
                    out = controller.controller(state, target, pos_control_timestep, wind_enabled=False)
                    desired_vel = np.array(out[:3], dtype=np.float64)
                    yaw_rate_setpoint = float(out[3])

                rpm = tello.compute_control(
                    desired_vel,
                    lin_vel,
                    quat,
                    ang_vel,
                    yaw_rate_setpoint,
                    timestep,
                )
                rpm = motor_model(rpm, prev_rpm, timestep)
                prev_rpm = rpm
                force, torque = compute_dynamics(rpm, np.array(lin_vel_world), quat)
                p.applyExternalForce(drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
                p.applyExternalTorque(drone_id, -1, torque, p.LINK_FRAME)
                spin_motors(drone_id, rpm, timestep)
                p.stepSimulation()

                pos_error = float(np.linalg.norm(np.array(target[:3]) - np.array(pos)))
                yaw_error = float(abs(wrap_angle(target[3] - yaw)))
                pos_errors.append(pos_error)
                yaw_errors.append(yaw_error)
                if first_under_20 is None and pos_error < 0.20 and yaw_error < 0.10:
                    first_under_20 = step * timestep
                if reached_time is None and pos_error < 0.05 and yaw_error < 0.05:
                    reached_time = step * timestep
                if first_under_20 is not None and pos_error > 0.30:
                    settle_failures += 1

            tail_samples = max(1, int(1.0 / timestep))
            final_pos, final_quat = p.getBasePositionAndOrientation(drone_id)
            final_yaw = p.getEulerFromQuaternion(final_quat)[2]
            final_pos_error = float(np.linalg.norm(np.array(target[:3]) - np.array(final_pos)))
            final_yaw_error = float(abs(wrap_angle(target[3] - final_yaw)))
            episode_metrics.append(
                {
                    "target": target,
                    "final_pos_error": final_pos_error,
                    "final_yaw_error": final_yaw_error,
                    "mean_pos_error": float(np.mean(pos_errors)),
                    "tail_pos_error": float(np.mean(pos_errors[-tail_samples:])),
                    "tail_yaw_error": float(np.mean(yaw_errors[-tail_samples:])),
                    "reached_time": reached_time,
                    "first_under_20": first_under_20,
                    "settle_failures": settle_failures,
                }
            )

        final_pos_mean = float(np.mean([m["final_pos_error"] for m in episode_metrics]))
        final_yaw_mean = float(np.mean([m["final_yaw_error"] for m in episode_metrics]))
        mean_pos_mean = float(np.mean([m["mean_pos_error"] for m in episode_metrics]))
        tail_pos_mean = float(np.mean([m["tail_pos_error"] for m in episode_metrics]))
        tail_yaw_mean = float(np.mean([m["tail_yaw_error"] for m in episode_metrics]))
        success_count = sum(1 for m in episode_metrics if m["reached_time"] is not None and m["reached_time"] <= horizon_seconds)
        success_rate = success_count / max(len(episode_metrics), 1)
        successful_20 = [m["first_under_20"] for m in episode_metrics if m["first_under_20"] is not None]
        time_to_20_mean = float(np.mean(successful_20)) if successful_20 else horizon_seconds
        settle_failures_mean = float(np.mean([m["settle_failures"] for m in episode_metrics]))

        score = (
            1500.0 * final_pos_mean
            + 350.0 * tail_pos_mean
            + 80.0 * mean_pos_mean
            + 120.0 * final_yaw_mean
            + 90.0 * tail_yaw_mean
            + 160.0 * time_to_20_mean
            + 1.2 * settle_failures_mean
            + 1200.0 * (1.0 - success_rate)
        )
        if success_rate < 1.0:
            score += 1800.0
        return {
            "score": float(score),
            "final_pos_mean": final_pos_mean,
            "final_yaw_mean": final_yaw_mean,
            "mean_pos_mean": mean_pos_mean,
            "tail_pos_mean": tail_pos_mean,
            "tail_yaw_mean": tail_yaw_mean,
            "time_to_20_mean": time_to_20_mean,
            "success_rate": success_rate,
            "episodes": episode_metrics,
        }
    finally:
        p.disconnect()


def run_twiddle(initial_gains, targets, horizon, passes, step_fraction, tolerance):
    best_gains = normalize_gains(deepcopy(initial_gains))
    best_metrics = evaluate_gains(best_gains, targets, horizon_seconds=horizon)
    best_score = best_metrics["score"]

    steps = {}
    for key in ACTIVE_TUNING_KEYS:
        lo, hi = SEARCH_SPACE[key]
        span = hi - lo
        steps[key] = max(span * step_fraction, span * 0.01)

    print(
        f"BASE score={best_score:.2f}"
        f" final_pos={best_metrics['final_pos_mean']:.3f}"
        f" tail_pos={best_metrics['tail_pos_mean']:.3f}"
        f" time20={best_metrics['time_to_20_mean']:.2f}"
        f" success={best_metrics['success_rate']:.2f}"
    )
    append_result(
        [
            "base",
            best_score,
            best_metrics["final_pos_mean"],
            best_metrics["final_yaw_mean"],
            best_metrics["mean_pos_mean"],
            best_metrics["tail_pos_mean"],
            best_metrics["tail_yaw_mean"],
            best_metrics["success_rate"],
            best_metrics["time_to_20_mean"],
        ]
    )

    for sweep in range(1, passes + 1):
        print(f"SWEEP {sweep:03d} step_sum={sum(steps.values()):.4f}")
        improved_this_sweep = False
        for key in ACTIVE_TUNING_KEYS:
            original = best_gains[key]
            accepted = False
            for direction, sign in (("plus", 1.0), ("minus", -1.0)):
                candidate = deepcopy(best_gains)
                candidate[key] = clamp_gain(key, original + sign * steps[key])
                candidate = normalize_gains(candidate)
                metrics = evaluate_gains(candidate, targets, horizon_seconds=horizon)
                append_result(
                    [
                        f"{sweep}.{key}.{direction}",
                        metrics["score"],
                        metrics["final_pos_mean"],
                        metrics["final_yaw_mean"],
                        metrics["mean_pos_mean"],
                        metrics["tail_pos_mean"],
                        metrics["tail_yaw_mean"],
                        metrics["success_rate"],
                        metrics["time_to_20_mean"],
                    ]
                )
                print(
                    f"TRY {direction:5s} {key:20s}"
                    f" step={steps[key]:.4f}"
                    f" score={metrics['score']:.2f}"
                    f" final_pos={metrics['final_pos_mean']:.3f}"
                    f" tail_pos={metrics['tail_pos_mean']:.3f}"
                    f" time20={metrics['time_to_20_mean']:.2f}"
                    f" success={metrics['success_rate']:.2f}"
                )
                if metrics["score"] < best_score:
                    best_gains = candidate
                    best_metrics = metrics
                    best_score = metrics["score"]
                    steps[key] *= 1.12
                    improved_this_sweep = True
                    accepted = True
                    write_gains(best_gains)
                    print(f"ACCEPT {direction:5s} {key} -> best_score={best_score:.2f}")
                    break

            if not accepted:
                steps[key] *= 0.70

        if sum(steps.values()) < tolerance:
            print(f"Stopping: step sum {sum(steps.values()):.4f} < tolerance {tolerance:.4f}")
            break
        if not improved_this_sweep:
            print("No improvement this sweep; continuing with reduced steps.")

    write_gains(best_gains)
    print(f"Best gains saved to {GAINS_PATH}")
    print(f"Autotune history saved to {AUTOTUNE_CSV}")
    return best_gains, best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-tune the staged cascade controller against the coursework targets.")
    parser.add_argument("--method", choices=["twiddle"], default="twiddle")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--step-fraction", type=float, default=0.08)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--reset-defaults", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = load_targets()
    base_gains = deepcopy(DEFAULT_GAINS) if args.reset_defaults else load_json_gains()
    run_twiddle(
        initial_gains=base_gains,
        targets=targets,
        horizon=args.horizon,
        passes=args.passes,
        step_fraction=args.step_fraction,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
