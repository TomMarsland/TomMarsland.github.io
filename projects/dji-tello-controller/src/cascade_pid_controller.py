import json
import os
from pathlib import Path

import numpy as np


def _as_array(values):
    return np.asarray(values, dtype=np.float64)


DEFAULT_GAINS = {
    "ascend_xy_gain": 0.45,
    "ascend_xy_damp": 0.65,
    "ascend_xy_limit_high": 0.34,
    "ascend_xy_limit_low": 0.58,
    "ascend_limit_switch_z": 0.70,
    "ascend_z_gain": 1.15,
    "ascend_z_damp": 0.75,
    "ascend_z_cap": 1.00,
    "cruise_speed_far": 0.95,
    "cruise_speed_mid": 0.78,
    "cruise_speed_near": 0.50,
    "cruise_far_radius": 1.25,
    "cruise_mid_radius": 0.65,
    "cruise_xy_damp": 0.30,
    "cruise_xy_cap": 0.90,
    "cruise_z_gain": 0.95,
    "cruise_z_damp": 0.65,
    "cruise_z_cap": 0.75,
    "settle_xy_gain": 0.75,
    "settle_xy_damp": 1.85,
    "settle_xy_cap": 0.18,
    "settle_z_gain": 0.85,
    "settle_z_damp": 1.70,
    "settle_z_cap": 0.18,
    "brake_radius": 0.55,
    "brake_gain": 0.85,
    "brake_damp": 2.10,
    "brake_cap": 0.26,
    "ascend_to_cruise_z": 0.40,
    "cruise_to_ascend_z": 0.55,
    "cruise_to_settle_xy": 0.22,
    "cruise_to_settle_z": 0.14,
    "settle_to_cruise_xy": 0.34,
    "settle_to_ascend_z": 0.30,
    "yaw_kp_far": 3.6,
    "yaw_kp_near": 2.4,
    "yaw_ki": 0.05,
    "yaw_kd_far": 0.22,
    "yaw_kd_near": 0.14,
    "yaw_switch": 0.12,
    "yaw_integral_band": 0.18,
    "cmd_slew_xy": 1.6,
    "cmd_slew_z": 2.4,
    "cmd_slew_yaw": 2.8,
}


GAINS_PATH = Path(
    os.environ.get(
        "AERO_CASCADE_PID_GAINS",
        Path(__file__).with_name("cascade_pid_gains.json"),
    )
)


def load_gains():
    gains = dict(DEFAULT_GAINS)
    if GAINS_PATH.exists():
        with GAINS_PATH.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for key, value in loaded.items():
            if key in gains:
                gains[key] = float(value)
    return gains


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def world_to_body(vector_world, yaw):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rotation = np.array(
        [
            [cy, sy, 0.0],
            [-sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation @ np.asarray(vector_world, dtype=np.float64)


def clip_action(action):
    action = np.asarray(action, dtype=np.float64).copy()
    action[:3] = np.clip(action[:3], -1.0, 1.0)
    action[3] = np.clip(action[3], -1.74533, 1.74533)
    return action


def slew_limit(target, current, rate_limit, dt):
    if dt <= 0.0:
        return target
    delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    max_delta = np.asarray(rate_limit, dtype=np.float64) * dt
    return np.asarray(current, dtype=np.float64) + np.clip(delta, -max_delta, max_delta)


class ControllerStateTracker:
    def __init__(self):
        self.velocity_filter_tau = 0.10
        self.reset()

    def reset(self):
        self.prev_position_world = None
        self.prev_yaw = None
        self.filtered_velocity_body = np.zeros(3, dtype=np.float64)
        self.last_target = None

    def observe(self, state, target_pos, dt):
        state = _as_array(state)
        target_pos = _as_array(target_pos)
        position_world = state[:3]
        _, _, yaw = state[3:6]

        if self.prev_position_world is None or dt <= 0.0 or dt > 0.5:
            raw_velocity_world = np.zeros(3, dtype=np.float64)
            yaw_rate = 0.0
        else:
            raw_velocity_world = (position_world - self.prev_position_world) / dt
            yaw_rate = wrap_angle(yaw - self.prev_yaw) / dt

        velocity_body = world_to_body(raw_velocity_world, yaw)
        if self.prev_position_world is None:
            self.filtered_velocity_body = velocity_body
        else:
            alpha = min(1.0, dt / (self.velocity_filter_tau + dt))
            self.filtered_velocity_body = (
                (1.0 - alpha) * self.filtered_velocity_body + alpha * velocity_body
            )

        position_error_world = target_pos[:3] - position_world
        position_error_body = world_to_body(position_error_world, yaw)
        yaw_error = wrap_angle(target_pos[3] - yaw)

        self.prev_position_world = position_world.copy()
        self.prev_yaw = yaw
        self.last_target = target_pos.copy()
        return {
            "position_error_body": position_error_body,
            "velocity_body": self.filtered_velocity_body.copy(),
            "yaw_error": yaw_error,
            "yaw_rate": yaw_rate,
        }


class CascadePIDRuntime:
    def __init__(self, gains=None):
        self.tracker = ControllerStateTracker()
        self.gains = dict(DEFAULT_GAINS if gains is None else gains)
        self.last_target = None
        self.last_debug = {}
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.yaw_integral = 0.0
        self.last_target = None
        self.last_debug = {}
        self.mode = "ascend"
        self.prev_command = np.zeros(3, dtype=np.float64)
        self.prev_yaw_rate_cmd = 0.0

    def _select_cruise_speed(self, horizontal_error):
        g = self.gains
        if horizontal_error > g["cruise_far_radius"]:
            return g["cruise_speed_far"]
        if horizontal_error > g["cruise_mid_radius"]:
            return g["cruise_speed_mid"]
        return g["cruise_speed_near"]

    def _update_mode(self, horizontal_error, vertical_error):
        g = self.gains
        if self.mode == "ascend":
            if vertical_error < g["ascend_to_cruise_z"] or (
                horizontal_error > 0.9 and vertical_error < g["ascend_limit_switch_z"]
            ):
                self.mode = "cruise"
        elif self.mode == "cruise":
            if vertical_error > g["cruise_to_ascend_z"]:
                self.mode = "ascend"
            elif horizontal_error < g["cruise_to_settle_xy"] and abs(vertical_error) < g["cruise_to_settle_z"]:
                self.mode = "settle"
        else:
            if vertical_error > g["settle_to_ascend_z"]:
                self.mode = "ascend"
            elif horizontal_error > g["settle_to_cruise_xy"]:
                self.mode = "cruise"

    def _compute_velocity_command(self, position_error, velocity_body, dt, wind_enabled):
        g = self.gains
        horizontal_error = float(np.linalg.norm(position_error[:2]))
        vertical_error = float(position_error[2])
        self._update_mode(horizontal_error, vertical_error)

        desired_velocity = np.zeros(3, dtype=np.float64)
        xy_limit = 1.0

        if self.mode == "ascend":
            xy_limit = (
                g["ascend_xy_limit_high"]
                if vertical_error > g["ascend_limit_switch_z"]
                else g["ascend_xy_limit_low"]
            )
            desired_velocity[:2] = np.clip(
                g["ascend_xy_gain"] * position_error[:2] - g["ascend_xy_damp"] * velocity_body[:2],
                -xy_limit,
                xy_limit,
            )
            desired_velocity[2] = np.clip(
                g["ascend_z_gain"] * vertical_error - g["ascend_z_damp"] * velocity_body[2],
                -0.18,
                g["ascend_z_cap"],
            )
        elif self.mode == "cruise":
            cruise_speed = self._select_cruise_speed(horizontal_error)
            direction = position_error[:2] / max(horizontal_error, 1e-6)
            desired_velocity[:2] = np.clip(
                cruise_speed * direction - g["cruise_xy_damp"] * velocity_body[:2],
                -g["cruise_xy_cap"],
                g["cruise_xy_cap"],
            )
            desired_velocity[2] = np.clip(
                g["cruise_z_gain"] * vertical_error - g["cruise_z_damp"] * velocity_body[2],
                -0.25,
                g["cruise_z_cap"],
            )
        else:
            xy_cap = g["settle_xy_cap"]
            z_cap = g["settle_z_cap"]
            if horizontal_error > g["brake_radius"]:
                xy_cap = g["brake_cap"]
                desired_velocity[:2] = np.clip(
                    g["brake_gain"] * position_error[:2] - g["brake_damp"] * velocity_body[:2],
                    -xy_cap,
                    xy_cap,
                )
            else:
                desired_velocity[:2] = np.clip(
                    g["settle_xy_gain"] * position_error[:2] - g["settle_xy_damp"] * velocity_body[:2],
                    -xy_cap,
                    xy_cap,
                )
            desired_velocity[2] = np.clip(
                g["settle_z_gain"] * vertical_error - g["settle_z_damp"] * velocity_body[2],
                -z_cap,
                z_cap,
            )
            xy_limit = xy_cap

        if wind_enabled:
            desired_velocity[:2] *= 1.08
            desired_velocity[2] *= 1.03

        commanded_velocity = slew_limit(
            desired_velocity,
            self.prev_command,
            [g["cmd_slew_xy"], g["cmd_slew_xy"], g["cmd_slew_z"]],
            dt,
        )
        commanded_velocity = np.clip(commanded_velocity, -1.0, 1.0)
        self.prev_command = commanded_velocity.copy()
        return desired_velocity, commanded_velocity, xy_limit

    def controller(self, state, target_pos, dt, wind_enabled=False):
        target_pos = _as_array(target_pos)
        if self.last_target is None or np.linalg.norm(target_pos - self.last_target) > 1e-9:
            self.yaw_integral = 0.0
            self.last_target = target_pos.copy()
            self.mode = "ascend"
            self.prev_command[:] = 0.0
            self.prev_yaw_rate_cmd = 0.0

        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / 50.0

        features = self.tracker.observe(state, target_pos, dt)
        position_error = features["position_error_body"]
        velocity_body = features["velocity_body"]
        yaw_error = features["yaw_error"]
        yaw_rate = features["yaw_rate"]
        g = self.gains

        desired_velocity, commanded_velocity, xy_limit = self._compute_velocity_command(
            position_error,
            velocity_body,
            dt,
            wind_enabled=wind_enabled,
        )

        yaw_kp = g["yaw_kp_far"] if abs(yaw_error) > g["yaw_switch"] else g["yaw_kp_near"]
        yaw_ki = g["yaw_ki"] if abs(yaw_error) < g["yaw_integral_band"] else 0.0
        yaw_kd = g["yaw_kd_far"] if abs(yaw_error) > g["yaw_switch"] else g["yaw_kd_near"]
        self.yaw_integral = float(np.clip(self.yaw_integral + yaw_error * dt, -0.6, 0.6))
        yaw_rate_cmd = yaw_kp * yaw_error + yaw_ki * self.yaw_integral - yaw_kd * yaw_rate
        yaw_rate_cmd = float(
            slew_limit(
                [yaw_rate_cmd],
                [self.prev_yaw_rate_cmd],
                [g["cmd_slew_yaw"]],
                dt,
            )[0]
        )
        self.prev_yaw_rate_cmd = yaw_rate_cmd

        final_action = clip_action(np.concatenate((commanded_velocity, [yaw_rate_cmd])))
        self.last_debug = {
            "position_error_body": position_error.copy(),
            "velocity_body": velocity_body.copy(),
            "desired_velocity": desired_velocity.copy(),
            "commanded_velocity": commanded_velocity.copy(),
            "yaw_error": float(yaw_error),
            "yaw_rate_cmd": float(yaw_rate_cmd),
            "xy_limit": float(xy_limit),
            "mode": self.mode,
        }
        return tuple(final_action.tolist())


_RUNTIME = CascadePIDRuntime(gains=load_gains())


def controller(state, target_pos, dt, wind_enabled=False):
    return _RUNTIME.controller(state, target_pos, dt, wind_enabled=wind_enabled)


def controller_status():
    return f"Cascade PID active (gains: {GAINS_PATH})"


def controller_reset():
    _RUNTIME.reset()


def controller_debug():
    return dict(_RUNTIME.last_debug)
