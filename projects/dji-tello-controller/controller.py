# wind_flag = True
# Cascade PID controller for 3D position and yaw stabilisation.

from src.cascade_pid_controller import controller as _cascade_pid_controller
from src.cascade_pid_controller import controller_debug as _cascade_pid_debug
from src.cascade_pid_controller import controller_reset as _cascade_pid_reset


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    return _cascade_pid_controller(state, target_pos, dt, wind_enabled=wind_enabled)


def controller_status():
    return "Cascade PID active"


def controller_reset():
    _cascade_pid_reset()


def controller_debug():
    return _cascade_pid_debug()
