# DJI Tello Drone Flying Controller

University drone control project using a simulated DJI Tello environment. The project includes the main simulator entry points, Tello controller code, PID/cascade PID controller work, wind model and Tello URDF/STL resources.

## Main files

- 
un.py - simulation runner.
- controller.py - top-level controller entry point.
- src/tello_controller.py - Tello control logic.
- src/PID_controller.py - base PID controller.
- src/cascade_pid_controller.py - cascade PID controller used for the drone task.
- src/cascade_pid_gains.json - tuned controller gains.
- 
esources/ - Tello model resources used by the environment.

Generated logs, cache folders, Mac metadata and trained model archives are excluded from this public portfolio copy.
