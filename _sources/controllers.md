# Controllers

The `whoc.controllers` module contains a library of wind and hybrid power plant
controllers. Each controller must inherit from `ControllerBase` (see 
controller_base.py) and implement a
mandatory `compute_setpoints()` method, which contains the relevant control 
algorithm and writes final control signals to the `setpoints_dict` attribute 
as key-value pairs. `compute_setpoints()` is, in turn, called in the `step()`
method of `ControllerBase`.

## Available controllers

### WakeSteeringADStandin
For yaw controller of actuator disk-type turbines (as a stand-in, will be 
updated).

### WakeSteeringROSCOStandin
May be combined into a universal simple wake steering controller.