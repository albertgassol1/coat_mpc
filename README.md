<p align="center">

  <h1 align="center">COAt-MPC: Performance-driven Constrained Optimal Auto-Tuner for MPC</h1>
  <p align="center">
    <a href="https://albertgassol1.github.io/">Albert Gassol Puigjaner</a>
    ·
    <a href="https://inf.ethz.ch/people/people-atoz/person-detail.MjQyMTM4.TGlzdC8zMDQsLTIxNDE4MTU0NjA=.html">Manish Prajapat</a>
    ·
    <a href="https://n.ethz.ch/~carrona/">Andrea Carron</a>
    ·
    <a href="https://las.inf.ethz.ch/krausea">Andreas Krause</a>
    ·
    <a href="https://idsc.ethz.ch/research-zeilinger/people/person-detail.MTQyNzM3.TGlzdC8xOTI5LDg4NTM5MTE3.html">Melanie N. Zeilinger</a>

  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2503.07127v1">ArXiv</a> | <a href="https://ieeexplore.ieee.org/document/10924398">IEEE RA-L</a> | <a href="https://albertgassol1.github.io/coat_mpc/">Project Page</a> </h3>
  <div align="center"></div>
</p>
Official implementation of COAt-MPC. COAt-MPC automatically tunes a Model Predictive Controller (MPC) 's cost function weights while always satisfying a performance constraint with high probability. The code of COATt-MPC reuses parts of the code of SafeOpt (https://github.com/befelix/SafeOpt). Furthermore, other methods such as Upper Confidence Bounds (UCB), Constrained Expected Improvement (EIC), Confidence Region BO (code reused from https://github.com/boschresearch/ConfidenceRegionBO), Weighted Maximum Likelihood (WML) and Metropolis-Hastings (MH). The pictures below show a comparison of the different algorithms tested on a real small-scale racing car.

Besides COAt-MPC, we provide the interface to tune an MPC for autonomous racing applications. The code uses ROS (Robot Operating System) to communicate with the MPC via dynamic reconfigure. The goal is to minimize the lap time while ensuring that it will always be below a certain threshold. 

<p align="center">
  <img src="assets/coat_mpc.png" width="400" />
  <img src="assets/safeopt_original.png" width="400" />
  <img src="assets/ucb.png" width="400" />
  <img src="assets/wml.png" width="400" />
</p>

This method has been tested with a modified version of the  CRS framework: [Chronos and CRS: Design of a miniature car-like robot and a software
framework for single and multi-agent robotics and control](https://arxiv.org/pdf/2209.12048.pdf). This framework includes an open-source simulator, estimator, MPCC, and many other functionalities. Code available here: https://gitlab.ethz.ch/ics/crs.

Additionally, COAt-MPC was used to tune the cost function weights of [AMZ](https://www.amzracing.ch/en)'s driverless racing car.

## Results Video
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Ep_BX3BDaeU/0.jpg)](https://www.youtube.com/watch?v=Ep_BX3BDaeU)

## Requirements
1. [ROS](https://wiki.ros.org/noetic) (Tested on version Noetic)

2. Python packages (see ```requirements.txt```)

3. This library assumes that the user has an MPC implementation with a [dynamic reconfigure](http://wiki.ros.org/dynamic_reconfigure) client that can be used to change its parameters in an online manner. For an MPCC for autonomous racing implementation (no dynamic reconfiguring) check https://github.com/alexliniger/MPCC or the CRS implementation.

4. (Optional) For the autonomous racing experiment, this library assumes that a racing simulator is available. The simulator or a trajectory optimizer must provide information about the track center. For a racing car simulator implementation check https://github.com/AMZ-Driverless/fssim or the CRS implementation.

5. (Optional) For the autonomous racing experiment, this library assumes that the message [car_state](./coat_mpc_msgs/msg/car_state.msg) is sent by the MPC/state estimator.

6. (Optional) For the autonomous racing experiment, all tests have been performed in an environment where the track is known beforehand (Trackdrive/Skidpad disciplines in Formula Student Driverless). 

## Usage
1. Clone this repository
```bash
git clone https://github.com/albertgassol1/safeopt_mpc.git
```
2. Install python packages (Tested on Python 3.8.10)
```bash
pip install -r requirements.txt
```
3. Modify [config/interfaces.yaml](./coat_mpc_ros_interface/config/interfaces.yaml) with the correct ROS message names from your simulator, MPC controller and state estimator, and the simulator+MPC launch file name. 

4. Modify [config/tunable_weights.yaml](./coat_mpc_ros_interface/config/tunable_weights.yaml) with the MPC weights you want to tune. Include an initial set of values, upper and lower bounds and the MPC dynamic reconfigure server name. 
Make sure that the ROS environment is set up and that the ROS master is running.

5. Modify [config/config.yaml](./coat_mpc_ros_interface/config/config.yaml) with the desired optimization configuration. IMPORTANT: Add a rough estimate of the optimal lap time, a penalty time for cases where the MPC goes out of track (typically the average time*1.5), and the number of laps that you want to use for the optimization.

## Run the code (Autonomous racing application)

Include this package in your ROS workspace. Build everything(```catkin build coat_mpc*```) and run:
```bash
roslaunch coat_mpc_ros_interface autotuner.launch
```

The code will start communicating with the MPC via dynamic reconfigure and start tuning the cost function weights.
The optimal laptimes and parameters will be saved in a folder inside [visualization/objects](./coat_mpc_ros_interface/visualization/).
After the maximum number of iterations, the optimal laptime and parameters will be printed in your terminal and the optimization will end.

The folder [visualization](./coat_mpc_ros_interface/visualization/) contains multiple scripts to visualize the performance of the algorithms. One can use the scripts to plot the cumulative regret over time as well as the posterior of the GP.

## Customization
The code can be easily customized to work with different MPCs and autonomous systems by changing the YAML files. In some cases, depending on your simulator you might need to change some lines of code.

## Notes
The code is set to use default parameters for the SafeOpt algorithm, which can be adjusted as needed in the [config/config.yaml](./coat_mpc_ros_interface/config/config.yaml) file.

## Citation
If you find our method useful for your research, please consider citing us with the following.

```
@article{puigjaner2015coatmpc,
  title={COAt-MPC: Performance-driven Constrained Optimal Auto-Tuner for MPC},
  author={Gassol Puigjaner, Albert and Prajapat, Manish and Carron, Andrea and Krause, Andreas and Zeilinger, Melanie N},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}
```

## License
This code is released under the MIT License and is free to use by anyone without any restrictions.

## Contact
For any questions or suggestions, please contact me at albertgassol1@gmail.com
