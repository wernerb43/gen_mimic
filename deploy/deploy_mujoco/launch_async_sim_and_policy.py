# Copyright (c) 2025 Lizhi Yang AMBER LAB

import argparse
import os
import subprocess

import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch MujocoSimNode and RLPolicyNode with a given config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file name (e.g., g1_21dof.yaml)",
    )
    parser.add_argument(
        "--dof",
        type=int,
        required=True,
        help="Number of degrees of freedom for policy (13/29)",
    )
    args = parser.parse_args()

    # Set environment variable if not already set
    if "G1_RL_ROOT_DIR" not in os.environ:
        os.environ["G1_RL_ROOT_DIR"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
    # read the config
    config_file = args.config
    with open(f"{G1_RL_ROOT_DIR}/deploy/configs/{config_file}") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    plot_feet = config.get("plot_feet", False)

    sim_cmd = [
        "python",
        os.path.join(os.path.dirname(__file__), "mujoco_sim_node.py"),
        "--config",
        args.config,
    ]
    policy_cmd_29 = [
        "python",
        os.path.join(os.path.dirname(__file__), "rl_policy_node.py"),
        "--config",
        args.config,
    ]
    policy_cmd_13 = [
        "python",
        os.path.join(os.path.dirname(__file__), "rl_policy_node_13dof.py"),
        "--config",
        args.config,
    ]
    if args.dof == 29:
        policy_cmd = policy_cmd_29
    elif args.dof == 13:
        policy_cmd = policy_cmd_13
    else:
        raise ValueError("Invalid DOF value. Must be 13 or 29.")
    if plot_feet:
        plotter_cmd = [
            "python",
            os.path.join(os.path.dirname(__file__), "foot_force_plotter.py"),
        ]

    # Launch both processes
    sim_proc = subprocess.Popen(sim_cmd, env=os.environ)
    policy_proc = subprocess.Popen(policy_cmd, env=os.environ)
    if plot_feet:
        plotter_proc = subprocess.Popen(plotter_cmd, env=os.environ)

    try:
        sim_proc.wait()
        policy_proc.wait()
        if plot_feet:
            plotter_proc.wait()
    except KeyboardInterrupt:
        sim_proc.terminate()
        policy_proc.terminate()
        if plot_feet:
            plotter_proc.terminate()
        sim_proc.wait()
        policy_proc.wait()
        if plot_feet:
            plotter_proc.wait()


if __name__ == "__main__":
    main()
