#!/usr/bin/env python3
"""
Unitree Robotics - Dex3 Hand Interface
A clean interface for controlling the Unitree Dex3 Hand

Usage:
    from dex_hand_interface import Dex3HandController
    
    hand = Dex3HandController("R", network_interface="eth0")
    hand.open_hand()
    hand.close_hand()
    hand.stop()
"""

import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

# Constants
MOTOR_MAX = 7
SENSOR_MAX = 9

# Motors List
MOTOR_DICT = {
    "thumb_rotate": 0,
    "thumb_base": 1,
    "thumb_tip": 2,
    "middle_base": 3,
    "middle_tip": 4,
    "index_base": 5,
    "index_tip": 6
}

# URDF Limits
MAX_LIMITS_LEFT = np.array([1.05, 1.05, 1.75, 0, 0, 0, 0])
MIN_LIMITS_LEFT = np.array([-1.05, -0.724, 0, -1.57, -1.75, -1.57, -1.75])
MAX_LIMITS_RIGHT = np.array([1.05, 0.742, 0, 1.57, 1.75, 1.57, 1.75])
MIN_LIMITS_RIGHT = np.array([-1.05, -1.05, -1.75, 0, 0, 0, 0])


class RIS_Mode:
    """Bitfield for motor mode"""
    def __init__(self, motor_id, status=0x01, timeout=0):
        self.id = motor_id & 0x0F
        self.status = status & 0x07
        self.timeout = timeout & 0x01
    
    def to_byte(self):
        mode = 0
        mode |= (self.id & 0x0F)
        mode |= (self.status & 0x07) << 4
        mode |= (self.timeout & 0x01) << 7
        return mode



class Dex3HandController:
    """
    Controller for Unitree Dex3 Hand
    
    Args:
        hand_id_str: "L" for left hand, "R" for right hand
        network_interface: Network interface name (e.g., "eth0")
    
    Methods:
        open_hand(): Open the hand to preset open position
        close_hand(): Close the hand to preset grip position
        stop(): Stop all motors
        send_command(positions, kp=1.5, kd=0.1): Send custom position commands
        get_hand_state(): Get current hand state
    """
    def __init__(self, hand_id_str, network_interface=None):
        self.hand_id_str = hand_id_str.upper()
        self.is_left_hand = (self.hand_id_str == "L")
        
        # Set topic names based on hand
        if self.is_left_hand:
            self.dds_namespace = "rt/dex3/left"
            self.sub_namespace = "rt/dex3/left/state"
        else:
            self.dds_namespace = "rt/dex3/right"
            self.sub_namespace = "rt/dex3/right/state"
        
        # Initialize DDS
        ChannelFactoryInitialize(0, network_interface)
        
        # Create publisher and subscriber
        self.handcmd_pub = ChannelPublisher(self.dds_namespace + "/cmd", HandCmd_)
        self.handcmd_pub.Init()
        
        self.handstate_sub = ChannelSubscriber(self.sub_namespace, HandState_)
        self.handstate_sub.Init()
        
        # State variables
        self.hand_state = None
        
        # Get limits for this hand
        if self.is_left_hand:
            self.max_limits = MAX_LIMITS_LEFT
            self.min_limits = MIN_LIMITS_LEFT
        else:
            self.max_limits = MAX_LIMITS_RIGHT
            self.min_limits = MIN_LIMITS_RIGHT
        
        print(f"Initialized {self.hand_id_str} Hand Controller")
        print(f"Command topic: {self.dds_namespace}/cmd")
        print(f"State topic: {self.sub_namespace}")
    
    def make_mode(self, motor_id, status=0x01, timeout=0):
        """Create motor mode byte"""
        ris_mode = RIS_Mode(motor_id, status, timeout)
        return ris_mode.to_byte()
    
    def build_hand_cmd(self):
        """Create a HandCmd_ message with proper initialization"""
        msg = unitree_hg_msg_dds__HandCmd_()
        return msg
    
    def close_hand(self):
        """Close the hand to preset grip position"""
        msg = self.build_hand_cmd()
        
        for i in range(MOTOR_MAX):
            mode = self.make_mode(i, status=0x01, timeout=0)
            msg.motor_cmd[i].mode = mode
            msg.motor_cmd[i].tau = 0.0
            
            # Move to closed/grip position
            if i == MOTOR_DICT["middle_base"] or i == MOTOR_DICT["index_base"]:
                # These motors close by going toward 70% of range
                close_pos = self.min_limits[i] + 0.70 * (self.max_limits[i] - self.min_limits[i])
            elif i == MOTOR_DICT["thumb_base"]:
                # Thumb closes at 30% of range
                close_pos = self.min_limits[i] + 0.30 * (self.max_limits[i] - self.min_limits[i])
            elif i == MOTOR_DICT["thumb_rotate"]:
                # Thumb rotate neutral position
                close_pos = self.min_limits[i] + 0.80 * (self.max_limits[i] - self.min_limits[i])
            else:
                # Other motors: midpoint
                close_pos = (self.max_limits[i] + self.min_limits[i]) / 2.0

            msg.motor_cmd[i].q = float(close_pos)
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].kp = 1.5
            msg.motor_cmd[i].kd = 0.1
        
        self.handcmd_pub.Write(msg)
    
    def open_hand(self):
        """Open the hand to preset open position"""
        msg = self.build_hand_cmd()
        
        for i in range(MOTOR_MAX):
            mode = self.make_mode(i, status=0x01, timeout=0)
            msg.motor_cmd[i].mode = mode
            msg.motor_cmd[i].tau = 0.0
            
            # Move to open position
            if i == MOTOR_DICT["middle_base"] or i == MOTOR_DICT["index_base"]: 
                open_pos = self.min_limits[i] + 0.20 * (self.max_limits[i] - self.min_limits[i])
            elif i == MOTOR_DICT["thumb_base"]:
                open_pos = self.min_limits[i] + 0.80 * (self.max_limits[i] - self.min_limits[i])
            elif i == MOTOR_DICT["thumb_rotate"]:
                # Thumb rotate neutral position
                open_pos = self.min_limits[i] + 0.80 * (self.max_limits[i] - self.min_limits[i])
            else:
                # Other motors: midpoint
                open_pos = (self.max_limits[i] + self.min_limits[i]) / 2.0
            
            msg.motor_cmd[i].q = float(open_pos)
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].kp = 1.5
            msg.motor_cmd[i].kd = 0.1
        
        self.handcmd_pub.Write(msg)
    
    def stop(self):
        """Stop all motors"""
        msg = self.build_hand_cmd()
        
        for i in range(MOTOR_MAX):
            mode = self.make_mode(i, status=0x01, timeout=1)
            msg.motor_cmd[i].mode = mode
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].kp = 0.0
            msg.motor_cmd[i].kd = 0.0
            msg.motor_cmd[i].q = 0.0
        
        self.handcmd_pub.Write(msg)
    
    def send_command(self, positions, kp=1.5, kd=0.1, velocities=None, torques=None):
        """
        Send custom position commands to the hand
        
        Args:
            positions: Array of 7 joint positions (radians)
            kp: Position gain (default 1.5)
            kd: Velocity gain (default 0.1)
            velocities: Array of 7 joint velocities (optional, defaults to 0)
            torques: Array of 7 joint torques (optional, defaults to 0)
        """
        msg = self.build_hand_cmd()
        
        if velocities is None:
            velocities = np.zeros(MOTOR_MAX)
        if torques is None:
            torques = np.zeros(MOTOR_MAX)
        
        for i in range(MOTOR_MAX):
            mode = self.make_mode(i, status=0x01, timeout=0)
            msg.motor_cmd[i].mode = mode
            msg.motor_cmd[i].q = float(positions[i])
            msg.motor_cmd[i].dq = float(velocities[i])
            msg.motor_cmd[i].tau = float(torques[i])
            msg.motor_cmd[i].kp = float(kp)
            msg.motor_cmd[i].kd = float(kd)
        
        self.handcmd_pub.Write(msg)
    
    def get_hand_state(self):
        """
        Get current hand state
        
        Returns:
            dict: Hand state with normalized joint positions [0, 1] and raw state
        """
        # Read latest state
        state_msg = self.handstate_sub.Read(timeout=0.1)
        
        if state_msg is not None:
            self.hand_state = state_msg
        
        if self.hand_state is None:
            return None
        
        # Normalize joint positions to [0, 1]
        q_normalized = np.zeros(7)
        q_raw = np.zeros(7)
        
        for i in range(min(7, len(self.hand_state.motor_state))):
            q_raw[i] = self.hand_state.motor_state[i].q
            q_normalized[i] = (q_raw[i] - self.min_limits[i]) / (self.max_limits[i] - self.min_limits[i])
            q_normalized[i] = np.clip(q_normalized[i], 0.0, 1.0)
        
        return {
            'positions_normalized': q_normalized,
            'positions_raw': q_raw,
            'state': self.hand_state
        }


def main():
    """Example usage of the Dex3HandController"""
    print("=== Unitree Robotics Dex3 Hand Interface ===\n")
    
    hand_id = input("Enter hand id (L for left, R for right): ").strip().upper()
    
    if hand_id not in ["L", "R"]:
        print("Invalid hand id. Please input 'L' or 'R'.")
        return -1
    
    network_interface = None
    if len(sys.argv) >= 2:
        network_interface = sys.argv[1]
        print(f"Using network interface: {network_interface}")
    
    try:
        # Create controller
        hand = Dex3HandController(hand_id, network_interface)
        
        print("\nExample commands:")
        print("  Opening hand...")
        hand.open_hand()
        time.sleep(2)
        
        print("  Closing hand...")
        hand.close_hand()
        time.sleep(2)
        
        print("  Opening hand again...")
        hand.open_hand()
        time.sleep(2)
        
        print("  Stopping motors...")
        hand.stop()
        
        print("\nDemo complete!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
