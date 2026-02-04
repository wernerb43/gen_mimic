#!/usr/bin/env python3
"""
Unitree Robotics - Dex3 Hand Example (Python version)
Translated from dex3_hand._test.cpp

States: INIT, ROTATE, GRIP, OPEN, STOP, PRINT
Commands:
  r - Rotate motors through sinusoidal motion
  g - Grip hand (move to mid-position)
  o - Open hand (move to 3/4 max limit)
  p - Print hand state
  s - Stop motors
  q - Quit
"""

import sys
import time
import math
import threading
import tty
import termios
import numpy as np
from enum import Enum
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


class State(Enum):
    INIT = 0
    ROTATE = 1
    GRIP = 2
    STOP = 3
    PRINT = 4
    OPEN = 5


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
        self.current_state = State.INIT
        self.state_lock = threading.Lock()
        self.hand_state = None
        self.running = True
        
        # Rotation counter for ROTATE state
        self.rotate_count = 1
        self.rotate_dir = 1
        
        # Get limits for this hand
        if self.is_left_hand:
            self.max_limits = MAX_LIMITS_LEFT
            self.min_limits = MIN_LIMITS_LEFT
        else:
            self.max_limits = MAX_LIMITS_RIGHT
            self.min_limits = MIN_LIMITS_RIGHT
        
        print(f"\n--- Initialized {self.hand_id_str} Hand Controller ---")
        print(f"Command topic: {self.dds_namespace}/cmd")
        print(f"State topic: {self.sub_namespace}")
    
    def make_mode(self, motor_id, status=0x01, timeout=0):
        """Create motor mode byte"""
        ris_mode = RIS_Mode(motor_id, status, timeout)
        return ris_mode.to_byte()
    
    def build_hand_cmd(self):
        """Create a HandCmd_ message with proper initialization"""
        msg = unitree_hg_msg_dds__HandCmd_()
        # Ensure motor_cmd has 7 entries
        if len(msg.motor_cmd) < MOTOR_MAX:
            # This shouldn't happen but handle it anyway
            pass
        return msg
    
    def rotate_motors(self):
        """Send sinusoidal position commands to motors"""
        msg = self.build_hand_cmd()
        
        for i in range(MOTOR_MAX):
            mode = self.make_mode(i, status=0x01, timeout=0)
            msg.motor_cmd[i].mode = mode
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = 0.5
            msg.motor_cmd[i].kd = 0.1
            
            # Calculate sinusoidal position within limits
            range_val = self.max_limits[i] - self.min_limits[i]
            mid = (self.max_limits[i] + self.min_limits[i]) / 2.0
            amplitude = range_val / 2.0
            q = mid + amplitude * math.sin(self.rotate_count / 20000.0 * math.pi)
            
            msg.motor_cmd[i].q = float(q)
            msg.motor_cmd[i].dq = 0.0
        
        self.handcmd_pub.Write(msg)
        
        # Update counter
        self.rotate_count += self.rotate_dir
        if self.rotate_count >= 10000:
            self.rotate_dir = -1
        if self.rotate_count <= -10000:
            self.rotate_dir = 1
        
        time.sleep(0.0001)  # 100 microseconds
    
    def grip_hand(self):
        """Send static mid-position commands to grip the hand"""
        msg = self.build_hand_cmd()
        
        for i in range(MOTOR_MAX):
            mode = self.make_mode(i, status=0x01, timeout=0)
            msg.motor_cmd[i].mode = mode
            msg.motor_cmd[i].tau = 0.0
            
            # Move to mid position
            if i == MOTOR_DICT["middle_base"] or i == MOTOR_DICT["index_base"]:
                # These motors close by going toward min, open by going toward 20% of range
                close_pos = MIN_LIMITS_RIGHT[i] + 0.70 * (MAX_LIMITS_RIGHT[i] - MIN_LIMITS_RIGHT[i])
            elif i == MOTOR_DICT["thumb_base"]:
                # Thumb opens by going toward 80% of range
                close_pos = MIN_LIMITS_RIGHT[i] + 0.30 * (MAX_LIMITS_RIGHT[i] - MIN_LIMITS_RIGHT[i])
            elif i == MOTOR_DICT["thumb_rotate"]:
                # Thumb rotate neutral position
                close_pos = MIN_LIMITS_RIGHT[i] + 0.80 * (MAX_LIMITS_RIGHT[i] - MIN_LIMITS_RIGHT[i])  # Neutral position
            else:
                # Other motors: midpoint for open, can adjust if needed
                close_pos = (MAX_LIMITS_RIGHT[i] + MIN_LIMITS_RIGHT[i]) / 2.0

            msg.motor_cmd[i].q = float(close_pos)
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].kp = 1.5
            msg.motor_cmd[i].kd = 0.1
        
        self.handcmd_pub.Write(msg)
        time.sleep(1.0)
    
    def open_hand(self):
        """Send commands to open the hand"""
        msg = self.build_hand_cmd()
        
        for i in range(MOTOR_MAX):
            mode = self.make_mode(i, status=0x01, timeout=0)
            msg.motor_cmd[i].mode = mode
            msg.motor_cmd[i].tau = 0.0
            
            # Move to 3/4 position (75% toward max limit)
            if i == MOTOR_DICT["middle_base"] or i == MOTOR_DICT["index_base"]: 
                open_pos = MIN_LIMITS_RIGHT[i] + 0.20 * (MAX_LIMITS_RIGHT[i] - MIN_LIMITS_RIGHT[i])
            elif i == MOTOR_DICT["thumb_base"]:
                open_pos = MIN_LIMITS_RIGHT[i] + 0.80 * (MAX_LIMITS_RIGHT[i] - MIN_LIMITS_RIGHT[i])
            elif i == MOTOR_DICT["thumb_rotate"]:
                # Thumb rotate neutral position
                open_pos = MIN_LIMITS_RIGHT[i] + 0.80 * (MAX_LIMITS_RIGHT[i] - MIN_LIMITS_RIGHT[i])  # Neutral position
            else:
                # Other motors: midpoint for open, can adjust if needed
                open_pos = (MAX_LIMITS_RIGHT[i] + MIN_LIMITS_RIGHT[i]) / 2.0
            
            msg.motor_cmd[i].q = float(open_pos)
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].kp = 1.5
            msg.motor_cmd[i].kd = 0.1
        
        self.handcmd_pub.Write(msg)
        time.sleep(1.0)
    
    def stop_motors(self):
        """Send zero commands to stop motors"""
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
        time.sleep(1.0)
    
    def print_state(self):
        """Read and print normalized hand state"""
        # Read latest state
        state_msg = self.handstate_sub.Read(timeout=0.1)
        
        if state_msg is not None:
            self.hand_state = state_msg
        
        if self.hand_state is None:
            print("No hand state received yet...")
            time.sleep(0.1)
            return
        
        # Normalize joint positions to [0, 1]
        q = np.zeros(7)
        for i in range(min(7, len(self.hand_state.motor_state))):
            q_raw = self.hand_state.motor_state[i].q
            q[i] = (q_raw - self.min_limits[i]) / (self.max_limits[i] - self.min_limits[i])
            q[i] = np.clip(q[i], 0.0, 1.0)
        
        # Clear screen and print
        print("\033[2J\033[H")  # Clear screen
        print("-- Hand State --")
        print(f"--- Current State: {self.current_state.name} ---")
        print("Commands:")
        print("  r - Rotate")
        print("  g - Grip")
        print("  o - Open")
        print("  p - Print state")
        print("  s - Stop")
        print("  q - Quit")
        
        hand_label = "L" if self.is_left_hand else "R"
        print(f" {hand_label}: {q}")
        
        time.sleep(0.1)
    
    def get_non_blocking_input(self):
        """Read single character without blocking (Unix only)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def user_input_thread(self):
        """Thread to monitor keyboard input"""
        while self.running:
            try:
                ch = self.get_non_blocking_input()
                
                with self.state_lock:
                    if ch == 'q':
                        print("\nExiting...")
                        self.current_state = State.STOP
                        self.running = False
                        break
                    elif ch == 'r':
                        self.current_state = State.ROTATE
                    elif ch == 'g':
                        self.current_state = State.GRIP
                    elif ch == 'o':
                        self.current_state = State.OPEN
                    elif ch == 'p':
                        self.current_state = State.PRINT
                    elif ch == 's':
                        self.current_state = State.STOP
            except:
                pass
            
            time.sleep(0.1)
    
    def run(self):
        """Main control loop"""
        # Start input thread
        input_thread = threading.Thread(target=self.user_input_thread, daemon=True)
        input_thread.start()
        
        last_state = None
        
        while self.running:
            with self.state_lock:
                state = self.current_state
            
            # Print state change
            if state != last_state:
                print(f"\n--- Current State: {state.name} ---")
                print("Commands:")
                print("  r - Rotate")
                print("  g - Grip")
                print("  o - Open")
                print("  p - Print state")
                print("  s - Stop")
                print("  q - Quit")
                last_state = state
            
            # Execute state
            if state == State.INIT:
                print("Initializing...")
                with self.state_lock:
                    self.current_state = State.ROTATE
            elif state == State.ROTATE:
                self.rotate_motors()
            elif state == State.GRIP:
                self.grip_hand()
            elif state == State.OPEN:
                self.open_hand()
            elif state == State.STOP:
                self.stop_motors()
            elif state == State.PRINT:
                self.print_state()
        
        # Cleanup
        input_thread.join(timeout=1.0)
        print("Controller stopped.")


def main():
    print(" --- Unitree Robotics --- ")
    print("     Dex3 Hand Example (Python)      \n")
    
    hand_id = input("Please input the hand id (L for left hand, R for right hand): ").strip().upper()
    
    if hand_id not in ["L", "R"]:
        print("Invalid hand id. Please input 'L' or 'R'.")
        return -1
    
    network_interface = None
    if len(sys.argv) >= 2:
        network_interface = sys.argv[1]
        print(f"Using network interface: {network_interface}")
    
    try:
        controller = Dex3HandController(hand_id, network_interface)
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
