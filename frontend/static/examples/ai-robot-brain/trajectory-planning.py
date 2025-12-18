#!/usr/bin/env python3
# Nav2 Trajectory Planning for Humanoid Robots

"""
This script demonstrates trajectory planning for humanoid robots using Nav2
"""
import math
import numpy as np

def plan_humanoid_trajectory(start_pose, goal_pose, obstacles=None):
    """
    Plans a trajectory for a humanoid robot from start_pose to goal_pose,
    accounting for bipedal locomotion constraints.
    
    Args:
        start_pose: Tuple (x, y, theta) representing starting position and orientation
        goal_pose: Tuple (x, y, theta) representing goal position and orientation
        obstacles: List of obstacles (optional)
    
    Returns:
        trajectory: List of waypoints for the humanoid robot
    """
    print("Planning trajectory for humanoid robot...")
    
    # Extract position components
    start_x, start_y, start_theta = start_pose
    goal_x, goal_y, goal_theta = goal_pose
    
    # Calculate distance to goal
    dist = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
    
    # For a humanoid robot, we need to consider step constraints
    # Humanoid robots have specific step size limitations
    max_step_size = 0.3  # meters
    
    # Generate waypoints based on step size constraints
    num_waypoints = max(int(dist / max_step_size), 5)  # At least 5 waypoints
    x_waypoints = np.linspace(start_x, goal_x, num_waypoints)
    y_waypoints = np.linspace(start_y, goal_y, num_waypoints)
    
    # Calculate orientations for each waypoint
    # Humanoid robots need to face their direction of motion
    orientations = []
    for i in range(len(x_waypoints)-1):
        dx = x_waypoints[i+1] - x_waypoints[i]
        dy = y_waypoints[i+1] - y_waypoints[i]
        orient = math.atan2(dy, dx)
        orientations.append(orient)
    # Add final orientation
    orientations.append(goal_theta)
    
    # Create trajectory - for humanoid, each point needs to be a stable position
    trajectory = []
    for i in range(len(x_waypoints)):
        waypoint = {
            'position': (x_waypoints[i], y_waypoints[i]),
            'orientation': orientations[i] if i < len(orientations) else orientations[-1],
            'velocity': 0.0,  # Will be calculated based on humanoid capabilities
            'is_stable': True,  # Placeholder for stability consideration
            'foot_placement': calculate_foot_placement((x_waypoints[i], y_waypoints[i]), 
                                                     orientations[i] if i < len(orientations) else orientations[-1])
        }
        trajectory.append(waypoint)
    
    print(f"✓ Planned trajectory with {len(trajectory)} waypoints")
    
    # Apply humanoid-specific constraints
    trajectory = apply_bipedal_constraints(trajectory)
    print("✓ Applied bipedal locomotion constraints")
    
    return trajectory

def calculate_foot_placement(position, orientation):
    """
    Calculates appropriate foot placements for a humanoid robot
    """
    # Simplified model - in real implementation this would be more complex
    base_x, base_y = position
    
    # Calculate left and right foot positions relative to robot base
    foot_separation = 0.2  # Distance between feet (hip width)
    step_length = 0.15     # Forward step length
    
    # Calculate positions based on orientation
    cos_th = math.cos(orientation)
    sin_th = math.sin(orientation)
    
    # Left foot slightly left and forward
    left_foot_x = base_x + (-foot_separation/2) * sin_th + step_length/2 * cos_th
    left_foot_y = base_y + (foot_separation/2) * cos_th + step_length/2 * sin_th
    
    # Right foot slightly right and forward
    right_foot_x = base_x + (foot_separation/2) * sin_th + step_length/2 * cos_th
    right_foot_y = base_y + (-foot_separation/2) * cos_th + step_length/2 * sin_th
    
    return {
        'left_foot': (left_foot_x, left_foot_y),
        'right_foot': (right_foot_x, right_foot_y),
        'orientation': orientation
    }

def apply_bipedal_constraints(trajectory):
    """
    Applies humanoid robot specific constraints to the trajectory
    """
    # Check for stability at each point
    for i, waypoint in enumerate(trajectory):
        # Ensure the robot can maintain balance
        waypoint['is_stable'] = True  # Simplified stability check
        
        # Adjust velocities based on humanoid capabilities
        if i < len(trajectory) - 1:
            next_waypoint = trajectory[i+1]
            dx = next_waypoint['position'][0] - waypoint['position'][0]
            dy = next_waypoint['position'][1] - waypoint['position'][1]
            dist_to_next = math.sqrt(dx*dx + dy*dy)
            
            # Humanoid max speed for safe locomotion
            max_speed = 0.5  # m/s - conservative for stability
            waypoint['velocity'] = min(max_speed, dist_to_next * 1.0)  # Simple velocity calculation
    
    return trajectory

def validate_trajectory_for_humanoid(trajectory):
    """
    Validates that the trajectory is suitable for a humanoid robot
    """
    print("Validating trajectory for humanoid robot constraints...")
    
    valid = True
    for i, waypoint in enumerate(trajectory):
        # Check if step sizes are within humanoid limits
        if i > 0:
            prev_waypoint = trajectory[i-1]
            dx = waypoint['position'][0] - prev_waypoint['position'][0]
            dy = waypoint['position'][1] - prev_waypoint['position'][1]
            step_size = math.sqrt(dx*dx + dy*dy)
            
            if step_size > 0.4:  # Max humanoid step size
                print(f"⚠ Waypoint {i} exceeds max step size: {step_size:.2f}m")
                valid = False
                
        # Check stability
        if not waypoint.get('is_stable', False):
            print(f"⚠ Waypoint {i} is not stable for humanoid locomotion")
            valid = False
    
    if valid:
        print("✓ Trajectory validation passed")
    else:
        print("✗ Trajectory validation failed")
    
    return valid

if __name__ == "__main__":
    # Example usage
    start = (0.0, 0.0, 0.0)  # x, y, theta
    goal = (3.0, 2.0, math.pi/4)  # x, y, theta
    obstacles = []  # No obstacles for this example

    # Plan the trajectory
    traj = plan_humanoid_trajectory(start, goal, obstacles)

    # Validate the trajectory
    is_valid = validate_trajectory_for_humanoid(traj)

    if is_valid:
        print(f"\nTrajectory planning successful with {len(traj)} waypoints:")
        for i, wp in enumerate(traj[:5]):  # Show first 5 waypoints
            pos = wp['position']
            vel = wp.get('velocity', 0.0)
            print(f"  Waypoint {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}), vel={vel:.2f}m/s")
        if len(traj) > 5:
            print(f"  ... and {len(traj)-5} more waypoints")
    else:
        print("\nTrajectory planning failed validation")