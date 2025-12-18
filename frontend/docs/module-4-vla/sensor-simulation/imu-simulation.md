# IMU Simulation in Digital Twins

## Introduction

This chapter covers simulating Inertial Measurement Units (IMUs) in digital twin environments. IMUs are crucial sensors for humanoid robots that provide information about the robot's acceleration, angular velocity, and orientation. In digital twins, accurate IMU simulation is essential for developing perception, control, and navigation systems that will work effectively on physical robots.

## IMU Technology Overview

### What is an IMU

An Inertial Measurement Unit (IMU) is a device that measures and reports a robot's specific force, angular rate, and sometimes magnetic field. A typical IMU consists of:

1. **Accelerometer**: Measures linear acceleration in three axes (x, y, z)
2. **Gyroscope**: Measures angular velocity around three axes (roll, pitch, yaw) 
3. **Magnetometer** (optional): Measures magnetic field strength and direction (provides heading reference)

### IMU Types and Specifications

In robotics applications, IMUs vary by quality and capability:

- **MEMS-based**: Small, low-cost, typical for consumer robotics
- **Fiber Optic Gyroscopes**: High accuracy, expensive, for precision applications
- **Ring Laser Gyroscopes**: Very high accuracy, mainly in aerospace applications

For humanoid robots, MEMS-based IMUs are typically used due to their compact size and acceptable performance for most applications.

## IMU Simulation in Gazebo

### Understanding IMU Physics

IMU simulation in Gazebo relies on the robot's dynamics to generate realistic measurements. The simulator calculates acceleration and angular velocity based on the robot's motion, then adds noise and biases to match real sensor behavior.

### Basic IMU Configuration

Here's an example of configuring an IMU sensor for a humanoid robot in Gazebo:

```xml
<!-- IMU sensor on the humanoid robot's torso -->
<gazebo reference="torso">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>  <!-- Update at 100 Hz -->
    <pose>0 0 0 0 0 0</pose>  <!-- Position relative to link -->
    
    <imu>
      <!-- Accelerometer parameters -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>  <!-- 0.2 mrad/s standard deviation -->
            <bias_mean>0.00001</bias_mean>
            <bias_stddev>0.000001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.00001</bias_mean>
            <bias_stddev>0.000001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.00001</bias_mean>
            <bias_stddev>0.000001</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      
      <!-- Gyroscope parameters -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>  <!-- 17 mg standard deviation -->
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    
    <!-- Gazebo ROS plugin for IMU -->
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>torso</body_name>
      <update_rate>100</update_rate>
      <gaussian_noise>0.017</gaussian_noise>  <!-- Used for all axes -->
    </plugin>
  </sensor>
</gazebo>
```

### Advanced IMU Configuration

For more realistic simulation, you can add additional parameters that match specific IMU models:

```xml
<!-- Advanced IMU with realistic parameters matching a popular IMU model -->
<gazebo reference="humanoid_head">
  <sensor name="head_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>  <!-- Higher update rate for head IMU -->
    <pose>0.1 0.0 0.1 0 0 0</pose>  <!-- Offset to approximate head center -->
    
    <imu>
      <!-- Angular velocity parameters (gyroscope) -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>  <!-- Typical for high-quality MEMS gyro (0.057 deg/s) -->
            <bias_mean>0.001</bias_mean>  <!-- Small bias in the sensor reading -->
            <bias_stddev>0.0001</bias_stddev>
            <!-- Dynamic bias that changes over time -->
            <dynamic_bias_correlation_time>100</dynamic_bias_correlation_time>
            <dynamic_bias_stddev>0.0001</dynamic_bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.001</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
            <dynamic_bias_correlation_time>100</dynamic_bias_correlation_time>
            <dynamic_bias_stddev>0.0001</dynamic_bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.001</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
            <dynamic_bias_correlation_time>100</dynamic_bias_correlation_time>
            <dynamic_bias_stddev>0.0001</dynamic_bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      
      <!-- Linear acceleration parameters (accelerometer) -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>  <!-- 17 mg typical for MEMS accelerometer -->
            <bias_mean>0.01</bias_mean>  <!-- Small offset in measurements -->
            <bias_stddev>0.001</bias_stddev>
            <dynamic_bias_correlation_time>300</dynamic_bias_correlation_time>
            <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.01</bias_mean>
            <bias_stddev>0.001</bias_stddev>
            <dynamic_bias_correlation_time>300</dynamic_bias_correlation_time>
            <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.01</bias_mean>
            <bias_stddev>0.001</bias_stddev>
            <dynamic_bias_correlation_time>300</dynamic_bias_correlation_time>
            <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    
    <plugin name="head_imu_controller" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/humanoid_robot/head</namespace>
        <remapping>~/out:=imu_data</remapping>
      </ros>
      <frame_name>head_imu_frame</frame_name>
      <body_name>head</body_name>
      <update_rate>200</update_rate>
      <gaussian_noise>0.017</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## Noise Modeling and Realistic Simulation

### Understanding IMU Noise Characteristics

Real IMUs have complex noise characteristics that need to be accurately modeled in simulation:

1. **White Noise**: Random noise at each measurement
2. **Bias**: Consistent offset from true values
3. **Drift**: Slowly changing bias over time
4. **Scale Factor Error**: Consistent multiplicative error
5. **Non-Orthogonality**: Misalignment between sensor axes

## Multiple IMU Placement

Humanoid robots often have multiple IMUs to measure orientation at different points:

```xml
<!-- Torso IMU for balance control -->
<gazebo reference="torso">
  <sensor name="torso_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <pose>0 0 0 0 0 0</pose>
    <!-- Configuration similar to above -->
  </sensor>
</gazebo>

<!-- Head IMU for gaze control -->
<gazebo reference="head">
  <sensor name="head_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>
    <pose>0.05 0 0 0 0 0</pose>
    <!-- Configuration similar to above but higher rate -->
  </sensor>
</gazebo>

<!-- Foot IMUs for gait control -->
<gazebo reference="left_foot">
  <sensor name="left_foot_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <pose>0 0 0 0 0 0</pose>
  </sensor>
</gazebo>

<gazebo reference="right_foot">
  <sensor name="right_foot_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <pose>0 0 0 0 0 0</pose>
  </sensor>
</gazebo>
```

### Advanced Noise Modeling

For more realistic simulation, include temperature effects and aging:

```xml
<!-- IMU with temperature dependent noise -->
<gazebo reference="torso">
  <sensor name="temp_dependent_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <pose>0 0 0 0 0 0</pose>
    
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.00001</bias_mean>
            <bias_stddev>0.000001</bias_stddev>
            <!-- Temperature dependent parameters -->
            <temperature_coefficient>0.00001</temperature_coefficient>  <!-- Change per degree Celsius -->
            <reference_temperature>25.0</reference_temperature>  <!-- Reference temperature in Celsius -->
            <temperature_variance>5.0</temperature_variance>  <!-- Temperature variation in simulation -->
          </noise>
        </x>
        <!-- Similar configuration for y and z axes -->
      </angular_velocity>
      
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.05</bias_mean>
            <bias_stddev>0.001</bias_stddev>
            <!-- Temperature dependent parameters -->
            <temperature_coefficient>0.0001</temperature_coefficient>
            <reference_temperature>25.0</reference_temperature>
            <temperature_variance>5.0</temperature_variance>
          </noise>
        </x>
        <!-- Similar configuration for y and z axes -->
      </linear_acceleration>
    </imu>
    
    <!-- Specialized plugin for temperature modeling -->
    <plugin name="temp_imu_plugin" filename="libgazebo_ros_temp_imu.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=imu/data_temp_compensated</remapping>
      </ros>
      <frame_name>temp_imu_link</frame_name>
      <body_name>torso</body_name>
      <update_rate>100</update_rate>
      <gaussian_noise>0.017</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## Processing IMU Data in Simulation

### Reading IMU Data in ROS 2

Here's how to read and process IMU data in a ROS 2 system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
import tf_transformations

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')
        
        # Subscribe to IMU data
        self.subscription = self.create_subscription(
            Imu,
            '/humanoid_robot/imu/data',
            self.imu_callback,
            10)
        
        # Publisher for processed orientation data
        self.orientation_publisher = self.create_publisher(Float64, '/humanoid_robot/orientation_estimate', 10)
        
        # Store state for filtering
        self.previous_orientation = None
        self.previous_time = None
        self.orientation_velocity = Vector3()
        
        # Initialize filter parameters
        self.filter_alpha = 0.98  # For complementary filter
        self.gravity_reference = np.array([0, 0, 9.81])  # Gravitational acceleration vector
        
        self.get_logger().info('IMU Processor initialized')
    
    def imu_callback(self, msg):
        """
        Process IMU data from simulation
        """
        # Extract measurements
        angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        # Get current timestamp
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        
        # If this is the first message, initialize
        if self.previous_time is None:
            self.previous_time = current_time
            # Initialize orientation from acceleration if available
            if np.linalg.norm(linear_acceleration) > 0.1:  # Check for reasonable acceleration reading
                self.previous_orientation = self.acceleration_to_orientation(linear_acceleration)
            else:
                self.previous_orientation = R.from_quat([0, 0, 0, 1]).as_quat()  # Identity quaternion
            return
        
        # Calculate time difference
        dt = current_time - self.previous_time
        self.previous_time = current_time
        
        # Skip if time difference is too small
        if dt == 0:
            return
        
        # Update orientation using sensor fusion
        estimated_orientation = self.fuse_sensor_data(
            angular_velocity, 
            linear_acceleration, 
            dt
        )
        
        # Publish processed data
        self.publish_processed_data(estimated_orientation, angular_velocity)
    
    def acceleration_to_orientation(self, acceleration):
        """
        Estimate orientation from acceleration vector (assuming robot is stationary)
        """
        # Normalize acceleration vector
        acc_norm = acceleration / np.linalg.norm(acceleration)
        
        # Calculate roll and pitch from accelerometer
        pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
        roll = np.arctan2(acc_norm[1], acc_norm[2])
        
        # Create quaternion from roll and pitch (ignoring yaw from accelerometer)
        quaternion = R.from_euler('xyz', [roll, pitch, 0]).as_quat()
        
        return quaternion
    
    def fuse_sensor_data(self, angular_velocity, linear_acceleration, dt):
        """
        Fuse gyroscope and accelerometer data using complementary filter
        """
        # Method 1: Integrate gyroscope readings (good for short-term accuracy)
        if self.previous_orientation is not None:
            # Convert to rotation object
            current_rotation = R.from_quat(self.previous_orientation)
            
            # Create rotation from angular velocity (small angle approximation)
            angle_magnitude = np.linalg.norm(angular_velocity) * dt
            if angle_magnitude > 1e-6:  # Avoid division by zero
                rotation_axis = angular_velocity / np.linalg.norm(angular_velocity)
                # Create rotation vector
                rotation_vector = rotation_axis * angle_magnitude
                # Create small rotation
                small_rotation = R.from_rotvec(rotation_vector)
                
                # Update rotation estimate
                integrated_rotation = small_rotation * current_rotation
                integrated_quat = integrated_rotation.as_quat()
            else:
                integrated_quat = self.previous_orientation
        else:
            integrated_quat = self.acceleration_to_orientation(linear_acceleration)
        
        # Method 2: Use accelerometer for reference (good for long-term stability)
        accel_quat = self.acceleration_to_orientation(linear_acceleration)
        
        # Apply complementary filter
        # High weight to gyroscope for fast changes, low weight to accelerometer for drift correction
        fused_quat = self.complementary_filter(
            integrated_quat, 
            accel_quat, 
            self.filter_alpha
        )
        
        # Normalize to prevent drift
        fused_quat = fused_quat / np.linalg.norm(fused_quat)
        
        # Store for next iteration
        self.previous_orientation = fused_quat
        
        return fused_quat
    
    def complementary_filter(self, quat_gyro, quat_accel, alpha):
        """
        Apply complementary filter to combine gyroscope and accelerometer estimates
        """
        # Convert to rotation objects for proper interpolation
        rot_gyro = R.from_quat(quat_gyro)
        rot_accel = R.from_quat(quat_accel)
        
        # Perform spherical linear interpolation based on alpha
        # For Quaternions, we can do a weighted combination
        # Convert alpha weighting to appropriate representation
        # For simplicity, use linear interpolation and renormalize
        combined_quat = alpha * quat_gyro + (1 - alpha) * quat_accel
        return combined_quat / np.linalg.norm(combined_quat)
    
    def publish_processed_data(self, orientation, angular_velocity):
        """
        Publish processed IMU data
        """
        # Publish orientation estimate
        orientation_msg = Float64()
        # For this example, publish just the z-component of orientation
        # In real implementations, you might publish full orientation
        if self.previous_orientation is not None:
            orientation_msg.data = self.euler_from_quaternion(self.previous_orientation)[2]  # Yaw
            self.orientation_publisher.publish(orientation_msg)
    
    def euler_from_quaternion(self, quaternion):
        """
        Convert quaternion to Euler angles
        """
        x, y, z, w = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)
```

### Advanced IMU Processing

More sophisticated processing using Extended Kalman Filters:

```python
import numpy as np
from scipy.linalg import block_diag

class IMUEKFProcessor:
    def __init__(self):
        # State vector: [qw, qx, qy, qz, wx, wy, wz, bx, by, bz]
        # (quaternion, angular velocity, gyroscope biases)
        self.state_dim = 10
        self.state = np.zeros(self.state_dim)
        self.state[0] = 1.0  # Initialize with identity quaternion [1, 0, 0, 0]
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 0.1  # Initial uncertainty
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(6) * 0.1  # [angular_velocity, linear_acceleration]
        
        # Previous time stamp
        self.prev_time = None
        
        # Gravity vector in world frame
        self.gravity = np.array([0, 0, 9.81])
    
    def predict(self, dt, control_input=None):
        """
        Prediction step of EKF
        """
        # Extract state components
        q = self.state[0:4]  # quaternion (w, x, y, z)
        omega = self.state[4:7]  # angular velocity
        b = self.state[7:10]  # gyroscope biases
        
        # Predict quaternion derivative
        omega_true = omega - b  # True angular velocity (bias removed)
        q_dot = self.quaternion_derivative(q, omega_true)
        
        # Update state
        new_state = self.state.copy()
        new_state[0:4] = q + q_dot * dt
        # Assume angular velocity and bias remain relatively constant in short time
        # For more accurate prediction, would use actual control inputs
        new_state[4:7] = omega
        new_state[7:10] = b
        
        # Normalize quaternion to prevent drift
        quat_norm = np.linalg.norm(new_state[0:4])
        if quat_norm > 0:
            new_state[0:4] /= quat_norm
        
        self.state = new_state
        
        # Predict covariance
        F = self.compute_jacobian_F(dt)
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement):
        """
        Update step of EKF using IMU measurements
        """
        # Measurement vector: [angular_velocity, linear_acceleration]
        # Expected measurement from current state
        expected_measurement = self.h(self.state)
        
        # Innovation
        innovation = measurement - expected_measurement
        
        # Jacobian of measurement function
        H = self.compute_jacobian_H()
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ innovation
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def compute_jacobian_F(self, dt):
        """
        Compute Jacobian of process model
        """
        F = np.eye(self.state_dim)
        
        # For the quaternion part, the derivative depends on current angular velocity
        omega_true = self.state[4:7] - self.state[7:10]  # True angular velocity
        Omega_matrix = self.skew_symmetric_omega(omega_true)
        
        # Jacobian of quaternion update with respect to quaternion
        F[0:4, 0:4] = np.eye(4) + Omega_matrix * dt
        
        # Jacobian of quaternion update with respect to angular velocity
        q = self.state[0:4]
        F[0:4, 4:7] = self.quaternion_omega_jacobian(q) * dt
        
        return F
    
    def compute_jacobian_H(self):
        """
        Compute Jacobian of measurement function
        """
        H = np.zeros((6, self.state_dim))
        
        # We can measure angular velocity directly (minus bias)
        H[0:3, 4:7] = np.eye(3)  # Angular velocity measurements
        H[0:3, 7:10] = -np.eye(3)  # Subtract bias
        
        # Acceleration measurements depend on orientation
        # This is more complex to compute accurately
        # For simplicity, we'll use a basic approximation
        # In reality, this would involve more complex derivatives
        q = self.state[0:4]
        R = self.quaternion_to_rotation_matrix(q)
        gravity_world = R.T @ self.gravity  # Transform gravity to body frame
        
        # The Jacobian of acceleration with respect to orientation
        # would involve complex derivatives of the rotation matrix
        H[3:6, 0:4] = self.acceleration_orientation_jacobian(q)
        
        return H
    
    def quaternion_derivative(self, q, omega):
        """
        Compute quaternion derivative from angular velocity
        """
        # Angular velocity vector (wx, wy, wz)
        # Quaternion [qw, qx, qy, qz]
        qw, qx, qy, qz = q
        wx, wy, wz = omega
        
        # Quaternion derivative
        q_dot = 0.5 * np.array([
            -wx*qx - wy*qy - wz*qz,
             wx*qw + wz*qy - wy*qz,
             wy*qw - wz*qx + wx*qz,
             wz*qw + wy*qx - wx*qy
        ])
        
        return q_dot
    
    def skew_symmetric_omega(self, omega):
        """
        Create skew-symmetric matrix from angular velocity vector
        """
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        
        # Create 4x4 matrix for quaternion derivative
        Omega_4x4 = np.zeros((4, 4))
        # The derivative matrix for quaternions
        Omega_4x4[0, 1:] = -omega
        Omega_4x4[1:, 0] = omega
        Omega_4x4[1:, 1:] = Omega
        
        return 0.5 * Omega_4x4
    
    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to rotation matrix
        """
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def acceleration_orientation_jacobian(self, q):
        """
        Approximate Jacobian of acceleration measurement with respect to orientation
        """
        # This would be computed using derivatives of the rotation matrix
        # For simulation purposes, we'll return a placeholder
        return np.zeros((3, 4))
    
    def quaternion_omega_jacobian(self, q):
        """
        Jacobian of quaternion derivative with respect to angular velocity
        """
        w, x, y, z = q
        J = 0.5 * np.array([
            [-x, -y, -z],
            [w, -z, y],
            [z, w, -x],
            [-y, x, w]
        ])
        
        return J
    
    def h(self, state):
        """
        Measurement model: expected measurement from state
        """
        # State: [qw, qx, qy, qz, wx, wy, wz, bx, by, bz]
        omega = state[4:7]  # Angular velocity
        b = state[7:10]    # Gyroscope bias
        q = state[0:4]     # Quaternion
        
        # Expected angular velocity measurement (true value + potential noise)
        expected_omega = omega - b
        
        # Expected acceleration measurement
        # Transform gravity to body frame using rotation matrix
        R = self.quaternion_to_rotation_matrix(q)
        expected_acc = R.T @ self.gravity
        
        # Combine measurements
        expected_measurement = np.concatenate([expected_omega, expected_acc])
        
        return expected_measurement
```

## Integration with Robot Control

### Using IMU Data for Balance Control

IMU data is critical for humanoid robot balance control:

```python
class HumanoidBalanceController:
    def __init__(self):
        self.imu_data = None
        self.target_orientation = np.array([0, 0, 0, 1])  # Identity quaternion
        self.balance_pid = {
            'pitch': PIDController(kp=100, ki=1, kd=10),
            'roll': PIDController(kp=100, ki=1, kd=10),
            'yaw': PIDController(kp=50, ki=0.5, kd=5)
        }
        
        # Subscribe to IMU data
        self.imu_subscriber = None  # Would be initialized with ROS2 subscription
    
    def update_balance_control(self, imu_msg):
        """
        Update balance control based on IMU readings
        """
        # Extract orientation from IMU
        current_orientation = np.array([
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ])
        
        # Convert to Euler angles for easier control
        euler_angles = self.quaternion_to_euler(current_orientation)
        roll, pitch, yaw = euler_angles
        
        # Calculate errors
        target_euler = self.quaternion_to_euler(self.target_orientation)
        target_roll, target_pitch, target_yaw = target_euler
        
        roll_error = target_roll - roll
        pitch_error = target_pitch - pitch
        yaw_error = target_yaw - yaw
        
        # Apply PID control to generate corrective torques
        corrective_torque = {
            'roll': self.balance_pid['roll'].update(roll_error),
            'pitch': self.balance_pid['pitch'].update(pitch_error),
            'yaw': self.balance_pid['yaw'].update(yaw_error)
        }
        
        # Generate joint commands to apply corrective torques
        joint_commands = self.calculate_balance_joint_commands(corrective_torque)
        
        return joint_commands
    
    def quaternion_to_euler(self, quat):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        """
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.previous_error = 0.0
        self.integral = 0.0
    
    def update(self, error, dt=0.01):
        """
        Update PID controller with new error
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Store error for next derivative calculation
        self.previous_error = error
        
        return output
```

## Quality Assurance for IMU Simulation

### Validation Approaches

```python
class IMUSimulationValidator:
    def __init__(self, imu_topic, ground_truth_pose_topic):
        self.imu_topic = imu_topic
        self.ground_truth_topic = ground_truth_pose_topic
        self.error_statistics = {
            'orientation_error': [],
            'angular_velocity_error': [],
            'linear_acceleration_error': []
        }
        self.max_error_threshold = 0.1  # Radians for orientation
        self.max_angular_velocity_error = 0.05  # rad/s
        self.max_linear_acc_error = 0.2  # m/s²
    
    def validate_imu_simulation(self, duration=60.0):
        """
        Validate IMU simulation quality by comparing with ground truth
        """
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Compare simulated IMU readings with ground truth pose
            self.compare_with_ground_truth()
            time.sleep(0.1)  # Check every 100ms
        
        # Generate validation report
        return self.generate_validation_report()
    
    def compare_with_ground_truth(self):
        """
        Compare IMU readings with ground truth robot state
        """
        # This would subscribe to both IMU and ground truth topics
        # For simulation, we'll create mock comparison logic
        pass
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report
        """
        report = {
            'overall_status': 'pass',
            'error_statistics': {},
            'validation_results': {}
        }
        
        # Calculate statistics for each measurement type
        if self.error_statistics['orientation_error']:
            orientation_errors = np.array(self.error_statistics['orientation_error'])
            report['error_statistics']['orientation'] = {
                'mean_error': np.mean(orientation_errors),
                'std_dev': np.std(orientation_errors),
                'max_error': np.max(orientation_errors),
                'median_error': np.median(orientation_errors)
            }
            
            # Check if errors exceed threshold
            if np.mean(orientation_errors) > self.max_error_threshold:
                report['overall_status'] = 'fail'
                report['validation_results']['orientation'] = 'fail'
            else:
                report['validation_results']['orientation'] = 'pass'
        
        if self.error_statistics['angular_velocity_error']:
            ang_vel_errors = np.array(self.error_statistics['angular_velocity_error'])
            report['error_statistics']['angular_velocity'] = {
                'mean_error': np.mean(ang_vel_errors),
                'std_dev': np.std(ang_vel_errors),
                'max_error': np.max(ang_vel_errors),
                'median_error': np.median(ang_vel_errors)
            }
            
            if np.mean(ang_vel_errors) > self.max_angular_velocity_error:
                report['overall_status'] = 'fail'
                report['validation_results']['angular_velocity'] = 'fail'
            else:
                report['validation_results']['angular_velocity'] = 'pass'
        
        if self.error_statistics['linear_acceleration_error']:
            lin_acc_errors = np.array(self.error_statistics['linear_acceleration_error'])
            report['error_statistics']['linear_acceleration'] = {
                'mean_error': np.mean(lin_acc_errors),
                'std_dev': np.std(lin_acc_errors),
                'max_error': np.max(lin_acc_errors),
                'median_error': np.median(lin_acc_errors)
            }
            
            if np.mean(lin_acc_errors) > self.max_linear_acc_error:
                report['overall_status'] = 'fail'
                report['validation_results']['linear_acceleration'] = 'fail'
            else:
                report['validation_results']['linear_acceleration'] = 'pass'
        
        return report
```

## Integration with Other Sensors

### Sensor Fusion with IMU Data

```python
class MultiSensorFusion:
    """
    Combine IMU data with other sensor data for improved state estimation
    """
    def __init__(self):
        self.imu_data = None
        self.lidar_data = None
        self.camera_data = None
        self.fused_state = {}
        
    def fuse_imu_with_lidar(self, imu_msg, lidar_msg):
        """
        Fuse IMU data with LiDAR data for improved pose estimation
        """
        # Use IMU to estimate orientation
        imu_orientation = self.extract_orientation(imu_msg)
        
        # Use LiDAR to correct for drift in position estimation
        # This is a simplified approach - real implementation would use
        # advanced filtering techniques like Extended Kalman Filters
        lidar_corrections = self.extract_position_from_lidar(lidar_msg)
        
        # Combine to get more accurate pose estimate
        fused_pose = self.complementary_filter(
            imu_orientation, 
            lidar_corrections, 
            alpha=0.8  # Weight more towards IMU for orientation
        )
        
        return fused_pose
    
    def fuse_imu_with_camera(self, imu_msg, camera_msg):
        """
        Fuse IMU data with camera data
        """
        # Use IMU to stabilize visual odometry from camera
        imu_angular_velocity = self.extract_angular_velocity(imu_msg)
        
        # Visual features would be tracked to estimate motion
        # This is a placeholder for the camera processing
        visual_motion_estimate = self.estimate_motion_from_camera(camera_msg)
        
        # Combine for better motion estimation
        combined_motion = self.fuse_motion_estimates(
            imu_angular_velocity, 
            visual_motion_estimate, 
            weights=(0.7, 0.3)  # Weight based on sensor reliability
        )
        
        return combined_motion
    
    def extract_orientation(self, imu_msg):
        """
        Extract orientation from IMU message
        """
        return np.array([
            imu_msg.orientation.w,
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z
        ])
    
    def extract_angular_velocity(self, imu_msg):
        """
        Extract angular velocity from IMU message
        """
        return np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])
    
    def extract_position_from_lidar(self, lidar_msg):
        """
        Extract position estimate from LiDAR scan
        """
        # This would implement scan matching algorithms
        # to estimate robot motion from LiDAR data
        return np.array([0.0, 0.0, 0.0])  # Placeholder
    
    def estimate_motion_from_camera(self, camera_msg):
        """
        Estimate motion from camera images (visual odometry)
        """
        # This would implement feature tracking and motion estimation
        return np.array([0.0, 0.0, 0.0])  # Placeholder
    
    def fuse_motion_estimates(self, imu_data, camera_data, weights=(0.5, 0.5)):
        """
        Fuse motion estimates from different sensors
        """
        return weights[0] * imu_data + weights[1] * camera_data
    
    def complementary_filter(self, estimate1, estimate2, alpha=0.5):
        """
        Apply complementary filter to combine estimates
        """
        return alpha * estimate1 + (1 - alpha) * estimate2
```

## Troubleshooting Common Issues

### Common IMU Simulation Problems

1. **Drift**: Implement proper bias estimation and correction
2. **Noise**: Calibrate noise parameters to match real sensors
3. **Coordinate Frame Mismatches**: Ensure proper transformation between frames
4. **Integration Errors**: Use appropriate integration methods for attitude estimation

## Summary

This chapter covered the comprehensive implementation of IMU simulation in digital twin environments. We explored:

1. **IMU Configuration**: How to set up realistic IMU sensors in Gazebo
2. **Noise Modeling**: Techniques for creating realistic sensor noise characteristics
3. **Data Processing**: How to process IMU data for robot control applications
4. **Integration with Control**: Using IMU data for balance control and state estimation
5. **Quality Assurance**: Methods for validating IMU simulation accuracy
6. **Multi-Sensor Fusion**: Combining IMU data with other sensors for improved performance

Proper IMU simulation is essential for developing robust humanoid robots that can maintain balance and navigate effectively. The simulation should accurately reflect the behavior of real IMUs, including their noise characteristics, drift patterns, and response to various motions and environmental conditions.

The next chapter will cover generating realistic sensor data streams for AI perception pipelines, integrating the LiDAR, depth camera, and IMU systems into a comprehensive perception framework.