# ROS 2 Action Sequences for LLM-Driven Planning

## Introduction

This chapter covers the implementation of ROS 2 action sequences for LLM-driven robotic planning. It focuses on how Large Language Models can be integrated with the Robot Operating System 2 (ROS 2) to generate executable action sequences that control humanoid robots. We'll explore the technical details of translating LLM-generated plans into ROS 2 actions that can be executed by the robot.

## Understanding ROS 2 Actions

### What are ROS 2 Actions?

ROS 2 Actions are a communication pattern designed for long-running tasks. They provide feedback during execution and support cancellation, making them ideal for robotics tasks that can take significant time to complete.

An action consists of:
- **Goal**: The request to begin a task
- **Feedback**: Information published during execution
- **Result**: The final outcome of the task

### Action Structure

Action interfaces are defined in `.action` files that specify three components:

```
# Goal
geometry_msgs/PoseStamped target_pose
---
# Result
int32 error_code
string error_string
---
# Feedback
geometry_msgs/PoseStamped current_pose
int32 remaining_waypoints
```

## Connecting LLM Output to ROS 2 Actions

### Mapping LLM Plans to ROS Actions

To connect LLM-generated plans to ROS 2, we need to map high-level concepts to specific ROS action types:

```python
class LLMToROSActionMapper:
    def __init__(self):
        # Define mappings from high-level LLM concepts to ROS 2 actions
        self.action_mapping = {
            "navigate_to": {
                "action_type": "nav2_msgs.action.NavigateToPose",
                "required_arguments": ["target_location", "orientation"],
                "parameter_mapping": {
                    "target_location": "pose.position",
                    "orientation": "pose.orientation",
                    "target_frame": "pose.header.frame_id"
                }
            },
            "pick_up_object": {
                "action_type": "manipulation_msgs.action.GraspObject", 
                "required_arguments": ["object_id", "grasp_pose"],
                "parameter_mapping": {
                    "object_id": "object_id",
                    "grasp_pose": "grasp_pose",
                    "approach_direction": "approach_direction"
                }
            },
            "place_object": {
                "action_type": "manipulation_msgs.action.PlaceObject",
                "required_arguments": ["placement_pose", "object_id"],
                "parameter_mapping": {
                    "placement_pose": "placement_pose",
                    "object_id": "object_id",
                    "release_method": "release_type"
                }
            },
            "detect_objects": {
                "action_type": "object_detection_msgs.action.DetectObjects",
                "required_arguments": ["roi", "object_classes"],
                "parameter_mapping": {
                    "roi": "roi",
                    "object_classes": "object_classes",
                    "confidence_threshold": "min_confidence"
                }
            },
            "move_arm": {
                "action_type": "control_msgs.action.FollowJointTrajectory",
                "required_arguments": ["joint_positions", "joint_names"],
                "parameter_mapping": {
                    "joint_positions": "trajectory.joint_names",
                    "joint_names": "trajectory.points.positions",
                    "time_from_start": "trajectory.points.time_from_start"
                }
            }
        }
        
        # Define frame transformations for coordinate systems
        self.coordinate_transforms = {
            "world_frame": "map",
            "robot_frame": "base_link",
            "camera_frame": "camera_depth_optical_frame"
        }
    
    def map_llm_plan_to_ros_actions(self, llm_plan):
        """
        Convert an LLM-generated plan to a sequence of ROS 2 actions
        """
        ros_action_sequence = []
        
        for step in llm_plan.get('steps', []):
            action_type = step.get('action_type')
            
            if action_type in self.action_mapping:
                ros_action = self.create_ros_action(step, self.action_mapping[action_type])
                
                # Validate required arguments are present
                missing_args = self.validate_required_arguments(ros_action, step)
                
                if missing_args:
                    raise ValueError(f"Missing required arguments for action {action_type}: {missing_args}")
                
                # Transform coordinates to ROS-standard frames
                self.transform_coordinates(ros_action)
                
                ros_action_sequence.append(ros_action)
            else:
                raise ValueError(f"Unknown action type from LLM: {action_type}. Cannot map to ROS action.")
        
        return ros_action_sequence
    
    def create_ros_action(self, llm_step, action_definition):
        """
        Create a ROS action from LLM step and action definition
        """
        action_type = action_definition['action_type']
        
        # Import the specific action message type
        module_path, class_name = action_type.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        action_class = getattr(module, class_name)
        
        # Create the goal message
        goal = self.create_action_goal(action_class, llm_step, action_definition)
        
        return {
            'action_type': action_type,
            'goal': goal,
            'timeout': llm_step.get('timeout', 30.0),  # Default 30 second timeout
            'expected_feedback': llm_step.get('enable_feedback', True)
        }
    
    def create_action_goal(self, action_class, llm_step, action_definition):
        """
        Create the goal message for a specific action
        """
        # Get the goal type from the action class
        goal_class = action_class.Goal
        
        # Create an instance of the goal
        goal = goal_class()
        
        # Map parameters from LLM step to ROS goal using parameter mapping
        param_mapping = action_definition['parameter_mapping']
        llm_params = llm_step.get('parameters', {})
        
        for llm_param, ros_param_path in param_mapping.items():
            if llm_param in llm_params:
                self.set_nested_attribute(goal, ros_param_path, llm_params[llm_param])
        
        return goal
    
    def set_nested_attribute(self, obj, path, value):
        """
        Set an attribute on a nested object using a path like 'pose.position.x'
        """
        attrs = path.split('.')
        current_obj = obj
        
        # Navigate to the parent of the final attribute
        for attr in attrs[:-1]:
            current_obj = getattr(current_obj, attr)
        
        # Set the final attribute value
        setattr(current_obj, attrs[-1], value)
    
    def validate_required_arguments(self, ros_action, llm_step):
        """
        Validate that all required arguments for an action are present
        """
        action_type = llm_step.get('action_type')
        required_args = self.action_mapping[action_type]['required_arguments']
        llm_params = llm_step.get('parameters', {})
        
        missing_args = []
        for arg in required_args:
            if arg not in llm_params:
                missing_args.append(arg)
        
        return missing_args
    
    def transform_coordinates(self, ros_action):
        """
        Transform coordinates to ROS-standard frames
        """
        # For now, this is a placeholder - actual implementation would use TF2
        # to transform coordinates between frames as needed
        pass
```

## Implementing Action Clients

### Generic Action Client

Create a generic client to handle different ROS action types:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import time
import threading
from concurrent.futures import Future

class GenericActionExecutor(Node):
    def __init__(self):
        super().__init__('generic_action_executor')
        
        # Store active action clients for different action types
        self.active_clients = {}
        self.active_goals = {}
    
    def execute_action_sequence(self, action_sequence):
        """
        Execute a sequence of ROS actions
        """
        results = []
        
        for i, action in enumerate(action_sequence):
            try:
                result = self.execute_single_action(action, i)
                results.append({
                    'index': i,
                    'action_type': action['action_type'],
                    'result': result,
                    'status': 'success'
                })
                
                # Check if the action sequence should continue based on result
                if not self.should_continue_after_action(result):
                    break
            except Exception as e:
                results.append({
                    'index': i,
                    'action_type': action['action_type'],
                    'error': str(e),
                    'status': 'failure'
                })
                # Decide whether to continue or abort based on configuration
                if not self.should_continue_on_failure():
                    break
        
        return results
    
    def execute_single_action(self, action, index):
        """
        Execute a single ROS action
        """
        action_module_path, action_class_name = action['action_type'].rsplit('.', 1)
        action_module = __import__(action_module_path, fromlist=[action_class_name])
        action_class = getattr(action_module, action_class_name)
        
        # Create action client if it doesn't exist
        action_name = self.derive_action_name_from_type(action['action_type'])
        if action_name not in self.active_clients:
            self.active_clients[action_name] = ActionClient(self, action_class, action_name)
        
        action_client = self.active_clients[action_name]
        
        # Wait for action server
        if not action_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Action server not available after waiting: {action_name}")
        
        # Create the goal
        goal_msg = action['goal']
        
        # Send the goal
        goal_future = action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.create_feedback_callback(index, action)
        )
        
        # Wait for result
        result = self.wait_for_result(goal_future, action['timeout'])
        return result
    
    def derive_action_name_from_type(self, action_type):
        """
        Derive action server name from action type
        This is a simplified heuristic - in practice, action names should be configurable
        """
        # Example conversions:
        # nav2_msgs.action.NavigateToPose -> /navigate_to_pose
        # manipulation_msgs.action.GraspObject -> /grasp_object (simplified)
        
        parts = action_type.split('.')
        if len(parts) >= 3:
            # Take the last two parts (action name and class name) and convert to snake_case
            action_class = parts[-1]
            # Convert CamelCase to snake_case
            import re
            snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', action_class).lower()
            return f"/{snake_case}"
        
        # Fallback
        return f"/{action_type.replace('.', '_').replace('action', '').lower()}"
    
    def create_feedback_callback(self, step_index, action):
        """
        Create feedback callback for an action
        """
        def feedback_callback(feedback_msg):
            self.get_logger().info(f"Step {step_index} feedback: {feedback_msg}")
            
            # Could store feedback, trigger other behaviors, etc.
            # For now, just log it
            pass
        
        return feedback_callback
    
    def wait_for_result(self, goal_future, timeout):
        """
        Wait for action result with timeout
        """
        # Create a timer to implement timeout
        timeout_occurred = [False]
        
        def timeout_callback():
            timeout_occurred[0] = True
        
        timer = self.create_timer(timeout, timeout_callback)
        
        # Wait for the future to complete or timeout
        start_time = time.time()
        while not goal_future.done() and not timeout_occurred[0]:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time >= timeout:
                timeout_occurred[0] = True
                break
        
        # Cancel the timer
        self.destroy_timer(timer)
        
        if timeout_occurred[0]:
            raise TimeoutError(f"Action timed out after {timeout} seconds")
        
        # Get the goal handle
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            raise RuntimeError('Goal was rejected by server')
        
        # Get the result
        result_future = goal_handle.get_result_async()
        
        # Wait for result with remaining time
        remaining_time = max(0.1, timeout - (time.time() - start_time))
        
        start_time = time.time()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time >= remaining_time:
                raise TimeoutError(f"Action result retrieval timed out after {remaining_time} seconds")
        
        result = result_future.result().result
        return result
    
    def should_continue_after_action(self, result):
        """
        Determine if execution should continue after an action completes
        """
        # Default: continue unless explicitly told to stop
        return True
    
    def should_continue_on_failure(self):
        """
        Determine if execution should continue after an action fails
        """
        # Default: stop on failure
        return False
```

### Action Sequence Orchestrator

An orchestrator to manage the execution of action sequences:

```python
class ActionSequenceOrchestrator:
    def __init__(self, ros_node):
        self.node = ros_node
        self.executor = GenericActionExecutor()
        self.llm_planner = LLMRobotPlanner()  # Initialized elsewhere
        self.action_mapper = LLMToROSActionMapper()
        
    def execute_natural_language_command(self, command, context=None):
        """
        Execute a natural language command end-to-end
        """
        try:
            # Step 1: Use LLM to generate a high-level plan
            llm_plan = self.llm_planner.plan_action_sequence(command, context)
            
            # Step 2: Validate the plan
            if not self.validate_plan(llm_plan):
                raise ValueError("Invalid plan generated by LLM")
            
            # Step 3: Map the LLM plan to ROS actions
            ros_action_sequence = self.action_mapper.map_llm_plan_to_ros_actions(llm_plan)
            
            # Step 4: Validate the ROS action sequence
            if not self.validate_ros_sequence(ros_action_sequence):
                raise ValueError("Invalid ROS action sequence generated")
            
            # Step 5: Execute the action sequence
            execution_results = self.executor.execute_action_sequence(ros_action_sequence)
            
            # Step 6: Process results and return meaningful output
            return self.process_execution_results(execution_results, command)
            
        except Exception as e:
            # Log error and potentially return meaningful error information
            self.node.get_logger().error(f"Error executing command '{command}': {str(e)}")
            return {
                "status": "failure",
                "error": str(e),
                "command": command
            }
    
    def validate_plan(self, plan):
        """
        Validate an LLM-generated plan for correctness
        """
        if not plan or "steps" not in plan:
            return False
        
        # Check that each step has required fields
        for step in plan.get("steps", []):
            if "action_type" not in step:
                return False
        
        return True
    
    def validate_ros_sequence(self, action_sequence):
        """
        Validate a ROS action sequence before execution
        """
        # Check that action sequence is not empty
        if not action_sequence:
            return False
        
        # Check that each action has required fields
        for action in action_sequence:
            if "action_type" not in action or "goal" not in action:
                return False
        
        return True
    
    def process_execution_results(self, results, original_command):
        """
        Process execution results and generate meaningful output
        """
        success_count = sum(1 for r in results if r['status'] == 'success')
        total_count = len(results)
        
        overall_status = "success" if success_count == total_count else "partial_success" if success_count > 0 else "failure"
        
        return {
            "status": overall_status,
            "command": original_command,
            "action_results": results,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "execution_summary": f"Executed {total_count} actions, {success_count} succeeded"
        }
```

## Handling Complex Sequences

### Conditional Action Sequences

For actions that depend on the results of previous actions:

```python
class ConditionalActionHandler:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    def execute_conditional_sequence(self, conditional_plan):
        """
        Execute actions with conditional dependencies
        """
        results = []
        
        for condition_block in conditional_plan:
            # Evaluate the condition
            if self.evaluate_condition(condition_block.get('condition', {})):
                # Execute the action sequence if condition is met
                actions = condition_block.get('actions', [])
                block_results = self.orchestrator.executor.execute_action_sequence(actions)
                results.extend(block_results)
                
                # Optionally break or continue based on results
                if condition_block.get('terminate_on_complete', False):
                    break
            else:
                # Handle the else case if specified
                else_actions = condition_block.get('else_actions', [])
                if else_actions:
                    else_results = self.orchestrator.executor.execute_action_sequence(else_actions)
                    results.extend(else_results)
        
        return results
    
    def evaluate_condition(self, condition):
        """
        Evaluate a condition for conditional execution
        """
        # Example condition evaluation - in reality this would be more complex
        condition_type = condition.get('type', 'default')
        
        if condition_type == 'sensor_reading':
            # Check a sensor reading against a threshold
            return self.check_sensor_condition(condition)
        elif condition_type == 'task_success':
            # Check if a previous task was successful
            return self.check_task_success(condition)
        elif condition_type == 'object_detected':
            # Check if an object was detected
            return self.check_object_detection(condition)
        else:
            # Default to True if condition type is unknown
            return True
    
    def check_sensor_condition(self, condition):
        """
        Check a sensor-based condition
        """
        sensor_topic = condition.get('sensor_topic')
        threshold = condition.get('threshold')
        comparison = condition.get('comparison', 'greater_than')  # greater_than, less_than, equals
        
        # In a real implementation, this would subscribe to the sensor topic
        # and get the latest value to compare against the threshold
        current_value = self.get_latest_sensor_value(sensor_topic)
        
        if comparison == 'greater_than':
            return current_value > threshold
        elif comparison == 'less_than':
            return current_value < threshold
        elif comparison == 'equals':
            return abs(current_value - threshold) < 0.001  # Small epsilon for floating point comparison
        else:
            return True  # Default to True if comparison type is unknown
```

### Loop-Based Action Sequences

For repetitive actions:

```python
class LoopActionHandler:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    def execute_loop_sequence(self, loop_plan):
        """
        Execute repetitive actions based on a loop condition
        """
        iteration = 0
        max_iterations = loop_plan.get('max_iterations', 10)
        results = []
        
        while self.check_loop_condition(loop_plan.get('condition', {}), iteration):
            if iteration >= max_iterations:
                self.orchestrator.node.get_logger().warn(
                    f"Maximum iterations ({max_iterations}) reached in loop"
                )
                break
            
            # Execute the loop body actions
            loop_actions = loop_plan.get('loop_body', [])
            try:
                iteration_results = self.orchestrator.executor.execute_action_sequence(loop_actions)
                
                # Store results
                for result in iteration_results:
                    result['iteration'] = iteration
                    results.append(result)
                
                # Check if we should break early
                if self.should_break_early(iteration_results, loop_plan):
                    self.orchestrator.node.get_logger().info(
                        f"Breaking loop early at iteration {iteration} based on results"
                    )
                    break
                
                iteration += 1
                
                # Optional: wait between iterations
                if 'iteration_delay' in loop_plan:
                    self.orchestrator.node.get_clock().sleep_for(
                        rclpy.duration.Duration(seconds=loop_plan['iteration_delay'])
                    )
                    
            except Exception as e:
                self.orchestrator.node.get_logger().error(
                    f"Error in loop iteration {iteration}: {str(e)}"
                )
                
                # Decide whether to break or continue based on error
                if loop_plan.get('abort_on_error', True):
                    break
                else:
                    iteration += 1
        
        return results
    
    def check_loop_condition(self, condition, iteration):
        """
        Check if the loop should continue
        """
        condition_type = condition.get('type', 'count_based')
        
        if condition_type == 'count_based':
            max_count = condition.get('max_count', 10)
            return iteration < max_count
        elif condition_type == 'sensor_based':
            # Check a sensor reading to decide whether to continue
            return self.check_sensor_condition(condition)
        elif condition_type == 'event_based':
            # Check for a specific event to occur
            return self.check_event_occurred(condition)
        else:
            return False  # Default to not continuing if condition is unknown
```

## Integration with Large Language Models

### LLM Plan Generation and Validation

Creating a complete pipeline that integrates LLMs with ROS 2 actions:

```python
class LLMROS2Planner:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.action_mapper = LLMToROSActionMapper()
        
        # Store robot capabilities to constrain LLM suggestions
        self.robot_capabilities = {
            "navigation": {
                "supported": True,
                "max_speed": 1.0,  # m/s
                "min_turn_radius": 0.3  # meters
            },
            "manipulation": {
                "supported": True,
                "max_payload": 2.0,  # kg
                "reachable_workspace": {
                    "min_x": -0.5, "max_x": 0.5,
                    "min_y": -0.3, "max_y": 0.3,
                    "min_z": 0.1, "max_z": 1.0
                }
            },
            "perception": {
                "lidar": {"range": 30.0, "fov_horizontal": 360},
                "camera": {"resolution": [640, 480], "fov_horizontal": 90}
            }
        }
    
    def generate_plan(self, natural_language_command, robot_state=None, environment_context=None):
        """
        Generate a plan from natural language command using LLM
        """
        prompt = self.construct_planning_prompt(
            command=natural_language_command,
            robot_capabilities=self.robot_capabilities,
            robot_state=robot_state,
            environment_context=environment_context
        )
        
        try:
            response = self.call_llm(prompt)
            plan = self.parse_llm_response(response)
            
            # Validate the plan against robot capabilities
            validated_plan = self.validate_plan_against_capabilities(plan, natural_language_command)
            
            return validated_plan
        except Exception as e:
            self.get_logger().error(f"Error generating plan: {str(e)}")
            return self.generate_fallback_plan(natural_language_command, str(e))
    
    def construct_planning_prompt(self, command, robot_capabilities, robot_state=None, environment_context=None):
        """
        Construct a detailed prompt for the LLM with all relevant context
        """
        prompt_parts = [
            "You are a robot task planner. Your role is to break down natural language commands ",
            "into specific, executable robot actions. The robot has the following capabilities:\n\n"
        ]
        
        # Add robot capabilities to prompt
        prompt_parts.append(self.capabilities_to_prompt(robot_capabilities))
        
        # Add current state if available
        if robot_state:
            prompt_parts.append(f"\nCurrent robot state: {robot_state}\n")
        
        # Add environment context if available
        if environment_context:
            prompt_parts.append(f"Environment context: {environment_context}\n")
        
        # The actual command to process
        prompt_parts.append(f"Command to process: {command}\n")
        
        # Expected response format
        prompt_parts.append(
            "Please respond with a structured plan in JSON format with the following structure:\n"
            "{\n"
            "  \"task_description\": \"Brief description of the overall task\",\n"
            "  \"steps\": [\n"
            "    {\n"
            "      \"step_number\": 1,\n"
            "      \"action_type\": \"action_type_from_supported_list\",\n"
            "      \"parameters\": {\"param_name\": \"param_value\"},\n"
            "      \"description\": \"Human-readable description of the action\",\n"
            "      \"timeout\": 30.0,\n"
            "      \"require_confirmation\": false\n"
            "    }\n"
            "  ],\n"
            "  \"estimated_completion_time\": 60.0\n"
            "}\n"
        )
        
        # Add specific constraints based on robot capabilities
        prompt_parts.append(self.generate_capability_constraints(robot_capabilities))
        
        return "".join(prompt_parts)
    
    def capabilities_to_prompt(self, capabilities):
        """
        Convert robot capabilities to a prompt string
        """
        cap_strings = []
        
        if capabilities.get('navigation', {}).get('supported'):
            nav_caps = capabilities['navigation']
            cap_strings.append("- Navigation: Max speed {:.1f} m/s, min turn radius {:.1f}m".format(
                nav_caps['max_speed'], nav_caps['min_turn_radius']))
        
        if capabilities.get('manipulation', {}).get('supported'):
            manip_caps = capabilities['manipulation']
            workspace = manip_caps['reachable_workspace']
            cap_strings.append(
                "- Manipulation: Max payload {:.1f}kg, reachable workspace x:[{:.1f},{:.1f}], y:[{:.1f},{:.1f}], z:[{:.1f},{:.1f}]m".format(
                    manip_caps['max_payload'],
                    workspace['min_x'], workspace['max_x'],
                    workspace['min_y'], workspace['max_y'],
                    workspace['min_z'], workspace['max_z']))
        
        if 'perception' in capabilities:
            perc_caps = capabilities['perception']
            lidar_info = perc_caps.get('lidar', {})
            camera_info = perc_caps.get('camera', {})
            
            if lidar_info:
                cap_strings.append("- LiDAR: Max range {:.1f}m, FOV {}°".format(
                    lidar_info['range'], lidar_info['fov_horizontal']))
            if camera_info:
                cap_strings.append("- Camera: {}x{}, FOV {}°".format(
                    camera_info['resolution'][0], camera_info['resolution'][1],
                    camera_info['fov_horizontal']))
        
        return "\n".join(cap_strings)
    
    def generate_capability_constraints(self, capabilities):
        """
        Generate constraints based on robot capabilities
        """
        constraints = []
        
        # Navigation constraints
        if capabilities.get('navigation', {}).get('supported'):
            max_speed = capabilities['navigation']['max_speed']
            constraints.append(
                f"\nConstraint: Robot navigation speed should not exceed {max_speed} m/s."
            )
        
        # Manipulation constraints
        if capabilities.get('manipulation', {}).get('supported'):
            max_payload = capabilities['manipulation']['max_payload']
            workspace = capabilities['manipulation']['reachable_workspace']
            constraints.append(
                f"\nConstraint: Robot can manipulate objects up to {max_payload} kg."
                f" Objects must be within workspace: x:[{workspace['min_x']},{workspace['max_x']}]m, "
                f"y:[{workspace['min_y']},{workspace['max_y']}]m, "
                f"z:[{workspace['min_z']},{workspace['max_z']}]m."
            )
        
        return "".join(constraints)
    
    def call_llm(self, prompt):
        """
        Call the LLM with the constructed prompt
        """
        if not self.api_key:
            # For demonstration purposes, returning a mock response
            # In a real implementation, you would call the actual LLM API
            return self.mock_llm_response(prompt)
        
        # Actual implementation would be:
        # import openai
        # openai.api_key = self.api_key
        # response = openai.ChatCompletion.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.2
        # )
        # return response.choices[0].message.content
    
    def mock_llm_response(self, prompt):
        """
        Mock response for demonstration purposes
        """
        # This is a mock response that would typically come from an actual LLM
        # In a real system, this would be replaced with actual LLM call
        if "navigate" in prompt.lower():
            return '''
            {
              "task_description": "Navigate to specified location",
              "steps": [
                {
                  "step_number": 1,
                  "action_type": "navigate_to",
                  "parameters": {
                    "target_location": {
                      "x": 2.5,
                      "y": 1.0,
                      "z": 0.0
                    },
                    "orientation": {
                      "w": 1.0,
                      "x": 0.0,
                      "y": 0.0,
                      "z": 0.0
                    }
                  },
                  "description": "Navigate to the specified location",
                  "timeout": 60.0,
                  "require_confirmation": false
                }
              ],
              "estimated_completion_time": 60.0
            }'''
        elif "pick up" in prompt.lower() or "grasp" in prompt.lower():
            return '''
            {
              "task_description": "Detect and pick up specified object",
              "steps": [
                {
                  "step_number": 1,
                  "action_type": "detect_objects",
                  "parameters": {
                    "roi": {
                      "x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0
                    },
                    "object_classes": ["cup", "bottle", "box"],
                    "confidence_threshold": 0.7
                  },
                  "description": "Detect target object in specified region",
                  "timeout": 10.0,
                  "require_confirmation": false
                },
                {
                  "step_number": 2,
                  "action_type": "pick_up_object", 
                  "parameters": {
                    "object_id": "target_object_detected_in_previous_step",
                    "grasp_pose": {
                      "x": 1.5, "y": 1.2, "z": 0.8,
                      "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0
                    }
                  },
                  "description": "Grasp the detected object",
                  "timeout": 30.0,
                  "require_confirmation": false
                }
              ],
              "estimated_completion_time": 45.0
            }'''
        else:
            return '''
            {
              "task_description": "Perform basic robot action",
              "steps": [
                {
                  "step_number": 1,
                  "action_type": "detect_objects",
                  "parameters": {
                    "roi": {
                      "x": 0.0, "y": 0.0, "width": 5.0, "height": 5.0
                    },
                    "object_classes": ["any"],
                    "confidence_threshold": 0.5
                  },
                  "description": "Scan environment for objects",
                  "timeout": 10.0,
                  "require_confirmation": false
                }
              ],
              "estimated_completion_time": 15.0
            }'''
    
    def parse_llm_response(self, response_text):
        """
        Parse the LLM's response into a structured plan
        """
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
                plan = json.loads(json_str)
                return plan
            else:
                raise ValueError("No JSON found in LLM response")
        except (json.JSONDecodeError, ValueError) as e:
            self.get_logger().error(f"Error parsing LLM response: {str(e)}")
            # Return a basic fallback plan
            return {
                "task_description": "Error in plan generation",
                "steps": [],
                "estimated_completion_time": 0.0
            }
    
    def validate_plan_against_capabilities(self, plan, original_command):
        """
        Validate that the generated plan is possible with the robot's capabilities
        """
        validated_plan = plan.copy()
        steps = validated_plan.get('steps', [])
        
        for step in steps:
            action_type = step.get('action_type')
            params = step.get('parameters', {})
            
            # Validate navigation actions
            if action_type == 'navigate_to':
                target_pos = params.get('target_location', {})
                if not self.validate_navigation_target(target_pos):
                    # Adjust target or generate error
                    self.get_logger().warn(f"Navigation target out of bounds: {target_pos}")
            
            # Validate manipulation actions
            elif action_type == 'pick_up_object':
                grasp_pose = params.get('grasp_pose', {})
                if not self.validate_manipulation_pose(grasp_pose):
                    self.get_logger().warn(f"Grasp pose out of reachable workspace: {grasp_pose}")
        
        return validated_plan
    
    def validate_navigation_target(self, target_pos):
        """
        Validate that a navigation target is within robot capabilities
        """
        # For now, we'll assume any target is valid
        # In a real implementation, this would check for obstacles, etc.
        return True
    
    def validate_manipulation_pose(self, grasp_pose):
        """
        Validate that a manipulation pose is within robot's reachable workspace
        """
        workspace = self.robot_capabilities['manipulation']['reachable_workspace']
        
        x = grasp_pose.get('x', 0)
        y = grasp_pose.get('y', 0) 
        z = grasp_pose.get('z', 0)
        
        return (workspace['min_x'] <= x <= workspace['max_x'] and
                workspace['min_y'] <= y <= workspace['max_y'] and
                workspace['min_z'] <= z <= workspace['max_z'])
    
    def generate_fallback_plan(self, command, error):
        """
        Generate a fallback plan when LLM planning fails
        """
        self.get_logger().info(f"Generating fallback plan for command: {command}, Error: {error}")
        return {
            "task_description": f"Error processing command: {command}. Original error: {error}",
            "steps": [
                {
                    "step_number": 1,
                    "action_type": "request_assistance",
                    "parameters": {"message": f"Could not understand command: {command}"},
                    "description": "Request human assistance due to command processing error",
                    "timeout": 10.0,
                    "require_confirmation": False
                }
            ],
            "estimated_completion_time": 10.0
        }
    
    def get_logger(self):
        """
        Simple logger for this class
        """
        import logging
        return logging.getLogger(__name__)
```

## Error Handling and Recovery

### Action Execution Error Handling

Implement robust error handling for action sequences:

```python
class ActionErrorHandler:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
        # Define recovery strategies for different error types
        self.recovery_strategies = {
            'navigation_failure': self.recover_from_navigation_failure,
            'manipulation_failure': self.recover_from_manipulation_failure,
            'sensor_timeout': self.recover_from_sensor_timeout,
            'object_not_found': self.recover_from_object_not_found
        }
    
    def handle_action_error(self, action, error, context=None):
        """
        Handle error from a failed action and attempt recovery
        """
        error_type = self.classify_error(error)
        
        if error_type in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[error_type](action, error, context)
                return recovery_result
            except Exception as recovery_error:
                self.orchestrator.node.get_logger().error(
                    f"Recovery from {error_type} failed: {recovery_error}"
                )
                return {
                    "status": "recovery_failed",
                    "original_error": str(error),
                    "recovery_error": str(recovery_error)
                }
        else:
            # No recovery strategy available
            return {
                "status": "no_recovery",
                "error": str(error),
                "error_type": error_type
            }
    
    def classify_error(self, error):
        """
        Classify an error to determine appropriate recovery strategy
        """
        error_msg = str(error).lower()
        
        # Simple classification based on error message content
        if 'navigation' in error_msg or 'path' in error_msg or 'goal' in error_msg:
            return 'navigation_failure'
        elif 'grasp' in error_msg or 'manipulat' in error_msg:
            return 'manipulation_failure'
        elif 'timeout' in error_msg or 'sensor' in error_msg:
            return 'sensor_timeout'
        elif 'not found' in error_msg or 'detect' in error_msg:
            return 'object_not_found'
        else:
            return 'unknown_error'
    
    def recover_from_navigation_failure(self, action, error, context):
        """
        Recovery strategy for navigation failures
        """
        self.orchestrator.node.get_logger().info("Attempting navigation recovery...")
        
        # Try alternative navigation approaches
        original_goal = action['goal']
        
        # 1. Try to find an alternative path to the same goal
        alternative_goals = self.generate_alternative_navigation_goals(original_goal)
        
        for alt_goal in alternative_goals:
            try:
                # Create a new action with alternative goal
                alt_action = action.copy()
                alt_action['goal'] = alt_goal
                
                result = self.orchestrator.executor.execute_single_action(alt_action, "recovery_attempt")
                if result['status'] == 'success':
                    return {
                        "status": "recovered",
                        "new_trajectory": alt_goal,
                        "message": "Successfully recovered with alternative navigation strategy"
                    }
            except:
                continue  # Try next alternative
        
        # 2. If alternative paths don't work, try getting closer to goal
        partial_goals = self.generate_partial_navigation_goals(original_goal)
        
        for partial_goal in partial_goals:
            try:
                alt_action = action.copy()
                alt_action['goal'] = partial_goal
                
                result = self.orchestrator.executor.execute_single_action(alt_action, "partial_goal")
                if result['status'] == 'success':
                    return {
                        "status": "partially_recovered",
                        "partial_goal": partial_goal,
                        "message": "Reached partial goal as recovery from navigation failure"
                    }
            except:
                continue
        
        return {
            "status": "recovery_failed",
            "message": "Could not recover from navigation failure"
        }
    
    def generate_alternative_navigation_goals(self, original_goal):
        """
        Generate alternative navigation goals near the original goal
        """
        # In a real implementation, this would use path planning algorithms
        # to find alternative routes or waypoints
        alternatives = []
        
        # Example: slightly shifted positions around original goal
        original_pos = (original_goal.pose.position.x, original_goal.pose.position.y)
        
        offsets = [
            (0.5, 0.0), (-0.5, 0.0),  # Slightly left/right
            (0.0, 0.5), (0.0, -0.5),  # Slightly forward/backward
            (0.3, 0.3), (0.3, -0.3), (-0.3, 0.3), (-0.3, -0.3)  # Diagonals
        ]
        
        for dx, dy in offsets:
            new_pos = (original_pos[0] + dx, original_pos[1] + dy)
            
            new_goal = copy.deepcopy(original_goal)
            new_goal.pose.position.x = new_pos[0]
            new_goal.pose.position.y = new_pos[1]
            
            alternatives.append(new_goal)
        
        return alternatives
    
    def generate_partial_navigation_goals(self, original_goal):
        """
        Generate goals that are steps toward the original goal
        """
        # In a real implementation, this would calculate waypoints
        # along the path to the original goal
        partials = []
        
        # Example: create goals at 75%, 50%, and 25% of the way to goal
        # This is a simplified example - real implementation would use robot's current pose
        current_pos = (0, 0)  # Robot's current position would be real
        goal_pos = (original_goal.pose.position.x, original_goal.pose.position.y)
        
        for ratio in [0.75, 0.5, 0.25]:
            partial_x = current_pos[0] + ratio * (goal_pos[0] - current_pos[0])
            partial_y = current_pos[1] + ratio * (goal_pos[1] - current_pos[1])
            
            partial_goal = copy.deepcopy(original_goal)
            partial_goal.pose.position.x = partial_x
            partial_goal.pose.position.y = partial_y
            
            partials.append(partial_goal)
        
        return partials
    
    def recover_from_manipulation_failure(self, action, error, context):
        """
        Recovery strategy for manipulation failures
        """
        self.orchestrator.node.get_logger().info("Attempting manipulation recovery...")
        
        # Try different grasp approaches
        original_grasp = action['parameters'].get('grasp_pose')
        
        if original_grasp:
            alternative_grasps = self.generate_alternative_grasps(original_grasp)
            
            for alt_grasp in alternative_grasps:
                try:
                    alt_action = copy.deepcopy(action)
                    alt_action['parameters']['grasp_pose'] = alt_grasp
                    
                    result = self.orchestrator.executor.execute_single_action(alt_action, "alternative_grasp")
                    if result['status'] == 'success':
                        return {
                            "status": "recovered",
                            "alternative_grasp": alt_grasp,
                            "message": "Successfully recovered with alternative grasp approach"
                        }
                except:
                    continue
        
        return {
            "status": "recovery_failed",
            "message": "Could not recover from manipulation failure"
        }
    
    def generate_alternative_grasps(self, original_grasp):
        """
        Generate alternative grasp poses around the original grasp
        """
        alternatives = []
        
        # Example: vary approach angle or grasp orientation
        for angle_offset in [0.2, -0.2, 0.4, -0.4]:  # Radians
            alt_grasp = copy.deepcopy(original_grasp)
            # Modify orientation based on offset (simplified)
            # In a real implementation, this would properly adjust the quaternion
            alt_grasp['qx'] += angle_offset * 0.1  # Small adjustment
            alt_grasp['qz'] += angle_offset * 0.1
            
            alternatives.append(alt_grasp)
        
        return alternatives
    
    def recover_from_sensor_timeout(self, action, error, context):
        """
        Recovery strategy for sensor timeouts
        """
        self.orchestrator.node.get_logger().info("Attempting sensor timeout recovery...")
        
        # Increase timeout and try again
        try:
            increased_timeout_action = copy.deepcopy(action)
            increased_timeout_action['timeout'] *= 2  # Double the timeout
            
            result = self.orchestrator.executor.execute_single_action(
                increased_timeout_action, "increased_timeout"
            )
            
            if result['status'] == 'success':
                return {
                    "status": "recovered",
                    "message": "Successfully recovered by increasing sensor timeout"
                }
        except Exception:
            pass
        
        # If increasing timeout didn't work, try repositioning robot
        reposition_action = self.generate_repositioning_action(context)
        if reposition_action:
            try:
                result = self.orchestrator.executor.execute_single_action(
                    reposition_action, "reposition_sensor"
                )
                
                if result['status'] == 'success':
                    # Retry original action after repositioning
                    original_result = self.orchestrator.executor.execute_single_action(
                        action, "retry_after_reposition"
                    )
                    
                    if original_result['status'] == 'success':
                        return {
                            "status": "recovered",
                            "message": "Successfully recovered by repositioning and retrying"
                        }
            except:
                pass
        
        return {
            "status": "recovery_failed",
            "message": "Could not recover from sensor timeout"
        }
    
    def generate_repositioning_action(self, context):
        """
        Generate an action to reposition the robot for better sensor view
        """
        # In a real implementation, this would calculate a new position
        # that provides better sensor coverage for the task
        if not context or 'target_object' not in context:
            return None
        
        target_pos = context['target_object'].get('position', {'x': 0, 'y': 0, 'z': 0})
        
        # Calculate a position that's closer to the target or at a better angle
        reposition_goal = {
            'action_type': 'navigate_to',
            'parameters': {
                'target_location': target_pos  # Simplified - in reality would be offset from target
            }
        }
        
        return reposition_goal
    
    def recover_from_object_not_found(self, action, error, context):
        """
        Recovery strategy when an expected object is not found
        """
        self.orchestrator.node.get_logger().info("Attempting object detection recovery...")
        
        # Try expanding the search area
        original_roi = action['parameters'].get('roi', {})
        expanded_roi = self.expand_search_area(original_roi)
        
        if expanded_roi:
            expanded_action = copy.deepcopy(action)
            expanded_action['parameters']['roi'] = expanded_roi
            expanded_action['timeout'] *= 2  # Allow more time for larger search
            
            try:
                result = self.orchestrator.executor.execute_single_action(
                    expanded_action, "expanded_search"
                )
                
                if result['status'] == 'success':
                    return {
                        "status": "recovered",
                        "message": "Successfully found object in expanded search area"
                    }
            except:
                pass
        
        # If expanding search didn't work, try requesting user guidance
        return {
            "status": "request_user_assistance",
            "message": "Object not found. Please help locate the target object."
        }
    
    def expand_search_area(self, roi):
        """
        Expand the region of interest for object detection
        """
        if not roi:
            return None
        
        # Increase the search area by 50% in each dimension
        expansion_factor = 1.5
        center_x = roi.get('x', 0) + roi.get('width', 0) / 2
        center_y = roi.get('y', 0) + roi.get('height', 0) / 2
        
        new_width = roi.get('width', 1) * expansion_factor
        new_height = roi.get('height', 1) * expansion_factor
        
        expanded_roi = {
            'x': center_x - new_width / 2,
            'y': center_y - new_height / 2,
            'width': new_width,
            'height': new_height
        }
        
        return expanded_roi
```

## Performance Optimization

### Efficient Action Execution

Optimize the performance of action sequences:

```python
class EfficientActionExecutor:
    def __init__(self, node):
        self.node = node
        self.action_clients = {}
        self.future_watchers = []
    
    def execute_action_sequence_optimized(self, action_sequence, parallelizable_threshold=3):
        """
        Execute action sequence with optimizations for performance
        """
        if len(action_sequence) <= parallelizable_threshold:
            # Execute sequentially for short sequences to avoid overhead
            return self.execute_sequentially(action_sequence)
        else:
            # For longer sequences, identify opportunities for parallelization
            return self.execute_with_parallel_opportunities(action_sequence)
    
    def execute_sequentially(self, action_sequence):
        """
        Execute actions one after another with minimal overhead
        """
        results = []
        
        for i, action in enumerate(action_sequence):
            start_time = self.node.get_clock().now()
            
            try:
                result = self.execute_single_action_optimized(action)
                
                execution_time = (self.node.get_clock().now() - start_time).nanoseconds / 1e9
                
                results.append({
                    'index': i,
                    'action_type': action['action_type'],
                    'result': result,
                    'status': 'success',
                    'execution_time': execution_time
                })
            except Exception as e:
                execution_time = (self.node.get_clock().now() - start_time).nanoseconds / 1e9
                
                results.append({
                    'index': i,
                    'action_type': action['action_type'],
                    'error': str(e),
                    'status': 'failure',
                    'execution_time': execution_time
                })
        
        return results
    
    def execute_with_parallel_opportunities(self, action_sequence):
        """
        Execute actions with intelligent parallelization where possible
        """
        results = []
        
        # Identify groups of actions that can potentially run in parallel
        action_groups = self.identify_action_groups(action_sequence)
        
        for group in action_groups:
            if len(group) == 1:
                # Single action - execute normally
                result = self.execute_single_action_optimized(group[0])
                results.append(result)
            else:
                # Multiple actions that can run in parallel
                group_results = self.execute_group_in_parallel(group)
                results.extend(group_results)
        
        return results
    
    def identify_action_groups(self, action_sequence):
        """
        Group actions that can potentially run in parallel
        This is a simplified implementation - real implementation would be more sophisticated
        """
        groups = []
        current_group = []
        
        for action in action_sequence:
            # Simple heuristic: actions that don't share resources can run in parallel
            if self.can_run_in_parallel(action, current_group):
                current_group.append(action)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [action]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def can_run_in_parallel(self, new_action, existing_group):
        """
        Determine if a new action can run in parallel with existing group
        """
        # Actions that access the same resource (e.g., gripper) cannot run in parallel
        resource_conflicts = {
            'manipulator_arm': ['pick_up_object', 'place_object', 'move_arm'],
            'navigation': ['navigate_to', 'move_to'],
            'sensor_system': ['detect_objects', 'capture_image', 'scan_environment']
        }
        
        for action in existing_group:
            for resource, action_types in resource_conflicts.items():
                if (action['action_type'] in action_types and 
                    new_action['action_type'] in action_types):
                    return False  # Conflict found
        
        return True  # No resource conflicts found
    
    def execute_group_in_parallel(self, action_group):
        """
        Execute a group of actions in parallel using threading
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(action_group)  # Pre-allocate results list
        
        # Use ThreadPoolExecutor to run actions in parallel
        with ThreadPoolExecutor(max_workers=min(len(action_group), 4)) as executor:
            # Submit all actions
            future_to_index = {
                executor.submit(self.execute_single_action_optimized, action): i 
                for i, action in enumerate(action_group)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    # Create error result for this action
                    action = action_group[index]
                    results[index] = {
                        'index': index,
                        'action_type': action['action_type'],
                        'error': str(e),
                        'status': 'failure'
                    }
        
        return results
    
    def execute_single_action_optimized(self, action):
        """
        Optimized single action execution with reduced overhead
        """
        # Get action client (cached to avoid repeated lookups)
        action_client = self.get_cached_action_client(action)
        
        if not action_client.wait_for_server(timeout_sec=1.0):
            raise RuntimeError(f"Action server not available: {action.get('action_name', 'unknown')}")
        
        goal_msg = action['goal']
        
        # Send goal asynchronously to reduce blocking time
        goal_future = action_client.send_goal_async(goal_msg)
        
        # Wait for result with timeout
        result = self.wait_for_result_optimized(goal_future, action.get('timeout', 30.0))
        
        return result
    
    def get_cached_action_client(self, action):
        """
        Get action client from cache, creating if necessary
        """
        action_name = self.derive_action_name_from_action(action)
        
        if action_name not in self.action_clients:
            # Import the action class dynamically
            action_module_path, action_class_name = action['action_type'].rsplit('.', 1)
            action_module = __import__(action_module_path, fromlist=[action_class_name])
            action_class = getattr(action_module, action_class_name)
            
            self.action_clients[action_name] = ActionClient(
                self.node, action_class, action_name
            )
        
        return self.action_clients[action_name]
    
    def derive_action_name_from_action(self, action):
        """
        Derive action server name from action definition
        """
        if 'action_name' in action:
            return action['action_name']
        
        # Use the heuristic from the generic executor
        parts = action['action_type'].split('.')
        if len(parts) >= 3:
            action_class = parts[-1]
            # Convert CamelCase to snake_case
            import re
            snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', action_class).lower()
            return f"/{snake_case}"
        
        # Fallback
        return f"/{action['action_type'].replace('.', '_').replace('action', '').lower()}"
    
    def wait_for_result_optimized(self, goal_future, timeout):
        """
        Optimized result waiting with more efficient polling
        """
        start_time = time.time()
        
        while not goal_future.done():
            # Use shorter sleep interval for more responsive execution
            # but not so short as to waste CPU cycles
            time.sleep(0.01)
            
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"Action timed out after {timeout} seconds")
        
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            raise RuntimeError('Goal was rejected by server')
        
        result_future = goal_handle.get_result_async()
        
        remaining_time = max(0.1, timeout - (time.time() - start_time))
        
        start_time = time.time()
        while not result_future.done():
            time.sleep(0.01)
            if time.time() - start_time >= remaining_time:
                raise TimeoutError(f"Action result retrieval timed out after {remaining_time} seconds")
        
        result = result_future.result().result
        return result
```

## Quality Assurance and Validation

### Action Sequence Validation

Create tools to verify the correctness of action sequences:

```python
class ActionSequenceValidator:
    def __init__(self):
        self.validation_rules = {
            'required_fields': ['action_type', 'goal'],
            'max_timeout': 300.0,  # 5 minutes max timeout
            'allowed_action_types': [
                'navigate_to', 'pick_up_object', 'place_object',
                'detect_objects', 'move_arm', 'speak', 'request_assistance'
            ]
        }
    
    def validate_action_sequence(self, sequence):
        """
        Validate an entire action sequence
        """
        validation_report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check if sequence is empty
        if not sequence:
            validation_report['valid'] = False
            validation_report['issues'].append("Action sequence is empty")
            return validation_report
        
        # Validate each action in the sequence
        for i, action in enumerate(sequence):
            action_report = self.validate_single_action(action, i)
            
            if not action_report['valid']:
                validation_report['valid'] = False
                validation_report['issues'].extend(action_report['issues'])
            
            validation_report['warnings'].extend(action_report['warnings'])
        
        # Perform sequence-level validations
        sequence_report = self.validate_sequence_properties(sequence)
        if not sequence_report['valid']:
            validation_report['valid'] = False
            validation_report['issues'].extend(sequence_report['issues'])
        
        validation_report['warnings'].extend(sequence_report['warnings'])
        
        # Add statistics
        validation_report['statistics'] = {
            'total_actions': len(sequence),
            'valid_actions': len(sequence) - len([issue for issue in validation_report['issues'] if 'Action at index' in issue]),
            'invalid_actions': len([issue for issue in validation_report['issues'] if 'Action at index' in issue])
        }
        
        return validation_report
    
    def validate_single_action(self, action, index):
        """
        Validate a single action
        """
        report = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in action:
                report['valid'] = False
                report['issues'].append(f"Action at index {index} missing required field: {field}")
        
        # Check action type validity
        if 'action_type' in action:
            if action['action_type'] not in self.validation_rules['allowed_action_types']:
                report['valid'] = False
                report['issues'].append(
                    f"Action at index {index} has invalid action type: {action['action_type']}"
                )
        
        # Check timeout value
        if 'timeout' in action:
            timeout = action['timeout']
            if timeout > self.validation_rules['max_timeout']:
                report['valid'] = False
                report['issues'].append(
                    f"Action at index {index} timeout ({timeout}s) exceeds maximum allowed ({self.validation_rules['max_timeout']}s)"
                )
            elif timeout <= 0:
                report['valid'] = False
                report['issues'].append(
                    f"Action at index {index} has invalid timeout ({timeout}s) - must be positive"
                )
        
        # Check for specific parameters based on action type
        if 'action_type' in action:
            specific_issues = self.validate_action_specific_requirements(action, index)
            if specific_issues:
                report['valid'] = False
                report['issues'].extend(specific_issues)
        
        return report
    
    def validate_action_specific_requirements(self, action, index):
        """
        Validate action-specific parameter requirements
        """
        issues = []
        action_type = action['action_type']
        
        if action_type == 'navigate_to':
            # Requires target location
            params = action.get('parameters', {})
            if 'target_location' not in params:
                issues.append(f"Action at index {index} (navigate_to) missing target_location parameter")
            else:
                target = params['target_location']
                if not all(key in target for key in ['x', 'y', 'z']):
                    issues.append(
                        f"Action at index {index} (navigate_to) target_location missing x, y, or z coordinates"
                    )
        
        elif action_type == 'pick_up_object':
            # Requires object identification
            params = action.get('parameters', {})
            if 'object_id' not in params and 'grasp_pose' not in params:
                issues.append(
                    f"Action at index {index} (pick_up_object) requires either object_id or grasp_pose parameter"
                )
        
        elif action_type == 'detect_objects':
            # Requires area of interest or object classes
            params = action.get('parameters', {})
            if 'roi' not in params and 'object_classes' not in params:
                issues.append(
                    f"Action at index {index} (detect_objects) requires either roi or object_classes parameter"
                )
        
        return issues
    
    def validate_sequence_properties(self, sequence):
        """
        Validate properties of the entire sequence
        """
        report = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for redundant consecutive actions
        prev_action = None
        for i, action in enumerate(sequence):
            if prev_action and prev_action.get('action_type') == action.get('action_type'):
                # Check if it's the same action repeated unnecessarily
                if self.are_actions_equivalent(prev_action, action):
                    report['warnings'].append(
                        f"Potential redundant action at index {i}: same action as at index {i-1}"
                    )
            prev_action = action
        
        # Check for extremely long sequences
        if len(sequence) > 50:  # Arbitrary threshold
            report['warnings'].append(
                f"Action sequence is very long ({len(sequence)} actions). Consider breaking into sub-tasks."
            )
        
        return report
    
    def are_actions_equivalent(self, action1, action2):
        """
        Determine if two actions are functionally equivalent
        """
        if action1.get('action_type') != action2.get('action_type'):
            return False
        
        # For now, compare parameters - in reality this would be more nuanced
        # For example, navigation to the same location would be equivalent
        return action1.get('parameters') == action2.get('parameters')
    
    def validate_execution_log(self, execution_results):
        """
        Validate the results of action sequence execution
        """
        validation_report = {
            'success_rate': 0,
            'failed_actions': [],
            'performance_metrics': {}
        }
        
        if not execution_results:
            return validation_report
        
        total_actions = len(execution_results)
        successful_actions = sum(1 for r in execution_results if r.get('status') == 'success')
        
        validation_report['success_rate'] = successful_actions / total_actions if total_actions > 0 else 0
        
        # Identify failed actions
        validation_report['failed_actions'] = [
            r for r in execution_results if r.get('status') == 'failure'
        ]
        
        # Calculate performance metrics
        execution_times = [
            r.get('execution_time', 0) for r in execution_results 
            if 'execution_time' in r
        ]
        
        if execution_times:
            validation_report['performance_metrics'] = {
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'total_execution_time': sum(execution_times)
            }
        
        return validation_report
```

## Summary

ROS 2 action sequences form the bridge between high-level LLM-generated plans and low-level robot execution. This chapter covered:

1. **Action Mapping**: Converting LLM concepts to specific ROS 2 action types
2. **Execution Mechanisms**: Implementing clients to execute action sequences
3. **Complex Sequences**: Handling conditional and loop-based action flows
4. **Error Handling**: Recovering from action failures in various contexts
5. **Performance Optimization**: Making action execution efficient
6. **Validation**: Ensuring action sequences are correct before execution

The integration of LLMs with ROS 2 action sequences enables the creation of sophisticated robotics systems that can interpret natural language commands and execute them as precise robot behaviors. The proper implementation of these concepts ensures that digital twins can accurately simulate the full pipeline from human instruction to robot action, making them invaluable tools for developing and testing robotic systems.

The next chapter will cover execution flow management, bringing together all elements of LLM-based cognitive planning into a cohesive system that manages the flow of robot behaviors from initial command to completion.