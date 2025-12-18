# Human-Robot Interaction Design in Unity

## Introduction

This chapter covers the design and implementation of human-robot interaction interfaces within Unity digital twins. Effective interaction design bridges the gap between human operators and simulated robots, enabling intuitive control, monitoring, and manipulation of the digital twin environment. We'll explore both direct interaction with the simulation and remote control interfaces.

## Types of Human-Robot Interaction

### Direct Interaction

Direct interaction involves manipulating the simulation environment through Unity's interface:

1. **Object Manipulation**: Moving objects in the environment
2. **Robot Control**: Directly commanding robot actions
3. **Environment Modification**: Changing the simulation environment in real-time
4. **Sensor Visualization**: Controlling and viewing sensor data feeds

### Indirect Interaction

Indirect interaction involves commanding the robot through external interfaces:

1. **API-based Control**: Sending commands via ROS 2 or other protocols
2. **Scripted Scenarios**: Running pre-defined behavior scripts
3. **Parameter Adjustment**: Modifying simulation parameters
4. **Goal Setting**: Defining high-level tasks for the robot

## Unity UI Systems for HRI

### Canvas and UI Elements

Unity's UI system allows creating interfaces for human-robot interaction:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class RobotInteractionUI : MonoBehaviour
{
    public Slider speedSlider;
    public Button[] actionButtons;
    public Text statusText;
    public Dropdown taskMenu;
    
    void Start()
    {
        // Initialize UI elements
        speedSlider.onValueChanged.AddListener(OnSpeedChanged);
        taskMenu.onValueChanged.AddListener(OnTaskSelected);
        
        // Initialize action buttons
        for (int i = 0; i < actionButtons.Length; i++)
        {
            int index = i; // Capture for closure
            actionButtons[i].onClick.AddListener(() => OnActionButtonClicked(index));
        }
    }
    
    void OnSpeedChanged(float value)
    {
        // Update robot speed in simulation
        RobotController.Instance.SetSpeed(value);
    }
    
    void OnTaskSelected(int index)
    {
        // Send task command to robot
        RobotController.Instance.SetTask(index);
    }
    
    void OnActionButtonClicked(int index)
    {
        // Execute action based on button
        RobotController.Instance.ExecuteAction(index);
    }
}
```

### Interaction Feedback Systems

Provide clear feedback for robot interactions:

1. **Visual Cues**: Highlighting interactable objects
2. **Status Indicators**: Showing robot state and capabilities
3. **Progress Bars**: Indicating execution of long actions
4. **Error Display**: Communicating when actions fail

## Creating Interactive Objects

### Raycasting for Selection

Use raycasting to detect user interactions with 3D objects:

```csharp
using UnityEngine;

public class ObjectSelector : MonoBehaviour
{
    public Camera mainCamera;
    public LayerMask interactionLayer;
    
    void Update()
    {
        if (Input.GetMouseButtonDown(0)) // Left mouse click
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit, Mathf.Infinity, interactionLayer))
            {
                // Object clicked - trigger interaction
                InteractableObject interactable = hit.collider.GetComponent<InteractableObject>();
                if (interactable != null)
                {
                    interactable.OnInteract();
                }
            }
        }
    }
}

// Base class for interactable objects
public abstract class InteractableObject : MonoBehaviour
{
    public virtual void OnInteract()
    {
        // Override in derived classes
    }
}

// Example implementation for robot parts
public class RobotPart : InteractableObject
{
    public string partName;
    public float maxTorque = 100f;
    
    public override void OnInteract()
    {
        Debug.Log($"Interacting with {partName}");
        // Trigger specific interaction based on part
        HighlightPart();
    }
    
    void HighlightPart()
    {
        // Visual feedback for selection
        GetComponent<Renderer>().material.color = Color.yellow;
        Invoke("ResetColor", 0.5f);
    }
    
    void ResetColor()
    {
        GetComponent<Renderer>().material.color = Color.white;
    }
}
```

## Control Interfaces

### Direct Robot Control

Allow users to directly control robot joints or movements:

```csharp
using UnityEngine;

public class DirectRobotController : MonoBehaviour
{
    public Transform[] joints; // Robot joint transforms
    public float[] jointAngles; // Current joint angles
    public float controlSensitivity = 1.0f;
    
    void Update()
    {
        if (Input.GetKey(KeyCode.UpArrow))
        {
            // Move specific joint
            JointControl(0, controlSensitivity * Time.deltaTime);
        }
        if (Input.GetKey(KeyCode.DownArrow))
        {
            JointControl(0, -controlSensitivity * Time.deltaTime);
        }
        
        // Apply joint positions to visual representation
        for (int i = 0; i < joints.Length; i++)
        {
            joints[i].localRotation = Quaternion.Euler(0, jointAngles[i], 0);
        }
    }
    
    void JointControl(int jointIndex, float delta)
    {
        jointAngles[jointIndex] += delta;
        // Apply joint limits
        jointAngles[jointIndex] = Mathf.Clamp(jointAngles[jointIndex], -90f, 90f);
    }
}
```

### Task-Based Control

Provide higher-level task interfaces:

```csharp
using UnityEngine;

public enum RobotTask
{
    MoveTo,
    PickUp,
    Place,
    Follow,
    Wait
}

public class TaskController : MonoBehaviour
{
    public RobotTask currentTask;
    public Vector3 targetPosition;
    public GameObject targetObject;
    
    void Update()
    {
        switch (currentTask)
        {
            case RobotTask.MoveTo:
                MoveToTarget();
                break;
            case RobotTask.PickUp:
                PickUpObject();
                break;
            case RobotTask.Place:
                PlaceObject();
                break;
            case RobotTask.Follow:
                FollowTarget();
                break;
            // Add more cases as needed
        }
    }
    
    void MoveToTarget()
    {
        // Move robot towards target position
        Vector3 direction = (targetPosition - transform.position).normalized;
        transform.position += direction * Time.deltaTime;
    }
    
    void PickUpObject()
    {
        if (targetObject != null)
        {
            // Attach object to robot's "hand"
            targetObject.transform.SetParent(transform);
            currentTask = RobotTask.Wait; // Complete task
        }
    }
    
    public void SetTask(RobotTask task, Vector3 pos, GameObject obj = null)
    {
        currentTask = task;
        targetPosition = pos;
        targetObject = obj;
    }
}
```

## Remote Operation Interfaces

### Network Communication

Implement networking for remote human-robot interaction:

```csharp
using System.Collections;
using UnityEngine;

// Example using simple TCP communication
public class RemoteController : MonoBehaviour
{
    public string serverIP = "127.0.0.1";
    public int serverPort = 12345;
    
    void Start()
    {
        StartCoroutine(ConnectToServer());
    }
    
    IEnumerator ConnectToServer()
    {
        // Implementation would connect to ROS bridge or similar
        yield return null;
    }
    
    public void SendCommand(string command)
    {
        // Send command to robot simulation
        // This is a placeholder for actual network implementation
    }
    
    public void ReceiveRobotState()
    {
        // Update UI with robot state
        // This is a placeholder for actual network implementation
    }
}
```

## Visual Feedback Systems

### Robot State Visualization

Display robot state through visual indicators:

```csharp
using UnityEngine;

public class RobotStateVisualizer : MonoBehaviour
{
    public GameObject batteryIndicator;
    public GameObject statusLight;
    public GameObject taskProgressBar;
    public RobotController robotController;
    
    void Update()
    {
        UpdateBatteryIndicator();
        UpdateStatusLight();
        UpdateTaskProgress();
    }
    
    void UpdateBatteryIndicator()
    {
        float batteryLevel = robotController.GetBatteryLevel();
        batteryIndicator.transform.localScale = new Vector3(
            batteryLevel, 
            batteryIndicator.transform.localScale.y, 
            batteryIndicator.transform.localScale.z
        );
    }
    
    void UpdateStatusLight()
    {
        Color statusColor = robotController.IsOperational() ? Color.green : Color.red;
        statusLight.GetComponent<Renderer>().material.color = statusColor;
    }
    
    void UpdateTaskProgress()
    {
        float progress = robotController.GetTaskProgress();
        taskProgressBar.transform.localScale = new Vector3(
            progress, 
            taskProgressBar.transform.localScale.y, 
            taskProgressBar.transform.localScale.z
        );
    }
}
```

### Sensor Visualization

Display sensor data in intuitive ways:

```csharp
using UnityEngine;

public class SensorVisualizer : MonoBehaviour
{
    public LineRenderer lidarVisualizer;
    public Camera depthCamera;
    public GameObject[] proximitySensors;
    
    void Update()
    {
        VisualizeLidarData();
        VisualizeDepthCamera();
        UpdateProximitySensors();
    }
    
    void VisualizeLidarData()
    {
        // Update lidar visualization based on simulated data
        // Points would come from the simulated LiDAR sensor
        Vector3[] points = new Vector3[360]; // 360 degree scan
        for (int i = 0; i < points.Length; i++)
        {
            float angle = i * Mathf.Deg2Rad;
            float distance = GetLidarDistance(i); // Simulated data
            points[i] = new Vector3(
                Mathf.Cos(angle) * distance,
                0,
                Mathf.Sin(angle) * distance
            );
        }
        
        lidarVisualizer.positionCount = points.Length;
        lidarVisualizer.SetPositions(points);
    }
    
    float GetLidarDistance(int angleIndex)
    {
        // Placeholder for actual LiDAR simulation data
        return 5.0f; // Fixed distance for example
    }
    
    void VisualizeDepthCamera()
    {
        // Visualize depth camera information
        // Could be as an overlay or in a separate window
    }
    
    void UpdateProximitySensors()
    {
        foreach (GameObject sensor in proximitySensors)
        {
            // Change color based on proximity to obstacles
            float distance = GetProximityDistance(sensor);
            Color sensorColor = distance < 1.0f ? Color.red : Color.green;
            sensor.GetComponent<Renderer>().material.color = sensorColor;
        }
    }
    
    float GetProximityDistance(GameObject sensor)
    {
        // Placeholder for proximity calculation
        return 2.0f;
    }
}
```

## Safety and Error Handling

### Interaction Safety

Implement safety measures for human-robot interactions:

1. **Safe Zones**: Prevent interactions in dangerous areas
2. **Physical Limits**: Respect joint and actuator limits
3. **Collision Avoidance**: Prevent robot from colliding with important objects
4. **Emergency Stop**: Immediate halt of robot motion

### Error Visualization

Communicate errors clearly to the user:

```csharp
using UnityEngine;

public class ErrorDisplayer : MonoBehaviour
{
    public GameObject errorPanel;
    public Text errorText;
    
    public void ShowError(string errorMessage)
    {
        errorPanel.SetActive(true);
        errorText.text = errorMessage;
        
        // Automatically hide after delay
        Invoke("HideError", 5f);
    }
    
    void HideError()
    {
        errorPanel.SetActive(false);
    }
    
    public void LogError(string context, string error)
    {
        Debug.LogError($"[{context}] {error}");
        ShowError($"{context}: {error}");
    }
}
```

## Design Patterns for HRI

### Command Pattern

Use the command pattern for undoable robot actions:

```csharp
// Command base class
public abstract class RobotCommand
{
    public abstract void Execute();
    public abstract void Undo();
}

// Move command
public class MoveCommand : RobotCommand
{
    private RobotController robot;
    private Vector3 startPosition;
    private Vector3 endPosition;
    
    public MoveCommand(RobotController robot, Vector3 targetPosition)
    {
        this.robot = robot;
        this.startPosition = robot.transform.position;
        this.endPosition = targetPosition;
    }
    
    public override void Execute()
    {
        robot.MoveTo(endPosition);
    }
    
    public override void Undo()
    {
        robot.MoveTo(startPosition);
    }
}

// Command manager
public class CommandManager : MonoBehaviour
{
    private Stack<RobotCommand> commandHistory = new Stack<RobotCommand>();
    
    public void ExecuteCommand(RobotCommand command)
    {
        command.Execute();
        commandHistory.Push(command);
    }
    
    public void UndoLastCommand()
    {
        if (commandHistory.Count > 0)
        {
            RobotCommand lastCommand = commandHistory.Pop();
            lastCommand.Undo();
        }
    }
}
```

### State Machine for Robot Behavior

Implement behavior using state machines:

```csharp
public enum RobotState
{
    Idle,
    Moving,
    Manipulating,
    Waiting,
    Error
}

public class RobotStateMachine : MonoBehaviour
{
    public RobotState currentState = RobotState.Idle;
    private RobotController robotController;
    
    void Update()
    {
        switch (currentState)
        {
            case RobotState.Idle:
                HandleIdleState();
                break;
            case RobotState.Moving:
                HandleMovingState();
                break;
            case RobotState.Manipulating:
                HandleManipulatingState();
                break;
            case RobotState.Waiting:
                HandleWaitingState();
                break;
            case RobotState.Error:
                HandleErrorState();
                break;
        }
    }
    
    void HandleIdleState()
    {
        // Robot is waiting for commands
        // Maybe do light animations
    }
    
    void HandleMovingState()
    {
        // Move robot based on target
        if (robotController.HasReachedTarget())
        {
            currentState = RobotState.Idle;
        }
    }
    
    void HandleManipulatingState()
    {
        // Execute manipulation task
        if (robotController.IsTaskComplete())
        {
            currentState = RobotState.Idle;
        }
    }
    
    void HandleWaitingState()
    {
        // Wait for external signal or timeout
    }
    
    void HandleErrorState()
    {
        // Handle error state - maybe emergency stop
    }
    
    public void SetState(RobotState newState)
    {
        currentState = newState;
    }
}
```

## Quality Assurance for HRI

### Usability Testing

Test interfaces with real users:

1. **Task Completion Rate**: Percentage of tasks completed successfully
2. **Time to Completion**: How long it takes to perform tasks
3. **Error Rate**: Frequency of incorrect commands or actions
4. **User Satisfaction**: Subjective measures of interface quality

### Performance Metrics

1. **Response Time**: How quickly the system responds to inputs
2. **Frame Rate Impact**: How HRI interfaces affect visualization performance
3. **System Resource Usage**: Memory and CPU usage of HRI systems

## Summary

Human-robot interaction in Unity digital twins requires careful design of both direct and indirect interaction methods. By implementing proper UI systems, feedback mechanisms, and safety measures, we can create intuitive and effective interfaces that allow humans to operate and monitor simulated robots effectively.

The next chapter will cover synchronization mechanisms between Gazebo and Unity, which is crucial for maintaining consistency between the physics simulation and the visual representation.