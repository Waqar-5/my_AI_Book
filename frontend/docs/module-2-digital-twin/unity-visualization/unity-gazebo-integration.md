# Unity-Gazebo Integration Patterns

## Introduction

This chapter explores various integration patterns for connecting Unity visualization with Gazebo physics simulation. Effective integration ensures seamless operation between these two systems, enabling the creation of compelling digital twins. We'll examine different architectural approaches, communication protocols, and implementation strategies that balance performance, accuracy, and maintainability.

## Integration Architecture Patterns

### 1. Bridge-Based Integration

The most common pattern uses a dedicated bridge application to relay information between Gazebo and Unity:

```
Gazebo Simulation → Bridge Application → Unity Visualization
```

**Advantages:**
- Clear separation of concerns
- Independent scaling of each component
- Easier debugging and maintenance
- Supports multiple visualization clients

**Disadvantages:**
- Additional point of failure
- Potential latency from extra network hop
- Increased system complexity

**Implementation Example:**

```csharp
// Bridge server in C# (for handling Gazebo to Unity communication)
using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class GazeboUnityBridge
{
    private TcpListener server;
    private bool isRunning;
    
    // Store latest states received from Gazebo
    private object stateLock = new object();
    private RobotState latestRobotState;
    
    public async Task StartAsync(int port)
    {
        server = new TcpListener(System.Net.IPAddress.Any, port);
        server.Start();
        isRunning = true;
        
        Console.WriteLine($"Bridge server started on port {port}");
        
        while (isRunning)
        {
            try
            {
                TcpClient client = await server.AcceptTcpClientAsync();
                _ = Task.Run(() => HandleClient(client));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error accepting client: {ex.Message}");
            }
        }
    }
    
    private async Task HandleClient(TcpClient client)
    {
        using (var stream = client.GetStream())
        {
            try
            {
                // Send latest state to new client
                SendLatestState(stream);
                
                // Continue updating client with new states
                while (client.Connected)
                {
                    // Check for new states periodically
                    await Task.Delay(16); // ~60 FPS
                    SendStateIfUpdated(stream);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Client handling error: {ex.Message}");
            }
        }
    }
    
    public void UpdateRobotState(RobotState newState)
    {
        lock (stateLock)
        {
            latestRobotState = newState;
        }
    }
    
    private void SendLatestState(NetworkStream stream)
    {
        lock (stateLock)
        {
            if (latestRobotState.timestamp > 0)
            {
                string json = JsonConvert.SerializeObject(latestRobotState);
                byte[] data = System.Text.Encoding.UTF8.GetBytes(json + "\n");
                stream.Write(data, 0, data.Length);
            }
        }
    }
    
    private void SendStateIfUpdated(NetworkStream stream)
    {
        // Implementation would send state updates to connected Unity clients
    }
}

public struct RobotState
{
    public float timestamp;
    public Vector3 position;
    public Quaternion rotation;
    public float[] jointPositions;
    public float[] jointVelocities;
}
```

### 2. Direct Integration (ROS 2 Bridge)

Using ROS 2's native bridge for direct communication:

```
Unity ←→ rosbridge_websocket ←→ ROS 2 Middleware ←→ Gazebo Plugins
```

**Advantages:**
- Leverages existing ROS 2 ecosystem
- Standardized communication patterns
- Broad community support
- Compatible with other ROS 2 tools

**Disadvantages:**
- Additional ROS 2 dependency
- Potential configuration complexity
- Network communication overhead

**Implementation Example:**

```xml
<!-- Gazebo plugin for ROS 2 bridge -->
<gazebo>
  <plugin filename="libgazebo_ros_p3d.so" name="gazebo_ros_p3d">
    <ros>
      <namespace>/humanoid_robot</namespace>
      <remapping>odom:=ground_truth/odom</remapping>
    </ros>
    <frame_name>map</frame_name>
    <body_name>base_link</body_name>
    <update_rate>60</update_rate>
    <always_on>true</always_on>
    <gaussian_noise>0.0</gaussian_noise>
  </plugin>
</gazebo>
```

```csharp
// Unity side using ROS 2 bridge
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Nav;
using RosMessageTypes.Geometry;

public class RobotStateSubscriber : MonoBehaviour
{
    private ROSConnection ros;
    private OdomData latestOdomData; // Custom struct for odometry data
    private bool hasNewData = false;
    
    [System.Serializable]
    public struct OdomData
    {
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 linearVelocity;
        public Vector3 angularVelocity;
    }
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        
        // Subscribe to the odometry topic published by Gazebo
        ros.Subscribe<Odometry>("/humanoid_robot/ground_truth/odom", UpdateRobotState);
        
        // Also subscribe to joint states
        ros.Subscribe<sensor_msgs.JointState>("/humanoid_robot/joint_states", UpdateJointState);
    }
    
    void UpdateRobotState(Odometry msg)
    {
        latestOdomData.position = new Vector3(
            (float)msg.pose.pose.position.x,
            (float)msg.pose.pose.position.z,  // Y and Z swap for Unity
            (float)msg.pose.pose.position.y
        );
        
        latestOdomData.rotation = new Quaternion(
            (float)msg.pose.pose.orientation.x,
            (float)msg.pose.pose.orientation.z,
            (float)msg.pose.pose.orientation.y,
            (float)msg.pose.pose.orientation.w
        );
        
        latestOdomData.linearVelocity = new Vector3(
            (float)msg.twist.twist.linear.x,
            (float)msg.twist.twist.linear.z,
            (float)msg.twist.twist.linear.y
        );
        
        latestOdomData.angularVelocity = new Vector3(
            (float)msg.twist.twist.angular.x,
            (float)msg.twist.twist.angular.z,
            (float)msg.twist.twist.angular.y
        );
        
        hasNewData = true;
    }
    
    void UpdateJointState(sensor_msgs.JointState msg)
    {
        // Update joint positions received from Gazebo
        // This would update the visual representation of the robot
    }
    
    void Update()
    {
        if (hasNewData)
        {
            // Apply the latest odometry data to the robot transform
            ApplyRobotState(latestOdomData);
            hasNewData = false;
        }
    }
    
    void ApplyRobotState(OdomData data)
    {
        transform.position = data.position;
        transform.rotation = data.rotation;
    }
}
```

### 3. File-Based Integration

For scenarios requiring offline processing or simplified deployment:

```
Gazebo Simulation → State Log Files → Unity Playback System
```

**Advantages:**
- No network dependencies
- Repeatable scenarios
- Easy to integrate into existing build systems

**Disadvantages:**
- No real-time interaction
- Large file sizes for detailed simulations
- Requires pre-processing step

## Communication Protocols

### WebSocket Communication

WebSockets provide bidirectional communication with low overhead:

```csharp
using System;
using System.Threading.Tasks;
using Newtonsoft.Json;
using WebSocketSharp;

public class UnityWebSocketClient
{
    private WebSocket ws;
    private bool isConnected = false;
    
    public async Task ConnectAsync(string uri)
    {
        ws = new WebSocket(uri);
        
        ws.OnOpen += (sender, e) => {
            isConnected = true;
            Console.WriteLine("Connected to server");
        };
        
        ws.OnMessage += (sender, e) => {
            ProcessMessage(e.Data);
        };
        
        ws.OnClose += (sender, e) => {
            isConnected = false;
            Console.WriteLine($"Disconnected: {e.Reason}");
        };
        
        ws.Connect();
    }
    
    public void SendState(RobotState state)
    {
        if (isConnected)
        {
            string json = JsonConvert.SerializeObject(state);
            ws.Send(json);
        }
    }
    
    private void ProcessMessage(string message)
    {
        try
        {
            RobotState state = JsonConvert.DeserializeObject<RobotState>(message);
            UpdateVisualization(state);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error processing message: {ex.Message}");
        }
    }
    
    private void UpdateVisualization(RobotState state)
    {
        // Update Unity visualization based on received state
    }
}
```

### UDP Broadcasting

For real-time scenarios with minimal latency:

```csharp
using System.Net;
using System.Net.Sockets;
using System.Text;

public class UDPBroadcaster
{
    private UdpClient udpClient;
    private IPEndPoint endPoint;
    private bool isRunning = false;
    
    public UDPBroadcaster(string ip, int port)
    {
        udpClient = new UdpClient();
        endPoint = new IPEndPoint(IPAddress.Parse(ip), port);
    }
    
    public void SendState(RobotState state)
    {
        string json = JsonConvert.SerializeObject(state);
        byte[] data = Encoding.UTF8.GetBytes(json);
        udpClient.Send(data, data.Length, endPoint);
    }
    
    public void StartReceiving(int port)
    {
        UdpClient receiveClient = new UdpClient(port);
        isRunning = true;
        
        Task.Run(async () =>
        {
            while (isRunning)
            {
                try
                {
                    UdpReceiveResult result = await receiveClient.ReceiveAsync();
                    string message = Encoding.UTF8.GetString(result.Buffer);
                    RobotState state = JsonConvert.DeserializeObject<RobotState>(message);
                    UpdateVisualization(state);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Receive error: {ex.Message}");
                }
            }
        });
    }
    
    private void UpdateVisualization(RobotState state)
    {
        // Update Unity visualization based on received state
    }
    
    public void Close()
    {
        isRunning = false;
        udpClient?.Close();
    }
}
```

## Data Mapping and Transformation

### Coordinate System Conversion

Unity and Gazebo use different coordinate systems:

- **Gazebo**: X forward, Y left, Z up (right-handed)
- **Unity**: X right, Y up, Z forward (left-handed)

```csharp
public static class CoordinateConverter
{
    // Convert from Gazebo coordinate system to Unity
    public static Vector3 GazeboToUnityPosition(Vector3 gazeboPos)
    {
        return new Vector3(
            gazeboPos.z,  // Gazebo X → Unity Z
            gazeboPos.y,  // Gazebo Y → Unity Y  
            gazeboPos.x   // Gazebo Z → Unity X
        );
    }
    
    // Convert from Unity coordinate system to Gazebo
    public static Vector3 UnityToGazeboPosition(Vector3 unityPos)
    {
        return new Vector3(
            unityPos.z,  // Unity Z → Gazebo X
            unityPos.x,  // Unity X → Gazebo Y
            unityPos.y   // Unity Y → Gazebo Z
        );
    }
    
    // Convert rotation quaternions
    public static Quaternion GazeboToUnityRotation(Quaternion gazeboRot)
    {
        // Convert from Gazebo coordinate system to Unity
        return new Quaternion(
            gazeboRot.z,  // i component
            gazeboRot.x,  // j component  
            gazeboRot.y,  // k component
            gazeboRot.w   // w component
        );
    }
    
    // Convert Euler angles
    public static Vector3 GazeboToUnityEuler(Vector3 gazeboEuler)
    {
        // Convert from Gazebo Euler angles to Unity Euler angles
        return new Vector3(
            gazeboEuler.z * Mathf.Rad2Deg,  // Roll (X) - convert from radians
            gazeboEuler.x * Mathf.Rad2Deg,  // Pitch (Y)
            gazeboEuler.y * Mathf.Rad2Deg   // Yaw (Z)
        );
    }
}
```

### Model and Joint Mapping

Create a mapping system between Gazebo models and Unity visual representations:

```csharp
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class JointMapping
{
    public string gazeboName;      // Name in Gazebo
    public string unityName;       // Name of GameObject in Unity
    public Transform unityTransform; // Reference to Unity transform
    public int unityIndex;         // Index in joints array
}

public class RobotModelMapper : MonoBehaviour
{
    public JointMapping[] jointMappings;
    public Dictionary<string, JointMapping> gazeboToUnityMap;
    
    void Start()
    {
        BuildMappingDictionary();
    }
    
    void BuildMappingDictionary()
    {
        gazeboToUnityMap = new Dictionary<string, JointMapping>();
        foreach (JointMapping mapping in jointMappings)
        {
            if (mapping.unityTransform == null)
            {
                mapping.unityTransform = transform.Find(mapping.unityName)?.GetComponent<Transform>();
            }
            gazeboToUnityMap[mapping.gazeboName] = mapping;
        }
    }
    
    public void UpdateJointPositions(Dictionary<string, float> jointPositions)
    {
        foreach (var jointPair in jointPositions)
        {
            if (gazeboToUnityMap.ContainsKey(jointPair.Key))
            {
                JointMapping mapping = gazeboToUnityMap[jointPair.Key];
                if (mapping.unityTransform != null)
                {
                    // Apply rotation based on joint position (in radians)
                    mapping.unityTransform.localRotation = 
                        Quaternion.Euler(0, jointPair.Value * Mathf.Rad2Deg, 0);
                }
            }
        }
    }
    
    public Dictionary<string, float> GetJointPositions()
    {
        Dictionary<string, float> positions = new Dictionary<string, float>();
        
        foreach (var mapping in gazeboToUnityMap)
        {
            if (mapping.Value.unityTransform != null)
            {
                // Convert Unity rotation back to radians for Gazebo
                float angleInRadians = mapping.Value.unityTransform.localEulerAngles.y * Mathf.Deg2Rad;
                positions[mapping.Key] = angleInRadians;
            }
        }
        
        return positions;
    }
}
```

## Performance Optimization

### Efficient Data Transmission

Optimize the data being transmitted between systems:

```csharp
// Efficient state structure
public struct CompactRobotState
{
    // Time stamps
    public uint sequenceId;      // Frame sequence number
    public uint timestampMs;     // Milliseconds since start
    
    // Position and orientation (using smaller types)
    public Vector3Int position;  // Fixed-point position
    public byte[] orientation;   // Compressed quaternion (128 bits)
    
    // Joint positions (quantized)
    public ushort[] jointPositions;  // Quantized to 16-bit
    
    // Convert to standard format
    public RobotState ToRobotState()
    {
        RobotState state = new RobotState();
        state.timestamp = timestampMs / 1000.0f;
        state.position = new Vector3(
            position.x / 1000.0f,  // Scale back from millimeters
            position.y / 1000.0f,
            position.z / 1000.0f
        );
        
        // Decompress quaternion
        state.rotation = DecompressQuaternion(orientation);
        
        // Convert joint positions from quantized values
        state.jointPositions = new float[jointPositions.Length];
        for (int i = 0; i < jointPositions.Length; i++)
        {
            // Convert from 0-65535 back to meaningful joint range
            state.jointPositions[i] = MapRange(jointPositions[i], 0, 65535, -Mathf.PI, Mathf.PI);
        }
        
        return state;
    }
    
    private float MapRange(ushort value, ushort inMin, ushort inMax, float outMin, float outMax)
    {
        return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    }
    
    private Quaternion DecompressQuaternion(byte[] compressed)
    {
        // Implementation depends on compression method
        // This is a placeholder
        return Quaternion.identity;
    }
}
```

### Level of Detail (LOD) for Integration

```csharp
public class IntegrationLOD : MonoBehaviour
{
    public float highDetailDistance = 5f;
    public float lowDetailDistance = 20f;
    
    private UnityWebSocketClient highDetailClient;
    private UnityWebSocketClient lowDetailClient;
    private UnityWebSocketClient minimalClient;
    
    void Start()
    {
        // Initialize different clients with different update rates
        highDetailClient = new UnityWebSocketClient(); // Update at 60Hz
        lowDetailClient = new UnityWebSocketClient();  // Update at 30Hz
        minimalClient = new UnityWebSocketClient();    // Update at 10Hz
    }
    
    void Update()
    {
        float distanceToRobot = Vector3.Distance(transform.position, Camera.main.transform.position);
        
        if (distanceToRobot <= highDetailDistance)
        {
            // High detail synchronization
            SendDetailedState(highDetailClient);
        }
        else if (distanceToRobot <= lowDetailDistance)
        {
            // Medium detail synchronization
            SendMediumState(lowDetailClient);
        }
        else
        {
            // Minimal detail synchronization
            SendMinimalState(minimalClient);
        }
    }
    
    void SendDetailedState(UnityWebSocketClient client)
    {
        // Send full state with all joint positions, velocities, etc.
        RobotState state = GetFullRobotState();
        client.SendState(state);
    }
    
    void SendMediumState(UnityWebSocketClient client)
    {
        // Send reduced state information
        RobotState state = GetPositionOnlyState();
        client.SendState(state);
    }
    
    void SendMinimalState(UnityWebSocketClient client)
    {
        // Send only essential information
        RobotState state = GetPositionAndOrientationOnlyState();
        client.SendState(state);
    }
}
```

## Quality Assurance and Validation

### Integration Validation Tools

Create tools to validate the integration:

```csharp
public class IntegrationValidator : MonoBehaviour
{
    public Transform gazeboTransform;  // Ground truth from Gazebo
    public Transform unityTransform;   // Visual representation in Unity
    
    [Header("Validation Parameters")]
    public float maxPositionError = 0.01f;  // 1cm tolerance
    public float maxRotationError = 0.5f;   // 0.5 degree tolerance
    
    [Header("Validation Results")]
    public float currentPositionError;
    public float currentRotationError;
    public bool isSynchronized = true;
    
    private float maxObservedError = 0f;
    private float averageError = 0f;
    private int errorSamples = 0;
    
    void Update()
    {
        ValidateSynchronization();
        CalculateStatistics();
    }
    
    void ValidateSynchronization()
    {
        // Calculate position error
        currentPositionError = Vector3.Distance(
            gazeboTransform.position, 
            unityTransform.position
        );
        
        // Calculate rotation error
        currentRotationError = Quaternion.Angle(
            gazeboTransform.rotation, 
            unityTransform.rotation
        );
        
        // Determine if synchronized
        isSynchronized = 
            currentPositionError <= maxPositionError && 
            currentRotationError <= maxRotationError;
        
        if (!isSynchronized)
        {
            Debug.LogWarning($"Desynchronization detected: pos={currentPositionError:F4}, rot={currentRotationError:F4}");
        }
        
        // Track statistics
        maxObservedError = Mathf.Max(maxObservedError, currentPositionError);
        averageError = (averageError * errorSamples + currentPositionError) / (errorSamples + 1);
        errorSamples++;
    }
    
    void CalculateStatistics()
    {
        // Update display of statistics
    }
    
    public void ReportIntegrationQuality()
    {
        Debug.Log($"Integration Quality Report:\n" +
                  $"Max Position Error: {maxObservedError:F4}m\n" +
                  $"Average Position Error: {averageError:F4}m\n" +
                  $"Current Status: {(isSynchronized ? "SYNCHRONIZED" : "DESYNCHRONIZED")}");
    }
    
    // Visualization of errors
    void OnDrawGizmos()
    {
        if (!isSynchronized)
        {
            Gizmos.color = Color.red;
        }
        else if (currentPositionError > maxPositionError * 0.5f)
        {
            Gizmos.color = Color.yellow;
        }
        else
        {
            Gizmos.color = Color.green;
        }
        
        Gizmos.DrawLine(gazeboTransform.position, unityTransform.position);
    }
}
```

## Deployment Strategies

### Container-Based Deployment

For consistent deployments across different environments:

```dockerfile
# Unity visualization container
FROM unityci/base:ubuntu-2021.3.17f1

WORKDIR /app

# Copy Unity project
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-rosbridge-suite \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Build Unity application
RUN /opt/unity/Editor/Unity \
    -batchmode \
    -nographics \
    -buildTarget StandaloneLinux64 \
    -projectPath /app \
    -executeMethod BuildScript.BuildLinux64 \
    -quit

# Expose port for WebSocket communication
EXPOSE 9090

# Run the Unity application
ENTRYPOINT ["/app/build/UnityVisualization.x86_64"]
```

### Client-Server Architecture

For scenarios with multiple viewers:

```csharp
// Server component for multi-client support
public class MultiClientServer
{
    private List<WebSocket> clients = new List<WebSocket>();
    private readonly object lockObject = new object();
    
    public void AddClient(WebSocket client)
    {
        lock (lockObject)
        {
            clients.Add(client);
            
            // Send current state to new client
            SendCurrentState(client);
        }
    }
    
    public void RemoveClient(WebSocket client)
    {
        lock (lockObject)
        {
            clients.Remove(client);
        }
    }
    
    public void BroadcastState(RobotState state)
    {
        lock (lockObject)
        {
            List<WebSocket> disconnectedClients = new List<WebSocket>();
            
            foreach (WebSocket client in clients)
            {
                if (client.ReadyState == WebSocketState.Open)
                {
                    try
                    {
                        string json = JsonConvert.SerializeObject(state);
                        client.Send(json);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"Failed to send to client: {ex.Message}");
                        disconnectedClients.Add(client);
                    }
                }
                else
                {
                    disconnectedClients.Add(client);
                }
            }
            
            // Remove disconnected clients
            foreach (WebSocket disconnected in disconnectedClients)
            {
                clients.Remove(disconnected);
            }
        }
    }
}
```

## Best Practices

### 1. Fail-Safe Mechanisms

```csharp
public class IntegrationFailureHandler : MonoBehaviour
{
    public float connectionTimeout = 5.0f;
    private float lastSuccessfulUpdate = 0f;
    private bool isConnectionLost = false;
    
    void Update()
    {
        if (Time.time - lastSuccessfulUpdate > connectionTimeout)
        {
            if (!isConnectionLost)
            {
                HandleConnectionLoss();
                isConnectionLost = true;
            }
        }
        else
        {
            if (isConnectionLost)
            {
                HandleConnectionRecovery();
                isConnectionLost = false;
            }
        }
    }
    
    void HandleConnectionLoss()
    {
        Debug.LogWarning("Connection to Gazebo lost. Activating safe mode.");
        // Switch to local simulation or hold position
    }
    
    void HandleConnectionRecovery()
    {
        Debug.Log("Connection to Gazebo restored.");
        // Resynchronize with Gazebo state
    }
}
```

### 2. Logging and Monitoring

```csharp
public class IntegrationLogger : MonoBehaviour
{
    private Queue<string> logBuffer = new Queue<string>();
    private const int MAX_LOG_ENTRIES = 100;
    
    public void LogIntegrationEvent(string category, string message)
    {
        string logEntry = $"[{Time.time:F3}] [{category}] {message}";
        logBuffer.Enqueue(logEntry);
        
        if (logBuffer.Count > MAX_LOG_ENTRIES)
        {
            logBuffer.Dequeue();
        }
        
        Debug.Log(logEntry);
    }
    
    public string GetLogSummary()
    {
        return string.Join("\n", logBuffer.ToArray());
    }
    
    void OnApplicationQuit()
    {
        // Save log to file
        string logPath = Path.Combine(Application.persistentDataPath, "integration_log.txt");
        File.WriteAllText(logPath, GetLogSummary());
    }
}
```

## Summary

Unity-Gazebo integration requires careful consideration of architecture, communication protocols, coordinate systems, and performance optimization. The right integration pattern depends on your specific use case, performance requirements, and deployment constraints.

By implementing proper validation tools and following best practices for fault tolerance and monitoring, you can create robust digital twin systems that provide accurate, real-time visualization of physics simulations. The integration patterns covered in this chapter provide a solid foundation for building reliable Unity-Gazebo digital twins.

The next chapter will address sensor simulation for perception, which is an essential component that works alongside the visualization systems covered in this module.