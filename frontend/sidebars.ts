import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'modules/robotic-nervous-system/intro',
        'modules/robotic-nervous-system/ros2-fundamentals',
        'modules/robotic-nervous-system/python-nodes-rclpy',
        'modules/robotic-nervous-system/urdf-modeling'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/index',
        {
          type: 'category',
          label: 'Physics-Based Simulation with Gazebo',
          items: [
            'module-2-digital-twin/physics-simulation/setup-and-configuration',
            'module-2-digital-twin/physics-simulation/gravity-collisions-dynamics',
            'module-2-digital-twin/physics-simulation/environment-modeling',
            'module-2-digital-twin/physics-simulation/humanoid-examples'
          ],
        },
        {
          type: 'category',
          label: 'High-Fidelity Interaction with Unity',
          items: [
            'module-2-digital-twin/unity-visualization/visual-realism',
            'module-2-digital-twin/unity-visualization/human-robot-interaction',
            'module-2-digital-twin/unity-visualization/synchronization',
            'module-2-digital-twin/unity-visualization/unity-gazebo-integration'
          ],
        },
        {
          type: 'category',
          label: 'Sensor Simulation for Perception',
          items: [
            'module-2-digital-twin/sensor-simulation/lidar-simulation',
            'module-2-digital-twin/sensor-simulation/depth-camera-simulation',
            'module-2-digital-twin/sensor-simulation/imu-simulation',
            'module-2-digital-twin/sensor-simulation/data-streams-ai-pipelines',
            'module-2-digital-twin/sensor-simulation/perception-accuracy'
          ],
        }
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain',
      items: [
        'modules/ai-robot-brain/photorealistic-simulation',
        'modules/ai-robot-brain/hardware-accelerated-perception',
        'modules/ai-robot-brain/path-planning-nav2'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/index',
        {
          type: 'category',
          label: 'Voice-to-Action Interfaces',
          items: [
            'module-4-vla/voice-interfaces/speech-recognition',
            'module-4-vla/voice-interfaces/intent-mapping',
            'module-4-vla/voice-interfaces/voice-robot-interaction',
            'module-4-vla/voice-interfaces/index'
          ],
        },
        {
          type: 'category',
          label: 'Cognitive Planning with LLMs',
          items: [
            'module-4-vla/llm-planning/llm-fundamentals',
            'module-4-vla/llm-planning/task-decomposition',
            'module-4-vla/llm-planning/action-sequences',
            'module-4-vla/llm-planning/planning-execution'
          ],
        },
        {
          type: 'category',
          label: 'End-to-End Autonomous Humanoid',
          items: [
            'module-4-vla/autonomous-capstone/vla-architecture',
            'module-4-vla/autonomous-capstone/perception-navigation',
            'module-4-vla/autonomous-capstone/manipulation-control',
            'module-4-vla/autonomous-capstone/full-workflow'
          ],
        }
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
