# Sensor Simulation Example

This directory contains a minimal runnable example demonstrating sensor simulation in digital twins.

## Files Included

- `sensor_config.yaml`: Configuration for LiDAR, camera, and IMU sensors
- `sensor_processor.py`: Python script to process and visualize sensor data
- `validation_metrics.py`: Implementation of perception accuracy metrics
- `README.md`: This file

## How to Run

1. Set up your Gazebo simulation environment with the sensor configuration
2. Launch the Gazebo simulation with the configured sensors
3. Run the sensor processor to collect and process simulated data
4. Evaluate the data using the validation metrics

## Purpose

This example demonstrates:
- Configuration of multiple sensor types in Gazebo
- Processing of simulated sensor data streams
- Basic perception algorithms applied to simulated data
- Validation of perception accuracy against ground truth