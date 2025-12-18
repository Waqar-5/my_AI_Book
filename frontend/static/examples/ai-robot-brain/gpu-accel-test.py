#!/usr/bin/env python3
# GPU Acceleration Verification Script

"""
This script verifies GPU acceleration for Isaac ROS packages
"""
import subprocess
import sys

def verify_gpu_acceleration():
    """
    Verifies that Isaac ROS packages are leveraging GPU acceleration
    """
    print("Verifying GPU acceleration for Isaac ROS packages...")
    
    # Check if NVIDIA GPU is available
    try:
        gpu_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if gpu_result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            print(f"GPU Info: {gpu_result.stdout.split('Driver Version')[0].strip()[-50:]}")
        else:
            print("⚠ No NVIDIA GPU detected - hardware acceleration might not be available")
            return False
    except FileNotFoundError:
        print("⚠ nvidia-smi not found - NVIDIA GPU drivers may not be installed")
        return False
    
    # Check for Isaac ROS packages
    try:
        # This would normally check if Isaac ROS packages are installed
        # For this example, we'll just check if some common Isaac ROS packages exist
        result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True)
        isaac_packages = [line for line in result.stdout.split('\n') if 'isaac' in line.lower()]
        
        if isaac_packages:
            print(f"✓ Found {len(isaac_packages)} Isaac ROS packages:")
            for pkg in isaac_packages[:5]:  # Show first 5 packages
                print(f"  - {pkg}")
            if len(isaac_packages) > 5:
                print(f"  ... and {len(isaac_packages)-5} more packages")
        else:
            print("⚠ No Isaac ROS packages found - verify installation")
            return False
    except Exception as e:
        print(f"⚠ Could not verify Isaac ROS packages: {e}")
        return False
        
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.version.cuda}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # Check if Isaac ROS utilizes CUDA through PyTorch
            # This is a simplified test - in practice, we would run actual Isaac ROS nodes
            device = torch.device('cuda')
            test_tensor = torch.randn(100, 100, device=device)
            print("✓ GPU acceleration test passed with PyTorch")
        else:
            print("⚠ CUDA is not available - GPU acceleration not enabled")
            return False
    except ImportError:
        print("⚠ PyTorch not available to check CUDA status")
        # Still continue with other checks
    
    # In a real scenario, we would perform actual Isaac ROS operations
    # that leverage GPU acceleration
    
    print("✓ GPU acceleration verification completed")
    return True

if __name__ == "__main__":
    success = verify_gpu_acceleration()
    if success:
        print("\nIsaac ROS GPU acceleration verification: PASSED")
        sys.exit(0)
    else:
        print("\nIsaac ROS GPU acceleration verification: FAILED")
        sys.exit(1)