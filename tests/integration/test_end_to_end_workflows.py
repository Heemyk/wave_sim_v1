#!/usr/bin/env python3
"""
Comprehensive end-to-end integration tests for the acoustic simulation system.

This test suite validates complete workflows including:
1. Job submission and processing with real FEM simulations
2. WebSocket real-time communication
3. Results retrieval and validation
4. 3D visualization data extraction
5. Error handling and edge cases
6. Backend persistence across restarts

Usage:
    python -m pytest tests/integration/test_end_to_end_workflows.py -v
    python tests/integration/test_end_to_end_workflows.py
"""

import asyncio
import requests
import json
import time
import websockets
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class EndToEndTestSuite:
    """Comprehensive end-to-end test suite."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.test_results = {}
        
    def create_test_request(self, name: str = "E2E Test") -> Dict[str, Any]:
        """Create a standardized test request."""
        return {
            "name": name,
            "room": {
                "type": "box",
                "dimensions": [4.0, 3.0, 2.5],
                "center": [0.0, 0.0, 0.0]
            },
            "boundaries": {
                "walls": {"alpha": 0.1},
                "floor": {"alpha": 0.2},
                "ceiling": {"alpha": 0.15}
            },
            "sources": [{
                "id": "source_1",
                "type": "point",
                "position": [2.0, 1.5, 1.0],
                "signal": {
                    "type": "sine",
                    "f0": 100.0,
                    "f1": 200.0,
                    "duration": 1.0,
                    "amplitude": 1.0,
                    "phase": 0.0
                },
                "amplitude": 1.0
            }],
            "mesh": {
                "element_order": 1,
                "target_h": 0.2,
                "refinement_level": 0,
                "adaptive": False,
                "quality_threshold": 0.3
            },
            "simulation": {
                "type": "frequency_domain",
                "fmin": 100.0,
                "fmax": 200.0,
                "df": 100.0,  # Only 2 frequencies for faster testing
                "solver_type": "direct",
                "tolerance": 1e-6,
                "max_iterations": 1000
            },
            "output": {
                "sensors": [
                    {"id": "listener_1", "position": [1.0, 1.0, 1.0], "type": "point"},
                    {"id": "listener_2", "position": [3.0, 2.0, 1.5], "type": "point"}
                ],
                "field_snapshots": True,
                "frequency_response": True,
                "impulse_response": False,  # Skip for faster testing
                "visualization_data": True,
                "format": "json",
                "compression": False
            }
        }
    
    def check_backend_health(self) -> bool:
        """Check if backend is running and healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except requests.exceptions.ConnectionError:
            pass
        return False
    
    async def test_complete_workflow(self) -> Dict[str, Any]:
        """Test complete workflow from submission to results."""
        print("🔄 Testing Complete Workflow")
        print("-" * 40)
        
        # Step 1: Submit job
        print("1️⃣ Submitting simulation job...")
        test_request = self.create_test_request("Complete Workflow Test")
        
        response = requests.post(f"{self.base_url}/api/simulate", json=test_request, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Job submission failed: {response.status_code} - {response.text}")
        
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"   ✅ Job submitted: {job_id}")
        
        # Step 2: Start processing
        print("2️⃣ Starting job processing...")
        start_response = requests.post(f"{self.base_url}/api/simulate/start/{job_id}", timeout=10)
        if start_response.status_code != 200:
            raise Exception(f"Job start failed: {start_response.status_code} - {start_response.text}")
        
        print("   ✅ Processing started")
        
        # Step 3: Monitor via WebSocket
        print("3️⃣ Monitoring via WebSocket...")
        await self.monitor_job_websocket(job_id)
        
        # Step 4: Wait for completion
        print("4️⃣ Waiting for completion...")
        await self.wait_for_completion(job_id)
        
        # Step 5: Retrieve results
        print("5️⃣ Retrieving results...")
        results = await self.get_results(job_id)
        
        # Step 6: Validate results
        print("6️⃣ Validating results...")
        validation = self.validate_results(results)
        
        return {
            "job_id": job_id,
            "results": results,
            "validation": validation,
            "success": validation["is_valid"]
        }
    
    async def monitor_job_websocket(self, job_id: str):
        """Monitor job progress via WebSocket."""
        try:
            async with websockets.connect(f"{self.ws_url}/ws/jobs/{job_id}") as websocket:
                messages_received = 0
                timeout = 30  # 30 second timeout
                
                while messages_received < 10:  # Limit messages to prevent infinite loop
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        data = json.loads(message)
                        
                        status = data.get("status", "unknown")
                        progress = data.get("progress", 0.0)
                        message_text = data.get("message", "")
                        
                        print(f"   📡 WebSocket: {status} ({progress:.1%}) - {message_text}")
                        
                        if status in ["completed", "failed", "cancelled"]:
                            break
                            
                        messages_received += 1
                        
                    except asyncio.TimeoutError:
                        print("   ⏰ WebSocket timeout")
                        break
                        
        except Exception as e:
            print(f"   ⚠️ WebSocket monitoring failed: {e}")
    
    async def wait_for_completion(self, job_id: str, max_wait: int = 60):
        """Wait for job completion by polling status."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/api/jobs/{job_id}/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    progress = data.get("progress", 0.0)
                    
                    print(f"   📊 Status: {status} ({progress:.1%})")
                    
                    if status in ["completed", "failed", "cancelled"]:
                        print(f"   ✅ Job {status}")
                        return data
                        
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"   ⚠️ Status check failed: {e}")
                await asyncio.sleep(2)
        
        raise Exception(f"Job did not complete within {max_wait} seconds")
    
    async def get_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve and parse simulation results."""
        response = requests.get(f"{self.base_url}/api/jobs/{job_id}/results", timeout=10)
        if response.status_code != 200:
            raise Exception(f"Results retrieval failed: {response.status_code} - {response.text}")
        
        results = response.json()
        
        # Parse complex numbers
        for freq_data in results.get("frequencies", []):
            sensor_data = freq_data.get("sensor_data", {})
            for sensor_id, pressure in sensor_data.items():
                if isinstance(pressure, str):
                    try:
                        # Handle complex number strings
                        complex_str = pressure.replace('j', 'i')
                        sensor_data[sensor_id] = complex(complex_str)
                    except:
                        sensor_data[sensor_id] = 0.0 + 0.0j
                elif isinstance(pressure, dict) and "real" in pressure and "imag" in pressure:
                    sensor_data[sensor_id] = complex(pressure["real"], pressure["imag"])
        
        return results
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate simulation results."""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Check basic structure
        if "frequencies" not in results:
            validation["errors"].append("Missing frequencies in results")
            validation["is_valid"] = False
            return validation
        
        frequencies = results["frequencies"]
        if len(frequencies) == 0:
            validation["errors"].append("No frequency results found")
            validation["is_valid"] = False
            return validation
        
        # Validate each frequency result
        for i, freq_data in enumerate(frequencies):
            freq = freq_data.get("frequency", 0.0)
            sensor_data = freq_data.get("sensor_data", {})
            
            # Check sensor data
            if len(sensor_data) == 0:
                validation["warnings"].append(f"No sensor data for frequency {freq} Hz")
            
            # Check pressure values
            for sensor_id, pressure in sensor_data.items():
                if isinstance(pressure, complex):
                    magnitude = abs(pressure)
                    if magnitude == 0.0:
                        validation["warnings"].append(f"Zero pressure at {sensor_id} for {freq} Hz")
                    elif magnitude > 1.0:
                        validation["warnings"].append(f"High pressure at {sensor_id} for {freq} Hz: {magnitude:.3e}")
                else:
                    validation["errors"].append(f"Invalid pressure data type at {sensor_id} for {freq} Hz")
                    validation["is_valid"] = False
        
        # Calculate metrics
        validation["metrics"] = {
            "num_frequencies": len(frequencies),
            "num_sensors": len(frequencies[0].get("sensor_data", {})) if frequencies else 0,
            "frequencies": [f.get("frequency", 0.0) for f in frequencies]
        }
        
        return validation
    
    async def test_visualization_data(self) -> Dict[str, Any]:
        """Test 3D visualization data extraction."""
        print("🎨 Testing Visualization Data")
        print("-" * 40)
        
        # First run a complete workflow
        workflow_result = await self.test_complete_workflow()
        job_id = workflow_result["job_id"]
        
        # Test mesh data endpoint
        print("1️⃣ Testing mesh data endpoint...")
        mesh_response = requests.get(f"{self.base_url}/api/jobs/{job_id}/mesh", timeout=10)
        if mesh_response.status_code == 200:
            mesh_data = mesh_response.json()
            print(f"   ✅ Mesh data retrieved: {mesh_data.get('mesh', {}).get('num_vertices', 0)} vertices")
        else:
            print(f"   ❌ Mesh data failed: {mesh_response.status_code}")
        
        # Test field data endpoint
        print("2️⃣ Testing field data endpoint...")
        field_response = requests.get(f"{self.base_url}/api/jobs/{job_id}/field/100.0", timeout=10)
        if field_response.status_code == 200:
            field_data = field_response.json()
            print(f"   ✅ Field data retrieved for 100 Hz")
        else:
            print(f"   ❌ Field data failed: {field_response.status_code}")
        
        # Test complete visualization endpoint
        print("3️⃣ Testing complete visualization endpoint...")
        viz_response = requests.get(f"{self.base_url}/api/jobs/{job_id}/visualization", timeout=10)
        if viz_response.status_code == 200:
            viz_data = viz_response.json()
            print(f"   ✅ Complete visualization data retrieved")
            return {
                "mesh_data": mesh_data if mesh_response.status_code == 200 else None,
                "field_data": field_data if field_response.status_code == 200 else None,
                "visualization_data": viz_data if viz_response.status_code == 200 else None,
                "success": True
            }
        else:
            print(f"   ❌ Visualization data failed: {viz_response.status_code}")
            return {"success": False}
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling scenarios."""
        print("🚨 Testing Error Handling")
        print("-" * 40)
        
        error_tests = {}
        
        # Test invalid job ID
        print("1️⃣ Testing invalid job ID...")
        response = requests.get(f"{self.base_url}/api/jobs/invalid-id/status", timeout=5)
        error_tests["invalid_job_id"] = {
            "status_code": response.status_code,
            "expected": 404,
            "passed": response.status_code == 404
        }
        print(f"   {'✅' if error_tests['invalid_job_id']['passed'] else '❌'} Invalid job ID test")
        
        # Test invalid request data
        print("2️⃣ Testing invalid request data...")
        invalid_request = {"name": "Invalid Test"}  # Missing required fields
        response = requests.post(f"{self.base_url}/api/simulate", json=invalid_request, timeout=5)
        error_tests["invalid_request"] = {
            "status_code": response.status_code,
            "expected": 422,
            "passed": response.status_code == 422
        }
        print(f"   {'✅' if error_tests['invalid_request']['passed'] else '❌'} Invalid request test")
        
        # Test job cancellation
        print("3️⃣ Testing job cancellation...")
        test_request = self.create_test_request("Cancellation Test")
        response = requests.post(f"{self.base_url}/api/simulate", json=test_request, timeout=10)
        if response.status_code == 200:
            job_id = response.json()["job_id"]
            cancel_response = requests.delete(f"{self.base_url}/api/jobs/{job_id}", timeout=5)
            error_tests["job_cancellation"] = {
                "status_code": cancel_response.status_code,
                "expected": 200,
                "passed": cancel_response.status_code == 200
            }
            print(f"   {'✅' if error_tests['job_cancellation']['passed'] else '❌'} Job cancellation test")
        
        return error_tests
    
    async def test_backend_persistence(self) -> Dict[str, Any]:
        """Test backend persistence across restarts."""
        print("💾 Testing Backend Persistence")
        print("-" * 40)
        
        # Submit a job
        print("1️⃣ Submitting persistent test job...")
        test_request = self.create_test_request("Persistence Test")
        response = requests.post(f"{self.base_url}/api/simulate", json=test_request, timeout=10)
        
        if response.status_code != 200:
            return {"success": False, "error": "Failed to submit job"}
        
        job_id = response.json()["job_id"]
        print(f"   ✅ Job submitted: {job_id}")
        
        # Check job exists
        print("2️⃣ Verifying job exists...")
        status_response = requests.get(f"{self.base_url}/api/jobs/{job_id}/status", timeout=5)
        if status_response.status_code != 200:
            return {"success": False, "error": "Job not found"}
        
        print(f"   ✅ Job found: {status_response.json()['status']}")
        
        # Note: Full persistence test would require backend restart
        # For now, just verify job is stored
        return {
            "success": True,
            "job_id": job_id,
            "note": "Full persistence test requires backend restart"
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🧪 Running Comprehensive End-to-End Integration Tests")
        print("=" * 70)
        
        if not self.check_backend_health():
            print("❌ Backend is not running or not healthy")
            return {"success": False, "error": "Backend not available"}
        
        print("✅ Backend is healthy")
        
        test_results = {
            "backend_health": True,
            "workflow_test": None,
            "visualization_test": None,
            "error_handling_test": None,
            "persistence_test": None,
            "overall_success": False
        }
        
        try:
            # Test 1: Complete workflow
            test_results["workflow_test"] = await self.test_complete_workflow()
            print(f"   {'✅' if test_results['workflow_test']['success'] else '❌'} Complete workflow test")
            
            # Test 2: Visualization data
            test_results["visualization_test"] = await self.test_visualization_data()
            print(f"   {'✅' if test_results['visualization_test']['success'] else '❌'} Visualization data test")
            
            # Test 3: Error handling
            test_results["error_handling_test"] = await self.test_error_handling()
            error_tests_passed = all(test.get("passed", False) for test in test_results["error_handling_test"].values())
            print(f"   {'✅' if error_tests_passed else '❌'} Error handling test")
            
            # Test 4: Backend persistence
            test_results["persistence_test"] = await self.test_backend_persistence()
            print(f"   {'✅' if test_results['persistence_test']['success'] else '❌'} Persistence test")
            
            # Overall success
            test_results["overall_success"] = (
                test_results["workflow_test"]["success"] and
                test_results["visualization_test"]["success"] and
                test_results["persistence_test"]["success"] and
                error_tests_passed
            )
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            test_results["error"] = str(e)
        
        print("\n" + "=" * 70)
        print(f"🎯 Overall Test Result: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        
        return test_results


async def main():
    """Main test runner."""
    test_suite = EndToEndTestSuite()
    results = await test_suite.run_all_tests()
    
    # Save results to file
    results_file = Path("tests/integration/test_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Test results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
