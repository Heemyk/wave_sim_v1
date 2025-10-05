#!/usr/bin/env python3
"""
Comprehensive test runner for the Acoustic Room Simulation System.

This script provides a unified interface to run all tests or specific test categories.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --unit             # Run unit tests only
    python tests/run_tests.py --integration      # Run integration tests only
    python tests/run_tests.py --fem              # Run FEM tests only
    python tests/run_tests.py --quick            # Run quick tests (unit + FEM only)
    python tests/run_tests.py --backend          # Run backend tests (unit + integration)
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import json


class TestRunner:
    """Comprehensive test runner."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}
        
    def run_pytest(self, test_path: str, test_name: str = None) -> Dict[str, Any]:
        """Run pytest on a specific test path."""
        print(f"🧪 Running {test_name or test_path}...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            test_path, 
            "-v", 
            "--tb=short",
            "--durations=10"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return {
                "name": test_name or test_path,
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": "N/A"  # pytest handles timing
            }
            
        except subprocess.TimeoutExpired:
            return {
                "name": test_name or test_path,
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "duration": "TIMEOUT"
            }
        except Exception as e:
            return {
                "name": test_name or test_path,
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": "ERROR"
            }
    
    def run_script_test(self, script_path: str, test_name: str) -> Dict[str, Any]:
        """Run a standalone test script."""
        print(f"🧪 Running {test_name}...")
        
        script_full_path = self.project_root / script_path
        
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(script_full_path)], 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            duration = time.time() - start_time
            
            return {
                "name": test_name,
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": f"{duration:.2f}s"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "name": test_name,
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "duration": "TIMEOUT"
            }
        except Exception as e:
            return {
                "name": test_name,
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": "ERROR"
            }
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔄 Running Integration Tests")
        print("=" * 50)
        
        # Run the comprehensive integration test
        result = self.run_script_test(
            "tests/integration/test_end_to_end_workflows.py",
            "End-to-End Integration Tests"
        )
        
        return result
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        print("🔧 Running Unit Tests")
        print("=" * 50)
        
        # Run pytest on unit tests
        result = self.run_pytest("tests/unit/", "Unit Tests")
        
        return result
    
    def run_fem_tests(self) -> Dict[str, Any]:
        """Run FEM tests."""
        print("🧮 Running FEM Tests")
        print("=" * 50)
        
        # Run pytest on FEM tests
        result = self.run_pytest("tests/fem/", "FEM Tests")
        
        return result
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run quick tests (unit + FEM, no integration)."""
        print("⚡ Running Quick Tests")
        print("=" * 50)
        
        results = {}
        
        # Run unit tests
        results["unit"] = self.run_unit_tests()
        
        # Run FEM tests
        results["fem"] = self.run_fem_tests()
        
        return results
    
    def run_backend_tests(self) -> Dict[str, Any]:
        """Run backend tests (unit + integration)."""
        print("🔧 Running Backend Tests")
        print("=" * 50)
        
        results = {}
        
        # Run unit tests
        results["unit"] = self.run_unit_tests()
        
        # Run integration tests (async)
        results["integration"] = asyncio.run(self.run_integration_tests())
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        print("🧪 Running All Tests")
        print("=" * 70)
        
        results = {}
        
        # Run unit tests
        results["unit"] = self.run_unit_tests()
        
        # Run FEM tests
        results["fem"] = self.run_fem_tests()
        
        # Run integration tests
        results["integration"] = asyncio.run(self.run_integration_tests())
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results summary."""
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        total_tests = 0
        passed_tests = 0
        
        for category, result in results.items():
            if isinstance(result, dict):
                if "success" in result:
                    # Single test result
                    total_tests += 1
                    if result["success"]:
                        passed_tests += 1
                    
                    status = "✅ PASSED" if result["success"] else "❌ FAILED"
                    duration = result.get("duration", "N/A")
                    print(f"{category.upper():<20} {status:<12} ({duration})")
                    
                    if not result["success"] and result.get("stderr"):
                        print(f"{'':20} Error: {result['stderr'][:100]}...")
                        
                else:
                    # Multiple test results
                    print(f"\n{category.upper()} TESTS:")
                    for test_name, test_result in result.items():
                        total_tests += 1
                        if test_result["success"]:
                            passed_tests += 1
                        
                        status = "✅ PASSED" if test_result["success"] else "❌ FAILED"
                        duration = test_result.get("duration", "N/A")
                        print(f"  {test_name:<30} {status:<12} ({duration})")
                        
                        if not test_result["success"] and test_result.get("stderr"):
                            print(f"  {'':30} Error: {test_result['stderr'][:80]}...")
        
        print("\n" + "=" * 70)
        print(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed")
            return False
    
    def save_results(self, results: Dict[str, Any], filename: str = "test_results.json"):
        """Save test results to file."""
        results_file = self.project_root / "tests" / filename
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")
    
    def run(self, args: argparse.Namespace):
        """Main test runner."""
        print("🧪 Acoustic Room Simulation System - Test Suite")
        print("=" * 70)
        
        start_time = time.time()
        
        # Determine which tests to run
        if args.unit:
            results = {"unit": self.run_unit_tests()}
        elif args.integration:
            results = {"integration": asyncio.run(self.run_integration_tests())}
        elif args.fem:
            results = {"fem": self.run_fem_tests()}
        elif args.quick:
            results = self.run_quick_tests()
        elif args.backend:
            results = self.run_backend_tests()
        else:
            # Run all tests
            results = self.run_all_tests()
        
        total_duration = time.time() - start_time
        
        # Print results
        success = self.print_results(results)
        
        print(f"\n⏱️  Total test duration: {total_duration:.2f} seconds")
        
        # Save results
        if args.save_results:
            self.save_results(results)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for the Acoustic Room Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py                    # Run all tests
  python tests/run_tests.py --unit             # Run unit tests only
  python tests/run_tests.py --integration      # Run integration tests only
  python tests/run_tests.py --fem              # Run FEM tests only
  python tests/run_tests.py --quick            # Run quick tests (unit + FEM)
  python tests/run_tests.py --backend          # Run backend tests (unit + integration)
  python tests/run_tests.py --save-results     # Save results to JSON file
        """
    )
    
    # Test category options (mutually exclusive)
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests only")
    test_group.add_argument("--fem", action="store_true", help="Run FEM tests only")
    test_group.add_argument("--quick", action="store_true", help="Run quick tests (unit + FEM)")
    test_group.add_argument("--backend", action="store_true", help="Run backend tests (unit + integration)")
    
    # Other options
    parser.add_argument("--save-results", action="store_true", help="Save test results to JSON file")
    
    args = parser.parse_args()
    
    # Run tests
    runner = TestRunner()
    runner.run(args)


if __name__ == "__main__":
    main()
