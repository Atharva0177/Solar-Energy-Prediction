#!/usr/bin/env python3
"""
Health check script for UNISOLAR Solar Platform.
Verifies all services are running and healthy.

Usage:
    python scripts/health_check.py [--verbose] [--timeout 30]
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple

class HealthChecker:
    """Check health of all platform services."""
    
    SERVICES = {
        'backend': ('http://localhost:8000/api/v1/health', 'Backend API'),
        'frontend': ('http://localhost:80/health', 'Frontend (Nginx)'),
        'db': ('localhost:5432', 'PostgreSQL Database'),
        'redis': ('localhost:6379', 'Redis Cache'),
        'mlflow': ('http://localhost:5000', 'MLflow Tracking'),
        'prometheus': ('http://localhost:9090', 'Prometheus'),
        'grafana': ('http://localhost:3000', 'Grafana'),
    }
    
    def __init__(self, timeout: int = 30, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results: Dict[str, Tuple[bool, str]] = {}
    
    def log(self, message: str, level: str = 'INFO'):
        """Print log message."""
        prefix = f"[{level}]"
        print(f"{prefix} {message}")
    
    def check_http(self, url: str, service_name: str) -> bool:
        """Check HTTP endpoint health."""
        try:
            result = subprocess.run(
                ['curl', '-sf', '--connect-timeout', '5', url],
                capture_output=True,
                timeout=self.timeout
            )
            return result.returncode == 0
        except Exception as e:
            if self.verbose:
                self.log(f"HTTP check failed for {service_name}: {e}", 'DEBUG')
            return False
    
    def check_port(self, host: str, port: int, service_name: str) -> bool:
        """Check if port is open."""
        try:
            result = subprocess.run(
                ['bash', '-c', f'timeout 5 bash -c "</dev/tcp/{host}/{port}"'],
                capture_output=True,
                timeout=self.timeout
            )
            return result.returncode == 0
        except Exception as e:
            if self.verbose:
                self.log(f"Port check failed for {service_name}: {e}", 'DEBUG')
            return False
    
    def check_service(self, service_key: str, endpoint: str, service_name: str) -> bool:
        """Check individual service health."""
        if self.verbose:
            self.log(f"Checking {service_name}...", 'DEBUG')
        
        if endpoint.startswith('http://') or endpoint.startswith('https://'):
            return self.check_http(endpoint, service_name)
        elif ':' in endpoint:
            host, port = endpoint.split(':')
            return self.check_port(host, int(port), service_name)
        return False
    
    def check_docker_compose(self) -> bool:
        """Check if docker-compose services are running."""
        try:
            result = subprocess.run(
                ['docker-compose', 'ps', '--quiet'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and len(result.stdout.strip().split('\n')) >= 1
        except Exception as e:
            if self.verbose:
                self.log(f"Docker Compose check failed: {e}", 'DEBUG')
            return False
    
    def run_checks(self) -> bool:
        """Run all health checks."""
        self.log("Running health checks for UNISOLAR Solar Platform...")
        self.log("")
        
        # Check Docker Compose
        self.log("Checking Docker Compose services...")
        if not self.check_docker_compose():
            self.log("⚠️  No Docker Compose services running", 'WARN')
            self.log("   Start with: docker-compose up -d", 'WARN')
            return False
        
        self.log("✅ Docker Compose services found", 'OK')
        self.log("")
        
        # Check individual services
        self.log("Checking individual service endpoints...")
        all_healthy = True
        
        for service_key, (endpoint, service_name) in self.SERVICES.items():
            healthy = self.check_service(service_key, endpoint, service_name)
            status = "✅" if healthy else "❌"
            
            self.results[service_key] = (healthy, service_name)
            self.log(f"{status} {service_name:30s} ({endpoint})")
            
            if not healthy:
                all_healthy = False
        
        self.log("")
        
        # Summary
        healthy_count = sum(1 for ok, _ in self.results.values() if ok)
        total_count = len(self.results)
        
        self.log(f"Summary: {healthy_count}/{total_count} services healthy")
        
        if all_healthy:
            self.log("✅ All services are healthy!", 'OK')
        else:
            self.log("⚠️  Some services are not healthy", 'WARN')
            self.log("", 'WARN')
            self.log("Unhealthy services:", 'WARN')
            for service_key, (healthy, service_name) in self.results.items():
                if not healthy:
                    self.log(f"  - {service_name}", 'WARN')
            
            self.log("", 'WARN')
            self.log("Troubleshooting:", 'WARN')
            self.log("  1. Check Docker Compose is running: docker-compose ps", 'WARN')
            self.log("  2. View logs: docker-compose logs -f <service>", 'WARN')
            self.log("  3. Restart services: docker-compose restart", 'WARN')
        
        self.log("")
        return all_healthy


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Health check for UNISOLAR Solar Platform'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout for checks in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    checker = HealthChecker(timeout=args.timeout, verbose=args.verbose)
    all_healthy = checker.run_checks()
    
    sys.exit(0 if all_healthy else 1)


if __name__ == '__main__':
    main()
