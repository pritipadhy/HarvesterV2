#!/usr/bin/env python3
"""
Harvester V2 Setup and Test Runner
===================================

This script handles:
1. Environment validation (Python, Docker, dependencies)
2. Database setup (PostgreSQL, Weaviate, Redis)
3. Schema migrations
4. Running unit, integration, and E2E tests

Usage:
    python scripts/setup_and_test.py --check       # Check environment
    python scripts/setup_and_test.py --setup       # Setup databases
    python scripts/setup_and_test.py --migrate     # Apply migrations
    python scripts/setup_and_test.py --test-unit   # Run unit tests
    python scripts/setup_and_test.py --test-int    # Run integration tests
    python scripts/setup_and_test.py --test-e2e    # Run E2E tests
    python scripts/setup_and_test.py --all         # Do everything
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
HARVESTER_V2_ROOT = PROJECT_ROOT / "harvester_v2"

# Docker compose file
DOCKER_COMPOSE_FILE = HARVESTER_V2_ROOT / "docker-compose.test.yml"

# Minimum versions
MIN_PYTHON_VERSION = (3, 11)
REQUIRED_PACKAGES = [
    "pytest",
    "pytest-asyncio",
    "structlog",
    "weaviate-client",
    "asyncpg",
    "redis",
]


def log_step(message: str, level: str = "INFO"):
    """Log a step with formatting."""
    colors = {
        "INFO": "\033[94m",
        "OK": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
    }
    reset = "\033[0m"
    color = colors.get(level, "")
    print(f"{color}[{level}]{reset} {message}")


def run_command(cmd: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            log_step(f"Command failed: {' '.join(cmd)}", "ERROR")
            log_step(f"Error: {e.stderr if e.stderr else e}", "ERROR")
        raise


def check_python_version():
    """Check Python version is 3.11+."""
    log_step(f"Checking Python version (need {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+)...")

    current = sys.version_info[:2]
    if current < MIN_PYTHON_VERSION:
        log_step(f"Python {current[0]}.{current[1]} found, need {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+", "ERROR")
        return False

    log_step(f"Python {current[0]}.{current[1]} OK", "OK")
    return True


def check_docker():
    """Check Docker is installed and running."""
    log_step("Checking Docker...")

    try:
        result = run_command(["docker", "--version"], capture=True)
        log_step(f"Docker installed: {result.stdout.strip()}", "OK")
    except (subprocess.CalledProcessError, FileNotFoundError):
        log_step("Docker not found. Please install Docker Desktop.", "ERROR")
        log_step("Download from: https://www.docker.com/products/docker-desktop/", "WARN")
        return False

    try:
        run_command(["docker", "info"], capture=True)
        log_step("Docker daemon is running", "OK")
    except subprocess.CalledProcessError:
        log_step("Docker daemon is not running. Please start Docker Desktop.", "ERROR")
        return False

    return True


def check_docker_compose():
    """Check docker-compose is available."""
    log_step("Checking docker-compose...")

    try:
        result = run_command(["docker-compose", "--version"], capture=True)
        log_step(f"docker-compose: {result.stdout.strip()}", "OK")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try docker compose (v2)
        try:
            result = run_command(["docker", "compose", "version"], capture=True)
            log_step(f"Docker Compose v2: {result.stdout.strip()}", "OK")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            log_step("docker-compose not found", "ERROR")
            return False


def check_packages():
    """Check required Python packages are installed."""
    log_step("Checking required packages...")

    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        log_step(f"Missing packages: {', '.join(missing)}", "WARN")
        log_step("Install with: pip install " + " ".join(missing), "WARN")
        return False

    log_step("All required packages installed", "OK")
    return True


def check_environment():
    """Run all environment checks."""
    log_step("=" * 50)
    log_step("Harvester V2 Environment Check")
    log_step("=" * 50)

    results = {
        "python": check_python_version(),
        "docker": check_docker(),
        "docker_compose": check_docker_compose(),
        "packages": check_packages(),
    }

    log_step("=" * 50)

    all_ok = all(results.values())
    if all_ok:
        log_step("All checks passed! Ready to proceed.", "OK")
    else:
        failed = [k for k, v in results.items() if not v]
        log_step(f"Failed checks: {', '.join(failed)}", "ERROR")
        log_step("Please fix the issues above before continuing.", "WARN")

    return all_ok


def start_databases():
    """Start database services using docker-compose."""
    log_step("Starting database services...")

    if not DOCKER_COMPOSE_FILE.exists():
        log_step(f"docker-compose file not found: {DOCKER_COMPOSE_FILE}", "ERROR")
        return False

    try:
        # Use docker compose (v2) or docker-compose (v1)
        try:
            run_command(["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "-d"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            run_command(["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "-d"])

        log_step("Waiting for services to be healthy...", "INFO")
        time.sleep(10)

        # Check service health
        check_service_health()

        log_step("Database services started", "OK")
        return True

    except subprocess.CalledProcessError as e:
        log_step(f"Failed to start databases: {e}", "ERROR")
        return False


def check_service_health():
    """Check if all services are healthy."""
    services = {
        "PostgreSQL": ("localhost", 5433),
        "Weaviate": ("localhost", 8081),
        "Redis": ("localhost", 6380),
    }

    import socket

    for name, (host, port) in services.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            log_step(f"{name} is reachable at {host}:{port}", "OK")
        else:
            log_step(f"{name} not reachable at {host}:{port}", "WARN")


def apply_migrations():
    """Apply PostgreSQL schema migrations."""
    log_step("Applying PostgreSQL migrations...")

    schema_file = HARVESTER_V2_ROOT / "storage" / "postgresql_schema.sql"
    if not schema_file.exists():
        log_step(f"Schema file not found: {schema_file}", "ERROR")
        return False

    try:
        # Read schema
        schema = schema_file.read_text()

        # Connect to PostgreSQL and apply
        import asyncio
        import asyncpg

        async def apply_schema():
            conn = await asyncpg.connect(
                host="localhost",
                port=5433,
                user="harvester",
                password="harvester_test_pass",
                database="harvester_v2"
            )
            try:
                await conn.execute(schema)
                log_step("Schema applied successfully", "OK")
            finally:
                await conn.close()

        asyncio.run(apply_schema())
        return True

    except Exception as e:
        log_step(f"Failed to apply migrations: {e}", "ERROR")
        return False


def create_weaviate_collections():
    """Create Weaviate collections."""
    log_step("Creating Weaviate collections...")

    try:
        # Add project to path
        sys.path.insert(0, str(PROJECT_ROOT))

        from harvester_v2.storage.weaviate_schema import HARVESTER_V2_SCHEMA, create_schema
        import weaviate

        client = weaviate.Client("http://localhost:8081")

        if client.is_ready():
            create_schema(client)
            log_step("Weaviate collections created", "OK")
            return True
        else:
            log_step("Weaviate is not ready", "ERROR")
            return False

    except Exception as e:
        log_step(f"Failed to create Weaviate collections: {e}", "ERROR")
        return False


def run_unit_tests():
    """Run unit tests."""
    log_step("Running unit tests...")

    try:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
        result = run_command([
            sys.executable, "-m", "pytest",
            str(HARVESTER_V2_ROOT / "tests"),
            "-v",
            "--tb=short",
            "-m", "not integration and not e2e",
        ], check=False)

        if result.returncode == 0:
            log_step("Unit tests passed", "OK")
            return True
        else:
            log_step("Some unit tests failed", "WARN")
            return False

    except Exception as e:
        log_step(f"Failed to run unit tests: {e}", "ERROR")
        return False


def run_integration_tests():
    """Run integration tests with real databases."""
    log_step("Running integration tests...")

    try:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
        os.environ["HARVESTER_V2_TEST_MODE"] = "integration"
        os.environ["POSTGRES_HOST"] = "localhost"
        os.environ["POSTGRES_PORT"] = "5433"
        os.environ["WEAVIATE_URL"] = "http://localhost:8081"
        os.environ["REDIS_URL"] = "redis://localhost:6380"

        result = run_command([
            sys.executable, "-m", "pytest",
            str(HARVESTER_V2_ROOT / "tests"),
            "-v",
            "--tb=short",
            "-m", "integration",
        ], check=False)

        if result.returncode == 0:
            log_step("Integration tests passed", "OK")
            return True
        else:
            log_step("Some integration tests failed", "WARN")
            return False

    except Exception as e:
        log_step(f"Failed to run integration tests: {e}", "ERROR")
        return False


def run_e2e_tests():
    """Run E2E tests."""
    log_step("Running E2E tests...")

    try:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
        os.environ["HARVESTER_V2_TEST_MODE"] = "e2e"

        result = run_command([
            sys.executable, "-m", "pytest",
            str(HARVESTER_V2_ROOT / "tests"),
            "-v",
            "--tb=short",
            "-m", "e2e",
        ], check=False)

        if result.returncode == 0:
            log_step("E2E tests passed", "OK")
            return True
        else:
            log_step("Some E2E tests failed", "WARN")
            return False

    except Exception as e:
        log_step(f"Failed to run E2E tests: {e}", "ERROR")
        return False


def stop_databases():
    """Stop database services."""
    log_step("Stopping database services...")

    try:
        try:
            run_command(["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "down"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            run_command(["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "down"])

        log_step("Database services stopped", "OK")
        return True

    except subprocess.CalledProcessError as e:
        log_step(f"Failed to stop databases: {e}", "ERROR")
        return False


def main():
    parser = argparse.ArgumentParser(description="Harvester V2 Setup and Test Runner")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--setup", action="store_true", help="Setup databases")
    parser.add_argument("--migrate", action="store_true", help="Apply migrations")
    parser.add_argument("--test-unit", action="store_true", help="Run unit tests")
    parser.add_argument("--test-int", action="store_true", help="Run integration tests")
    parser.add_argument("--test-e2e", action="store_true", help="Run E2E tests")
    parser.add_argument("--stop", action="store_true", help="Stop databases")
    parser.add_argument("--all", action="store_true", help="Run everything")

    args = parser.parse_args()

    # Default to check if no args
    if not any(vars(args).values()):
        args.check = True

    success = True

    if args.check or args.all:
        success = check_environment() and success

    if args.setup or args.all:
        success = start_databases() and success

    if args.migrate or args.all:
        success = apply_migrations() and success
        success = create_weaviate_collections() and success

    if args.test_unit or args.all:
        success = run_unit_tests() and success

    if args.test_int or args.all:
        success = run_integration_tests() and success

    if args.test_e2e or args.all:
        success = run_e2e_tests() and success

    if args.stop:
        success = stop_databases() and success

    log_step("=" * 50)
    if success:
        log_step("All operations completed successfully!", "OK")
        return 0
    else:
        log_step("Some operations failed. Check the logs above.", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
