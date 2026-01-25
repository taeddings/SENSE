"""
SENSE-v2 Health Monitoring
System health checks and monitoring utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import os
import psutil


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus = HealthStatus.UNKNOWN
    checks: List[HealthCheck] = field(default_factory=list)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                }
                for c in self.checks
            ],
            "resources": {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "disk_percent": self.disk_percent,
            },
            "timestamp": self.timestamp.isoformat(),
        }


class HealthMonitor:
    """
    Health monitoring for SENSE-v2.

    Provides:
    - System resource monitoring
    - Component health checks
    - Alerting on degraded status
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Registered health checks
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}

        # Health history
        self._history: List[SystemHealth] = []
        self.max_history = 100

        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 75.0
        self.memory_critical_threshold = 90.0
        self.disk_warning_threshold = 80.0
        self.disk_critical_threshold = 95.0

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("cpu", self._check_cpu)
        self.register_check("memory", self._check_memory)
        self.register_check("disk", self._check_disk)

    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheck],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_func

    def _check_cpu(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)

            if cpu_percent >= self.cpu_critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"CPU critical: {cpu_percent}%"
            elif cpu_percent >= self.cpu_warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"CPU high: {cpu_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU normal: {cpu_percent}%"

            return HealthCheck(
                name="cpu",
                status=status,
                message=message,
                metadata={"cpu_percent": cpu_percent},
            )

        except Exception as e:
            return HealthCheck(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check CPU: {e}",
            )

    def _check_memory(self) -> HealthCheck:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent >= self.memory_critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Memory critical: {memory_percent}%"
            elif memory_percent >= self.memory_warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Memory high: {memory_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory normal: {memory_percent}%"

            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                metadata={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory: {e}",
            )

    def _check_disk(self) -> HealthCheck:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            if disk_percent >= self.disk_critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Disk critical: {disk_percent}%"
            elif disk_percent >= self.disk_warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Disk high: {disk_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk normal: {disk_percent}%"

            return HealthCheck(
                name="disk",
                status=status,
                message=message,
                metadata={
                    "disk_percent": disk_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheck(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk: {e}",
            )

    def check_health(self) -> SystemHealth:
        """Run all health checks and return overall status."""
        checks = []

        for name, check_func in self._checks.items():
            try:
                start = datetime.now()
                result = check_func()
                result.latency_ms = int((datetime.now() - start).total_seconds() * 1000)
                checks.append(result)
            except Exception as e:
                checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                ))

        # Determine overall status
        statuses = [c.status for c in checks]

        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        # Get resource usage
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
        except Exception:
            cpu = memory = disk = 0.0

        health = SystemHealth(
            status=overall,
            checks=checks,
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk,
        )

        # Store in history
        self._history.append(health)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        return health

    async def check_health_async(self) -> SystemHealth:
        """Async version of health check."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.check_health
        )

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health history."""
        return [h.to_dict() for h in self._history[-limit:]]

    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        if not self._history:
            self.check_health()

        if self._history:
            return self._history[-1].status == HealthStatus.HEALTHY

        return False

    def add_component_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        description: str = "",
    ) -> None:
        """
        Add a simple boolean health check for a component.

        Args:
            name: Component name
            check_func: Function returning True if healthy
            description: Description of what's being checked
        """
        def wrapped_check() -> HealthCheck:
            try:
                is_healthy = check_func()
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                    message=description or f"{name} {'healthy' if is_healthy else 'unhealthy'}",
                )
            except Exception as e:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                )

        self.register_check(name, wrapped_check)


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
