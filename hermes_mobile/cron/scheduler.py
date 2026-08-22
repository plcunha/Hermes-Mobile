"""Enhanced Cron System - Full scheduling like Hermes Desktop"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    from croniter import croniter

    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False

from hermes_mobile.config.settings import get_settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Configuration — lazily resolved to avoid crashes on Android
# at import time (Path.home() may fail in sandbox).
# ═══════════════════════════════════════════════════════════════

TICKER_INTERVAL_SECONDS = 60
ONESHOT_GRACE_SECONDS = 120
_JOBS_LOCK_TIMEOUT_SECONDS = 30.0


def _get_cron_dir() -> Path:
    """Resolve the cron directory lazily."""
    settings = get_settings()
    return Path(settings.data_dir).resolve() / "cron"


def _get_jobs_file() -> Path:
    return _get_cron_dir() / "jobs.json"


def _get_output_dir() -> Path:
    return _get_cron_dir() / "output"


def _get_ticker_heartbeat_file() -> Path:
    return _get_cron_dir() / "ticker_heartbeat"


def _get_ticker_success_file() -> Path:
    return _get_cron_dir() / "ticker_last_success"


# ═══════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class CronJob:
    """Represents a cron job."""

    id: str
    name: str
    schedule: str  # cron expression or "oneshot"
    command: str
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    last_status: Optional[str] = None  # success, failed, running
    last_output: Optional[str] = None
    run_count: int = 0
    failure_count: int = 0
    timeout: int = 300  # seconds
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "command": self.command,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "last_status": self.last_status,
            "last_output": self.last_output,
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "timeout": self.timeout,
            "env_vars": self.env_vars,
            "working_dir": self.working_dir,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CronJob:
        return cls(**data)


@dataclass
class CronOutput:
    """Represents cron job output."""

    job_id: str
    timestamp: str
    status: str  # success, failed, running
    stdout: str
    stderr: str
    return_code: int
    duration: float

    def to_markdown(self) -> str:
        return f"""# Cron Job Output

**Job ID:** {self.job_id}
**Timestamp:** {self.timestamp}
**Status:** {self.status}
**Duration:** {self.duration:.2f}s
**Return Code:** {self.return_code}

## Stdout
```
{self.stdout or "(empty)"}
```

## Stderr
```
{self.stderr or "(empty)"}
```
"""


# ═══════════════════════════════════════════════════════════════
# Cron Store Management
# ═══════════════════════════════════════════════════════════════

_jobs_file_lock = threading.RLock()
_jobs_lock_state = threading.local()


def _lock_file_path() -> Path:
    return _get_cron_dir() / ".jobs.lock"


@contextlib.contextmanager
def _jobs_lock():
    """Cross-process advisory file lock for jobs.json."""
    lock_file = _lock_file_path()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    lock_acquired = False
    lock_fh = None

    try:
        lock_fh = open(lock_file, "w")
        if fcntl:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            # No locking available
            pass
        lock_acquired = True
        yield
    except (BlockingIOError, OSError):
        # Lock not acquired - another process has it
        raise TimeoutError("Could not acquire jobs lock")
    finally:
        if lock_acquired and lock_fh:
            try:
                if fcntl:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                elif msvcrt:
                    msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            lock_fh.close()


def _ensure_cron_dirs():
    """Ensure cron directories exist."""
    _get_cron_dir().mkdir(parents=True, exist_ok=True)
    _get_output_dir().mkdir(parents=True, exist_ok=True)


def _load_jobs() -> Dict[str, CronJob]:
    """Load jobs from JSON file."""
    _ensure_cron_dirs()
    if not _get_jobs_file().exists():
        return {}

    try:
        with open(_get_jobs_file()) as f:
            data = json.load(f)
        return {job_id: CronJob.from_dict(job_data) for job_id, job_data in data.items()}
    except Exception as e:
        logger.error(f"Failed to load cron jobs: {e}")
        return {}


def _save_jobs(jobs: Dict[str, CronJob]) -> None:
    """Save jobs to JSON file atomically."""
    _ensure_cron_dirs()
    temp_file = _get_jobs_file().with_suffix(".tmp")
    try:
        with open(temp_file, "w") as f:
            json.dump({job_id: job.to_dict() for job_id, job in jobs.items()}, f, indent=2)
        temp_file.replace(_get_jobs_file())
    except Exception as e:
        logger.error(f"Failed to save cron jobs: {e}")
        if temp_file.exists():
            temp_file.unlink()


def _compute_next_run(schedule: str, from_time: Optional[datetime] = None) -> Optional[str]:
    """Compute next run time from cron expression."""
    if schedule == "oneshot":
        return None

    if not HAS_CRONITER:
        logger.warning("croniter not installed, cannot compute next run")
        return None

    try:
        base = from_time or datetime.now()
        cron = croniter(schedule, base)
        return cron.get_next(datetime).isoformat()
    except Exception as e:
        logger.error(f"Failed to compute next run for schedule '{schedule}': {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════


def list_jobs() -> List[CronJob]:
    """List all cron jobs."""
    with _jobs_lock():
        jobs = _load_jobs()
    return list(jobs.values())


def get_job(job_id: str) -> Optional[CronJob]:
    """Get a specific cron job."""
    with _jobs_lock():
        jobs = _load_jobs()
    return jobs.get(job_id)


def create_job(
    name: str,
    schedule: str,
    command: str,
    enabled: bool = True,
    timeout: int = 300,
    env_vars: Optional[Dict[str, str]] = None,
    working_dir: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> CronJob:
    """Create a new cron job."""
    job = CronJob(
        id=str(uuid.uuid4())[:8],
        name=name,
        schedule=schedule,
        command=command,
        enabled=enabled,
        timeout=timeout,
        env_vars=env_vars or {},
        working_dir=working_dir,
        description=description,
        tags=tags or [],
    )
    job.next_run = _compute_next_run(schedule)

    with _jobs_lock():
        jobs = _load_jobs()
        jobs[job.id] = job
        _save_jobs(jobs)

    return job


def update_job(job_id: str, **kwargs) -> Optional[CronJob]:
    """Update a cron job."""
    with _jobs_lock():
        jobs = _load_jobs()
        if job_id not in jobs:
            return None

        job = jobs[job_id]
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        job.updated_at = datetime.now().isoformat()
        if "schedule" in kwargs:
            job.next_run = _compute_next_run(kwargs["schedule"])

        _save_jobs(jobs)
        return job


def delete_job(job_id: str) -> bool:
    """Delete a cron job."""
    with _jobs_lock():
        jobs = _load_jobs()
        if job_id not in jobs:
            return False
        del jobs[job_id]
        _save_jobs(jobs)
    return True


def enable_job(job_id: str) -> bool:
    """Enable a cron job."""
    return update_job(job_id, enabled=True) is not None


def disable_job(job_id: str) -> bool:
    """Disable a cron job."""
    return update_job(job_id, enabled=False) is not None


def run_job_now(job_id: str) -> CronOutput:
    """Run a cron job immediately."""
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    return _execute_job(job)


def _execute_job(job: CronJob) -> CronOutput:
    """Execute a cron job and capture output."""
    start_time = time.time()
    timestamp = datetime.now().isoformat()

    # Prepare environment
    env = os.environ.copy()
    env.update(job.env_vars)

    # Prepare working directory
    cwd = job.working_dir or str(Path(get_settings().data_dir).resolve())

    # Update job status
    update_job(job.id, last_status="running", last_run=timestamp)

    try:
        # Run command
        result = subprocess.run(
            job.command,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=job.timeout,
        )

        duration = time.time() - start_time
        status = "success" if result.returncode == 0 else "failed"

        output = CronOutput(
            job_id=job.id,
            timestamp=timestamp,
            status=status,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            duration=duration,
        )

        # Save output to file
        _save_output(job.id, output)

        # Update job
        update_job(
            job.id,
            last_status=status,
            last_output=output.to_markdown()[:5000],  # Truncate
            run_count=job.run_count + 1,
            failure_count=job.failure_count + (1 if status == "failed" else 0),
            next_run=_compute_next_run(job.schedule) if job.schedule != "oneshot" else None,
        )

        return output

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        output = CronOutput(
            job_id=job.id,
            timestamp=timestamp,
            status="failed",
            stdout="",
            stderr=f"Job timed out after {job.timeout} seconds",
            return_code=-1,
            duration=duration,
        )
        _save_output(job.id, output)
        update_job(job.id, last_status="failed", last_output=output.to_markdown()[:5000])
        return output

    except Exception as e:
        duration = time.time() - start_time
        output = CronOutput(
            job_id=job.id,
            timestamp=timestamp,
            status="failed",
            stdout="",
            stderr=str(e),
            return_code=-1,
            duration=duration,
        )
        _save_output(job.id, output)
        update_job(job.id, last_status="failed", last_output=output.to_markdown()[:5000])
        return output


def _save_output(job_id: str, output: CronOutput) -> None:
    """Save job output to file."""
    job_output_dir = _get_output_dir() / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = job_output_dir / f"{timestamp}.md"

    try:
        output_file.write_text(output.to_markdown())
    except Exception as e:
        logger.error(f"Failed to save cron output: {e}")


def get_job_output(job_id: str, limit: int = 10) -> List[CronOutput]:
    """Get recent output for a job."""
    job_output_dir = _get_output_dir() / job_id
    if not job_output_dir.exists():
        return []

    outputs = []
    for output_file in sorted(job_output_dir.glob("*.md"), reverse=True)[:limit]:
        try:
            content = output_file.read_text()
            # Parse markdown back to CronOutput (simplified)
            outputs.append(
                CronOutput(
                    job_id=job_id,
                    timestamp=output_file.stem,
                    status="unknown",
                    stdout=content,
                    stderr="",
                    return_code=0,
                    duration=0,
                )
            )
        except Exception:
            pass

    return outputs


# ═══════════════════════════════════════════════════════════════
# Ticker (Background Scheduler)
# ═══════════════════════════════════════════════════════════════

_ticker_thread: Optional[threading.Thread] = None
_ticker_stop_event = threading.Event()
_ticker_running = False


def _ticker_loop():
    """Background ticker loop."""
    global _ticker_running
    _ticker_running = True

    while not _ticker_stop_event.is_set():
        try:
            _tick()
        except Exception as e:
            logger.error(f"Cron ticker error: {e}")

        # Update heartbeat
        try:
            _get_ticker_heartbeat_file().write_text(datetime.now().isoformat())
        except Exception:
            pass

        # Wait for next tick
        _ticker_stop_event.wait(TICKER_INTERVAL_SECONDS)

    _ticker_running = False


def _tick():
    """Check for due jobs and run them."""
    now = datetime.now()

    with _jobs_lock():
        jobs = _load_jobs()

    for job in jobs.values():
        if not job.enabled:
            continue

        if job.schedule == "oneshot":
            # Oneshot jobs run once when created
            if job.last_run is None:
                _execute_job(job)
            continue

        if job.next_run:
            try:
                next_run = datetime.fromisoformat(job.next_run)
                if next_run <= now:
                    _execute_job(job)
            except Exception as e:
                logger.error(f"Error checking job {job.id}: {e}")

    # Update success timestamp
    try:
        _get_ticker_success_file().write_text(datetime.now().isoformat())
    except Exception:
        pass


def start_ticker():
    """Start the background cron ticker."""
    global _ticker_thread
    if _ticker_thread and _ticker_thread.is_alive():
        return

    _ticker_stop_event.clear()
    _ticker_thread = threading.Thread(target=_ticker_loop, daemon=True, name="cron-ticker")
    _ticker_thread.start()
    logger.info("Cron ticker started")


def stop_ticker():
    """Stop the background cron ticker."""
    global _ticker_thread
    _ticker_stop_event.set()
    if _ticker_thread:
        _ticker_thread.join(timeout=5)
        _ticker_thread = None
    logger.info("Cron ticker stopped")


def is_ticker_running() -> bool:
    """Check if ticker is running."""
    return _ticker_running and _ticker_thread and _ticker_thread.is_alive()


def get_ticker_status() -> Dict[str, Any]:
    """Get ticker status."""
    heartbeat = None
    last_success = None

    try:
        if _get_ticker_heartbeat_file().exists():
            heartbeat = _get_ticker_heartbeat_file().read_text().strip()
    except Exception:
        pass

    try:
        if _get_ticker_success_file().exists():
            last_success = _get_ticker_success_file().read_text().strip()
    except Exception:
        pass

    return {
        "running": is_ticker_running(),
        "heartbeat": heartbeat,
        "last_success": last_success,
        "interval": TICKER_INTERVAL_SECONDS,
    }


# ═══════════════════════════════════════════════════════════════
# Default Jobs
# ═══════════════════════════════════════════════════════════════

DEFAULT_JOBS = [
    {
        "name": "cleanup_expired_memory",
        "schedule": "0 3 * * *",  # Daily at 3 AM
        "command": "python -m hermes_mobile.cron.cleanup_memory",
        "description": "Clean up expired memory entries",
        "tags": ["maintenance", "memory"],
    },
    {
        "name": "check_updates",
        "schedule": "0 12 * * *",  # Daily at noon
        "command": "python -m hermes_mobile.cron.check_updates",
        "description": "Check for outdated Python packages",
        "tags": ["maintenance", "updates"],
    },
    {
        "name": "backup_data",
        "schedule": "0 4 * * *",  # Daily at 4 AM
        "command": "python -m hermes_mobile.cron.backup_data",
        "description": "Back up local app data",
        "tags": ["backup", "maintenance"],
    },
]


def ensure_default_jobs():
    """Ensure default cron jobs exist."""
    existing = {job.name for job in list_jobs()}

    for job_def in DEFAULT_JOBS:
        if job_def["name"] not in existing:
            create_job(**job_def)
            logger.info(f"Created default cron job: {job_def['name']}")
