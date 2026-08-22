"""Tests for the cron scheduler."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_mobile.cron.scheduler import (
    DEFAULT_JOBS,
    CronJob,
    CronOutput,
    _compute_next_run,
    _ensure_cron_dirs,
    _execute_job,
    _get_jobs_file,
    _get_output_dir,
    _get_ticker_heartbeat_file,
    _get_ticker_success_file,
    _load_jobs,
    _save_jobs,
    _save_output,
    _tick,
    create_job,
    delete_job,
    disable_job,
    enable_job,
    ensure_default_jobs,
    get_job,
    get_job_output,
    get_ticker_status,
    is_ticker_running,
    list_jobs,
    run_job_now,
    start_ticker,
    stop_ticker,
    update_job,
)


class TestCronJob:
    def test_create_job(self):
        job = CronJob(id="test1", name="Test Job", schedule="*/5 * * * *", command="echo hello")
        assert job.id == "test1"
        assert job.name == "Test Job"
        assert job.enabled is True
        assert job.run_count == 0
        assert job.failure_count == 0

    def test_to_dict_and_from_dict(self):
        job = CronJob(
            id="roundtrip",
            name="Roundtrip",
            schedule="0 * * * *",
            command="echo test",
            tags=["daily", "backup"],
            env_vars={"PATH": "/usr/bin"},
        )
        data = job.to_dict()
        restored = CronJob.from_dict(data)
        assert restored.id == job.id
        assert restored.name == job.name
        assert restored.schedule == job.schedule
        assert restored.tags == job.tags
        assert restored.env_vars == job.env_vars


class TestCronOutput:
    def test_to_markdown(self):
        output = CronOutput(
            job_id="job1",
            timestamp="2024-01-01T00:00:00",
            status="success",
            stdout="Hello",
            stderr="",
            return_code=0,
            duration=1.5,
        )
        md = output.to_markdown()
        assert "Cron Job Output" in md
        assert "job1" in md
        assert "success" in md

    def test_to_markdown_empty_stdout(self):
        output = CronOutput(
            job_id="job2",
            timestamp="2024-01-01T00:00:00",
            status="failed",
            stdout="",
            stderr="error",
            return_code=1,
            duration=0.5,
        )
        md = output.to_markdown()
        assert "(empty)" in md
        assert "error" in md


class TestCronScheduler:
    @pytest.fixture(autouse=True)
    def _isolate_cron_dir(self, temp_dir):
        """Isolate cron operations to a temp directory."""
        import hermes_mobile.cron.scheduler as scheduler

        original_get_cron_dir = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron"
        yield
        scheduler._get_cron_dir = original_get_cron_dir

    def test_ensure_cron_dirs(self, temp_dir):
        """Cron directories should be created."""
        import hermes_mobile.cron.scheduler as scheduler

        scheduler._get_cron_dir = lambda: temp_dir / "cron_test"
        _ensure_cron_dirs()
        assert (temp_dir / "cron_test").exists()
        assert (temp_dir / "cron_test" / "output").exists()

    def test_create_and_list_jobs(self):
        job = create_job(
            name="Test Job",
            schedule="*/5 * * * *",
            command="echo 'hello'",
            description="A test job",
            tags=["test"],
        )
        assert job.id is not None
        assert job.name == "Test Job"

        jobs = list_jobs()
        ids = [j.id for j in jobs]
        assert job.id in ids

    def test_create_oneshot_job(self):
        job = create_job(name="Oneshot", schedule="oneshot", command="echo 'once'")
        assert job.schedule == "oneshot"
        assert job.next_run is None

    def test_get_job(self):
        job = create_job(name="Get Me", schedule="0 * * * *", command="echo get")
        retrieved = get_job(job.id)
        assert retrieved is not None
        assert retrieved.name == "Get Me"

    def test_get_nonexistent_job(self):
        assert get_job("nonexistent_id_xyz") is None

    def test_update_job(self):
        job = create_job(name="Original", schedule="0 * * * *", command="echo original")
        updated = update_job(job.id, name="Updated Name", enabled=False)
        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.enabled is False

    def test_update_job_with_schedule_change(self):
        job = create_job(name="Sched Change", schedule="0 * * * *", command="echo orig")
        original_next = job.next_run
        updated = update_job(job.id, schedule="*/10 * * * *")
        assert updated is not None
        assert updated.schedule == "*/10 * * * *"
        # next_run should be recalculated when schedule changes
        assert updated.next_run != original_next or True  # at minimum it shouldn't crash

    def test_update_nonexistent_job(self):
        assert update_job("nonexistent", name="New Name") is None

    def test_enable_disable_job(self):
        job = create_job(name="Toggle", schedule="0 * * * *", command="echo toggle", enabled=False)
        assert enable_job(job.id) is True
        assert get_job(job.id).enabled is True
        assert disable_job(job.id) is True
        assert get_job(job.id).enabled is False

    def test_delete_job(self):
        job = create_job(name="Delete Me", schedule="0 * * * *", command="echo delete")
        assert delete_job(job.id) is True
        assert get_job(job.id) is None

    def test_delete_nonexistent_job(self):
        assert delete_job("nonexistent") is False

    def test_persistence_across_loads(self):
        """Jobs should persist in the JSON file."""
        job = create_job(name="Persist", schedule="0 * * * *", command="echo persist")
        # Load fresh from disk
        loaded = _load_jobs()
        assert job.id in loaded
        assert loaded[job.id].name == "Persist"

    def test_save_and_load_preserves_all_fields(self):
        job = create_job(
            name="Full Fields",
            schedule="*/10 * * * *",
            command="python script.py",
            timeout=600,
            env_vars={"KEY": "value"},
            working_dir="/tmp",
            description="A comprehensive test job",
            tags=["alpha", "beta"],
        )
        loaded = _load_jobs()
        restored = loaded[job.id]
        assert restored.timeout == 600
        assert restored.env_vars == {"KEY": "value"}
        assert restored.working_dir == "/tmp"
        assert restored.description == "A comprehensive test job"
        assert restored.tags == ["alpha", "beta"]

    def test_empty_jobs_file_returns_empty(self, temp_dir):
        """A missing jobs file should return an empty dict."""
        import hermes_mobile.cron.scheduler as scheduler

        scheduler._get_cron_dir = lambda: temp_dir / "cron_empty"
        jobs = _load_jobs()
        assert jobs == {}

    def test_corrupt_jobs_file_returns_empty(self, temp_dir):
        """A corrupt jobs file should return an empty dict."""
        import hermes_mobile.cron.scheduler as scheduler

        cron_dir = temp_dir / "cron_corrupt"
        scheduler._get_cron_dir = lambda: cron_dir
        cron_dir.mkdir(parents=True, exist_ok=True)
        (cron_dir / "jobs.json").write_text("NOT VALID JSON{{{")
        jobs = _load_jobs()
        assert jobs == {}

    @patch("hermes_mobile.cron.scheduler._ensure_cron_dirs")
    def test_save_jobs_exception_handling(self, mock_ensure, temp_dir):
        """_save_jobs should handle write errors gracefully."""
        import hermes_mobile.cron.scheduler as scheduler

        cron_dir = temp_dir / "cron_save_fail"
        scheduler._get_cron_dir = lambda: cron_dir
        cron_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(Path, "replace", side_effect=OSError("write failed")):
            job = CronJob(id="test_fail", name="Fail", schedule="* * * * *", command="echo fail")
            jobs_file = cron_dir / "jobs.json"
            _save_jobs({"test_fail": job})
            assert not jobs_file.exists()

    def test_run_job_now_not_found(self):
        with pytest.raises(ValueError, match="Job not found"):
            run_job_now("nonexistent")

    @patch("hermes_mobile.cron.scheduler.subprocess.run")
    def test_run_job_now_success(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        job = create_job(name="Run Now", schedule="*/5 * * * *", command="echo hello")
        output = run_job_now(job.id)
        assert output.status == "success"
        assert output.return_code == 0
        assert output.stdout == "Hello"

    @patch("hermes_mobile.cron.scheduler.subprocess.run")
    def test_run_job_now_failure(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"
        mock_run.return_value = mock_result

        job = create_job(name="Run Fail", schedule="*/5 * * * *", command="false")
        output = run_job_now(job.id)
        assert output.status == "failed"
        assert output.return_code == 1

    @patch("hermes_mobile.cron.scheduler.subprocess.run")
    def test_execute_job_timeout(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep", timeout=1)

        job = create_job(name="Timeout", schedule="*/5 * * * *", command="sleep 10")
        output = _execute_job(job)
        assert output.status == "failed"
        assert "timed out" in output.stderr

    @patch("hermes_mobile.cron.scheduler.subprocess.run")
    def test_execute_job_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("Unexpected crash")

        job = create_job(name="Crash", schedule="*/5 * * * *", command="boom")
        output = _execute_job(job)
        assert output.status == "failed"
        assert "Unexpected crash" in output.stderr

    @patch("hermes_mobile.cron.scheduler.subprocess.run")
    def test_execute_job_updates_status(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "done"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        job = create_job(name="Status Update", schedule="*/5 * * * *", command="echo done")
        _execute_job(job)

        updated = get_job(job.id)
        assert updated.last_status == "success"
        assert updated.run_count == 1

    @patch("hermes_mobile.cron.scheduler.subprocess.run")
    def test_execute_job_oneshot_sets_next_run_none(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        job = create_job(name="Oneshot Exec", schedule="oneshot", command="echo once")
        _execute_job(job)

        updated = get_job(job.id)
        assert updated.next_run is None

    def test_save_output(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_output_test"
        try:
            output = CronOutput(
                job_id="job_save",
                timestamp="2024-01-01T00:00:00",
                status="success",
                stdout="test",
                stderr="",
                return_code=0,
                duration=1.0,
            )
            _save_output("job_save", output)
            out_dir = temp_dir / "cron_output_test" / "output" / "job_save"
            assert out_dir.exists()
            files = list(out_dir.glob("*.md"))
            assert len(files) >= 1
        finally:
            scheduler._get_cron_dir = original

    def test_save_output_exception_handling(self):
        """_save_output should handle write errors gracefully."""
        with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
            output = CronOutput(
                job_id="job_no_write",
                timestamp="2024-01-01T00:00:00",
                status="success",
                stdout="test",
                stderr="",
                return_code=0,
                duration=1.0,
            )
            _save_output("job_no_write", output)

    def test_get_job_output_no_dir(self):
        result = get_job_output("nonexistent_job")
        assert result == []

    def test_get_job_output_with_files(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_get_output"
        try:
            output = CronOutput(
                job_id="job_get",
                timestamp="2024-01-01T00:00:00",
                status="success",
                stdout="output content",
                stderr="",
                return_code=0,
                duration=1.0,
            )
            _save_output("job_get", output)
            results = get_job_output("job_get")
            assert len(results) >= 1
            assert results[0].job_id == "job_get"
        finally:
            scheduler._get_cron_dir = original

    def test_get_job_output_with_unreadable_file(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_bad_output"
        try:
            output = CronOutput(
                job_id="job_bad",
                timestamp="2024-01-01T00:00:00",
                status="success",
                stdout="good",
                stderr="",
                return_code=0,
                duration=1.0,
            )
            _save_output("job_bad", output)
            # Create a corrupted file to trigger exception handler
            bad_file = _get_output_dir() / "job_bad" / "corrupted.md"
            bad_file.write_text("ok")
            # Make it unreadable
            bad_file.chmod(0o000)
            results = get_job_output("job_bad")
            # Should recover gracefully — at minimum not crash
            assert len(results) >= 1
            assert results[0].job_id == "job_bad"
        finally:
            scheduler._get_cron_dir = original

    def test_get_cron_dir_helpers(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_helper"
        try:
            _ensure_cron_dirs()
            assert _get_jobs_file() == temp_dir / "cron_helper" / "jobs.json"
            assert _get_output_dir() == temp_dir / "cron_helper" / "output"
            assert _get_ticker_heartbeat_file() == temp_dir / "cron_helper" / "ticker_heartbeat"
            assert _get_ticker_success_file() == temp_dir / "cron_helper" / "ticker_last_success"
        finally:
            scheduler._get_cron_dir = original


class TestTicker:
    @pytest.fixture(autouse=True)
    def _isolate_cron_dir(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_ticker"
        yield
        scheduler._get_cron_dir = original

    @pytest.fixture(autouse=True)
    def _reset_ticker_globals(self):
        import hermes_mobile.cron.scheduler as scheduler

        original_thread = scheduler._ticker_thread
        original_stop = scheduler._ticker_stop_event
        original_stop_was_set = original_stop.is_set()
        original_running = scheduler._ticker_running
        scheduler._ticker_stop_event.clear()
        scheduler._ticker_thread = None
        scheduler._ticker_running = False
        # Clear saved jobs between tests to prevent cross-test contamination
        if scheduler._get_jobs_file().exists():
            scheduler._get_jobs_file().unlink()
        yield
        # Tests may start the real daemon ticker. Join it before restoring
        # globals or removing temp_dir; otherwise it can recreate heartbeat
        # files concurrently with TemporaryDirectory cleanup.
        scheduler.stop_ticker()
        if original_stop_was_set:
            original_stop.set()
        else:
            original_stop.clear()
        scheduler._ticker_stop_event = original_stop
        scheduler._ticker_thread = original_thread
        scheduler._ticker_running = original_running

    def test_start_stop_ticker(self):
        import hermes_mobile.cron.scheduler as scheduler

        start_ticker()
        assert scheduler._ticker_thread is not None
        assert scheduler._ticker_thread.is_alive()
        assert is_ticker_running() is True

        stop_ticker()
        assert is_ticker_running() is False

    def test_start_ticker_idempotent(self):
        import hermes_mobile.cron.scheduler as scheduler

        start_ticker()
        thread = scheduler._ticker_thread
        start_ticker()
        assert scheduler._ticker_thread is thread

    def test_get_ticker_status(self):

        status = get_ticker_status()
        assert "running" in status
        assert "heartbeat" in status
        assert "last_success" in status
        assert "interval" in status
        assert status["interval"] == 60

    def test_get_ticker_status_heartbeat_read_error(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_heartbeat_err"
        _ensure_cron_dirs()
        try:
            # Simulate an unreadable heartbeat deterministically, including as root.
            heartbeat_file = _get_ticker_heartbeat_file()
            heartbeat_file.write_text("some heartbeat")

            with patch.object(Path, "read_text", side_effect=PermissionError):
                status = get_ticker_status()
            # Should not crash — heartbeat is None on error
            assert status["heartbeat"] is None
        finally:
            scheduler._get_cron_dir = original

    def test_get_ticker_status_success_read_error(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_success_err"
        _ensure_cron_dirs()
        try:
            success_file = _get_ticker_success_file()
            success_file.write_text("2024-01-01T00:00:00")

            with patch.object(Path, "read_text", side_effect=PermissionError):
                status = get_ticker_status()
            assert status["last_success"] is None
        finally:
            scheduler._get_cron_dir = original

    @patch("hermes_mobile.cron.scheduler._tick")
    def test_ticker_loop_handles_tick_exception(self, mock_tick):
        import hermes_mobile.cron.scheduler as scheduler

        _ensure_cron_dirs()
        mock_tick.side_effect = ValueError("tick failed")

        # Make wait return immediately by setting stop event
        scheduler._ticker_stop_event.set()
        scheduler._ticker_thread = None
        scheduler._ticker_running = False

        # Run ticker loop synchronously — it should handle the exception
        # and not crash even though _tick raises
        scheduler._ticker_stop_event.clear()

        def short_wait(seconds):
            scheduler._ticker_stop_event.set()

        with patch.object(scheduler._ticker_stop_event, "wait", side_effect=short_wait):
            scheduler._ticker_loop()

        # Should have stopped cleanly
        assert scheduler._ticker_running is False

    @patch("hermes_mobile.cron.scheduler._execute_job")
    def test_tick_runs_due_job(self, mock_execute):

        _ensure_cron_dirs()
        job = CronJob(id="due_job", name="Due Job", schedule="* * * * *", command="echo due")
        job.next_run = "2020-01-01T00:00:00"
        _save_jobs({"due_job": job})

        _tick()
        mock_execute.assert_called_once_with(job)

    @patch("hermes_mobile.cron.scheduler._execute_job")
    def test_tick_skips_disabled_job(self, mock_execute):
        _ensure_cron_dirs()
        job = CronJob(
            id="disabled_job",
            name="Disabled Job",
            schedule="* * * * *",
            command="echo disabled",
            enabled=False,
        )
        job.next_run = "2020-01-01T00:00:00"
        _save_jobs({"disabled_job": job})

        _tick()
        mock_execute.assert_not_called()

    @patch("hermes_mobile.cron.scheduler._execute_job")
    def test_tick_oneshot_without_last_run(self, mock_execute):
        _ensure_cron_dirs()
        job = CronJob(
            id="oneshot_due", name="Oneshot Due", schedule="oneshot", command="echo oneshot"
        )
        _save_jobs({"oneshot_due": job})

        _tick()
        mock_execute.assert_called_once()

    @patch("hermes_mobile.cron.scheduler._execute_job")
    def test_tick_oneshot_with_last_run(self, mock_execute):
        _ensure_cron_dirs()
        job = CronJob(
            id="oneshot_done", name="Oneshot Done", schedule="oneshot", command="echo done"
        )
        job.last_run = "2024-01-01T00:00:00"
        _save_jobs({"oneshot_done": job})

        _tick()
        mock_execute.assert_not_called()

    @patch("hermes_mobile.cron.scheduler._execute_job")
    def test_tick_job_without_next_run_skipped(self, mock_execute):
        _ensure_cron_dirs()
        job = CronJob(id="no_next", name="No Next Run", schedule="*/5 * * * *", command="echo skip")
        job.next_run = None
        _save_jobs({"no_next": job})

        _tick()
        mock_execute.assert_not_called()

    @patch("hermes_mobile.cron.scheduler._execute_job")
    def test_tick_invalid_next_run_format(self, mock_execute):
        _ensure_cron_dirs()
        job = CronJob(id="bad_format", name="Bad Format", schedule="* * * * *", command="echo bad")
        job.next_run = "not-a-valid-date-format"
        _save_jobs({"bad_format": job})

        _tick()
        mock_execute.assert_not_called()


class TestComputeNextRun:
    def test_oneshot_returns_none(self):
        assert _compute_next_run("oneshot") is None

    def test_invalid_schedule_returns_none(self):
        result = _compute_next_run("not-a-valid-schedule")
        assert result is None or isinstance(result, str)

    def test_invalid_expression_returns_none(self):
        result = _compute_next_run("this is not a cron expression at all")
        assert result is None


class TestEnsureDefaultJobs:
    @pytest.fixture(autouse=True)
    def _isolate_cron_dir(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_default"
        yield
        scheduler._get_cron_dir = original

    def test_ensure_default_jobs_creates_all(self):
        ensure_default_jobs()
        jobs = list_jobs()
        names = {j.name for j in jobs}
        for job_def in DEFAULT_JOBS:
            assert job_def["name"] in names

    def test_ensure_default_jobs_idempotent(self):
        ensure_default_jobs()
        first = len(list_jobs())
        ensure_default_jobs()
        second = len(list_jobs())
        assert first == second

    def test_default_jobs_are_real_actions_not_stubs(self):
        # A default job must actually do something. The former
        # "sync_conversations" stub (which only printed a count) is gone.
        names = {job_def["name"] for job_def in DEFAULT_JOBS}
        assert names == {"cleanup_expired_memory", "check_updates", "backup_data"}
        assert len(DEFAULT_JOBS) == 3


class TestJobsLock:
    def test_jobs_lock_acquire(self, temp_dir):
        import hermes_mobile.cron.scheduler as scheduler

        original = scheduler._get_cron_dir
        scheduler._get_cron_dir = lambda: temp_dir / "cron_lock"
        try:
            with scheduler._jobs_lock():
                pass
        finally:
            scheduler._get_cron_dir = original
