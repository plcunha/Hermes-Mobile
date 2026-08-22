"""Tests for standalone cron job scripts."""

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch


class TestCleanupMemory:
    """Tests for cleanup_memory.py"""

    @patch("hermes_mobile.cron.cleanup_memory.get_settings")
    @patch("hermes_mobile.cron.cleanup_memory.MobileMemoryProvider")
    def test_main(self, MockProvider, mock_get_settings, temp_dir):
        mock_settings = MagicMock()
        mock_settings.get_memory_db_path.return_value = str(temp_dir / "memory.db")
        mock_settings.encrypt_memory = False
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_provider.cleanup_expired = AsyncMock()
        MockProvider.return_value = mock_provider

        import asyncio

        from hermes_mobile.cron.cleanup_memory import main

        asyncio.run(main())

        MockProvider.assert_called_once_with(
            db_path=str(temp_dir / "memory.db"),
            encrypt=False,
        )
        mock_provider.cleanup_expired.assert_awaited_once()
        mock_provider.close.assert_called_once()


class TestCheckUpdates:
    """Tests for check_updates.py"""

    @patch("hermes_mobile.cron.check_updates.subprocess.run")
    def test_main_no_updates(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "[]"
        mock_run.return_value = mock_result

        import asyncio

        from hermes_mobile.cron.check_updates import main

        asyncio.run(main())

        mock_run.assert_called_once()

    @patch("hermes_mobile.cron.check_updates.subprocess.run")
    def test_main_with_updates(self, mock_run):
        import json

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            [{"name": "requests", "version": "2.28.0", "latest_version": "2.31.0"}]
        )
        mock_run.return_value = mock_result

        import asyncio

        from hermes_mobile.cron.check_updates import main

        asyncio.run(main())

        mock_run.assert_called_once()

    @patch("hermes_mobile.cron.check_updates.subprocess.run")
    def test_main_pip_fails(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", timeout=30)

        import asyncio

        from hermes_mobile.cron.check_updates import main

        # Should not raise
        asyncio.run(main())
        mock_run.assert_called_once()


class TestBackupData:
    """Tests for backup_data.py"""

    @patch("hermes_mobile.cron.backup_data.get_settings")
    @patch("hermes_mobile.cron.backup_data.shutil.ignore_patterns")
    @patch("hermes_mobile.cron.backup_data.shutil.copytree")
    @patch("hermes_mobile.cron.backup_data.shutil.rmtree")
    def test_main_creates_backup(
        self, mock_rmtree, mock_copytree, mock_ignore, mock_get_settings, temp_dir
    ):
        mock_settings = MagicMock()
        mock_settings.get_data_dir.return_value = temp_dir
        mock_get_settings.return_value = mock_settings

        import asyncio

        from hermes_mobile.cron.backup_data import main

        asyncio.run(main())

        assert mock_copytree.call_count == 1
        args, _ = mock_copytree.call_args
        assert args[0] == temp_dir
        assert "hermes_backup_" in str(args[1])
        mock_ignore.assert_called_once()
        # First run, no old backups to clean up
        mock_rmtree.assert_not_called()

    @patch("hermes_mobile.cron.backup_data.get_settings")
    @patch("hermes_mobile.cron.backup_data.shutil.copytree")
    @patch("hermes_mobile.cron.backup_data.shutil.rmtree")
    def test_main_cleans_old_backups(self, mock_rmtree, mock_copytree, mock_get_settings, temp_dir):
        backup_dir = temp_dir / "backups"
        backup_dir.mkdir()
        # Create 10 old backup dirs
        for i in range(10):
            (backup_dir / f"hermes_backup_20240101_{i:04d}").mkdir()

        mock_settings = MagicMock()
        mock_settings.get_data_dir.return_value = temp_dir
        mock_get_settings.return_value = mock_settings

        import asyncio

        from hermes_mobile.cron.backup_data import main

        asyncio.run(main())

        # Should remove 3 oldest (keep last 7)
        assert mock_rmtree.call_count == 3

    @patch("hermes_mobile.cron.backup_data.get_settings")
    @patch("hermes_mobile.cron.backup_data.shutil.copytree")
    def test_main_backup_failure(self, mock_copytree, mock_get_settings, temp_dir):
        mock_settings = MagicMock()
        mock_settings.get_data_dir.return_value = temp_dir
        mock_get_settings.return_value = mock_settings

        mock_copytree.side_effect = PermissionError("Access denied")

        import asyncio

        from hermes_mobile.cron.backup_data import main

        # Should not raise
        asyncio.run(main())
        mock_copytree.assert_called_once()

    @patch("hermes_mobile.cron.backup_data.get_settings")
    @patch("hermes_mobile.cron.backup_data.shutil.ignore_patterns")
    @patch("hermes_mobile.cron.backup_data.shutil.copytree")
    def test_main_ignores_specific_patterns(
        self, mock_copytree, mock_ignore, mock_get_settings, temp_dir
    ):
        mock_settings = MagicMock()
        mock_settings.get_data_dir.return_value = temp_dir
        mock_get_settings.return_value = mock_settings

        import asyncio

        from hermes_mobile.cron.backup_data import main

        asyncio.run(main())

        mock_ignore.assert_called_once_with("backups", "*.log", "__pycache__")
