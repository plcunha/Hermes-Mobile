"""Hermes Mobile Configuration Management"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _get_default_data_dir() -> str:
    """Resolve a durable writable data directory across desktop and native builds."""
    # Flet 0.86 exposes the platform application-support directory synchronously.
    # On Android this is /data/data/<package>/files/data; Path.home() is /data and
    # must never be used because it is outside the app sandbox.
    flet_storage = os.environ.get("FLET_APP_STORAGE_DATA")
    if flet_storage:
        return str(Path(flet_storage).expanduser())

    try:
        home = Path.home()
        if home.exists() and os.access(home, os.W_OK):
            return str(home / ".hermes_mobile")
    except (OSError, RuntimeError):
        pass

    try:
        cwd = Path.cwd()
        if cwd.exists() and os.access(cwd, os.W_OK):
            return str(cwd / ".hermes_mobile")
    except OSError:
        pass

    # Last-resort scratch path keeps startup functional on unusual embedded
    # runtimes. Packaged Flet apps should always take the durable env path above.
    return str(Path(tempfile.gettempdir()) / "hermes_mobile")


class HermesMobileSettings(BaseSettings):
    """Main settings for Hermes Mobile app"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App settings
    app_name: str = "Hermes Mobile"
    app_version: str = "0.1.0"
    debug: bool = False

    # Data directory
    data_dir: str = Field(default_factory=_get_default_data_dir)

    # AI Provider settings
    default_provider: str = "openrouter"
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Model settings
    default_model: str = "anthropic/claude-3.5-sonnet"
    fallback_models: list[str] = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-1.5-pro",
    ]

    # Agent settings
    max_iterations: int = 20
    max_tokens: int = 8192
    temperature: float = 0.7
    system_prompt: str = """You are Hermes, a helpful AI assistant running on a mobile device.
You have access to various tools and can help with a wide range of tasks.
Be concise but thorough. Use tools when appropriate."""

    # Memory settings
    memory_enabled: bool = True
    memory_db_path: Optional[str] = None
    max_memory_entries: int = 10000
    memory_ttl_days: int = 30

    # Skills settings
    skills_enabled: bool = True
    skills_dir: Optional[str] = None
    auto_install_skills: bool = False

    # Cron/Scheduler settings
    cron_enabled: bool = True
    cron_check_interval_seconds: int = 60

    # Gateway settings
    gateway_enabled: bool = False
    gateway_port: int = 8080
    push_notifications_enabled: bool = True

    # Runtime backend: local embedded agent or a Desktop-compatible Hermes
    # remote backend (``hermes serve`` / dashboard JSON-RPC gateway).
    runtime_mode: str = "local"
    remote_url: str = ""
    remote_auth_mode: str = "auto"
    remote_username: str = ""
    remote_profile: str = ""
    remote_allow_insecure: bool = False

    # Home Assistant integration (smart-home tool parity). Secrets stay in
    # .env as HA_URL / HA_TOKEN; never persisted to settings JSON.
    ha_url: str = ""
    ha_token: str = ""

    # UI settings
    theme: str = "system"  # light, dark, system
    language: str = "en"
    font_size: int = 16
    pet_roam: bool = True
    show_tool_calls: bool = True
    stream_responses: bool = True

    # Network settings
    request_timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0

    # Security
    encrypt_memory: bool = True
    biometric_auth: bool = False
    auto_lock_minutes: int = 5

    def get_data_dir(self) -> Path:
        """Get the data directory as a Path object"""
        path = Path(self.data_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_skills_dir(self) -> Path:
        """Get the skills directory as a Path object"""
        if self.skills_dir:
            path = Path(self.skills_dir).expanduser()
        else:
            path = self.get_data_dir() / "skills"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_memory_db_path(self) -> Path:
        """Get the memory database path"""
        if self.memory_db_path:
            return Path(self.memory_db_path).expanduser()
        return self.get_data_dir() / "memory.db"

    def get_config_dir(self) -> Path:
        """Get the config directory"""
        path = self.get_data_dir() / "config"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _PERSISTED_FIELDS = (
        "default_provider",
        "default_model",
        "max_iterations",
        "max_tokens",
        "temperature",
        "memory_enabled",
        "max_memory_entries",
        "memory_ttl_days",
        "skills_enabled",
        "cron_enabled",
        "gateway_enabled",
        "gateway_port",
        "push_notifications_enabled",
        "runtime_mode",
        "remote_url",
        "remote_auth_mode",
        "remote_username",
        "remote_profile",
        "remote_allow_insecure",
        "theme",
        "language",
        "font_size",
        "pet_roam",
        "show_tool_calls",
        "stream_responses",
        "request_timeout",
        "max_retries",
        "encrypt_memory",
    )

    def settings_file(self) -> Path:
        """Path to the persisted settings JSON."""
        return self.get_config_dir() / "settings.json"

    def to_dict(self) -> dict:
        """Serializable dict of persisted fields (secrets excluded by default)."""
        return {k: getattr(self, k) for k in self._PERSISTED_FIELDS}

    def load_persisted(self) -> "HermesMobileSettings":
        """Overlay persisted JSON values onto this instance (env still wins)."""
        path = self.settings_file()
        if not path.exists():
            return self
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            legacy_keys = {
                "openrouter_api_key": "openrouter",
                "openai_api_key": "openai",
                "anthropic_api_key": "anthropic",
                "gemini_api_key": "google",
            }
            migrated = False
            pending = {
                provider: str(data.get(field) or "").strip()
                for field, provider in legacy_keys.items()
                if str(data.get(field) or "").strip()
            }
            if pending:
                from hermes_mobile.remote.secrets import ProviderSecretStore

                secret_store = ProviderSecretStore(self.get_data_dir())
                for provider, api_key in pending.items():
                    secret_store.save_key(provider, api_key)
                migrated = True
            for k, v in data.items():
                if k in self._PERSISTED_FIELDS and v is not None:
                    setattr(self, k, v)
            if migrated:
                save_settings(self)
        except Exception as e:
            logger.warning("Failed to load persisted settings: %s", e)
        return self


def save_settings(s: HermesMobileSettings) -> bool:
    """Persist runtime settings atomically and report whether the write succeeded."""
    fd = -1
    raw_tmp = ""
    try:
        path = s.settings_file()
        payload = json.dumps(s.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")
        fd, raw_tmp = tempfile.mkstemp(prefix="settings-", dir=path.parent)
        os.write(fd, payload)
        os.fchmod(fd, 0o600)
        os.close(fd)
        fd = -1
        os.replace(raw_tmp, path)
        raw_tmp = ""
        return True
    except Exception as e:
        logger.warning("Failed to save settings: %s", e)
        return False
    finally:
        if fd >= 0:
            os.close(fd)
        if raw_tmp:
            try:
                Path(raw_tmp).unlink(missing_ok=True)
            except OSError:
                pass


# Global settings instance
settings = HermesMobileSettings()


def get_settings() -> HermesMobileSettings:
    """Get the global settings instance (env first, then persisted JSON)."""
    return settings.load_persisted()


def reload_settings() -> HermesMobileSettings:
    """Reload settings from environment and persisted JSON"""
    global settings
    settings = HermesMobileSettings().load_persisted()
    return settings
