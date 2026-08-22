"""Tests for i18n/locales module."""

from pathlib import Path

from hermes_mobile.locales import (
    _AVAILABLE_LOCALES,
    _DEFAULT_LOCALE,
    _count_keys,
    _load_locale,
    _translations,
    available_locales,
    get_locale,
    init,
    set_locale,
    t,
    translate_dict,
)


class TestConstants:
    def test_available_locales(self):
        assert "en" in _AVAILABLE_LOCALES
        assert "pt-br" in _AVAILABLE_LOCALES

    def test_default_locale_is_english(self):
        assert _DEFAULT_LOCALE == "en"


class TestLoadLocale:
    def test_loads_english(self):
        data = _load_locale("en")
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_loads_portuguese(self):
        data = _load_locale("pt-br")
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_unknown_locale_returns_empty(self):
        data = _load_locale("xx")
        assert data == {}

    def test_missing_file_returns_empty(self, monkeypatch):
        monkeypatch.setattr("hermes_mobile.locales._LOCALE_DIR", Path("/nonexistent"))
        assert _load_locale("en") == {}

    def test_corrupt_json_returns_empty(self, monkeypatch, tmp_path):
        """Loading a file with invalid JSON returns empty dict."""
        bad_file = tmp_path / "en.json"
        bad_file.write_text("{invalid json}")
        monkeypatch.setattr("hermes_mobile.locales._LOCALE_DIR", tmp_path)
        result = _load_locale("en")
        assert result == {}

    def test_locale_catalogs_have_identical_key_paths(self):
        def key_paths(data, prefix=""):
            paths = set()
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    paths.update(key_paths(value, path))
                else:
                    paths.add(path)
            return paths

        assert key_paths(_load_locale("en")) == key_paths(_load_locale("pt-br"))


class TestInit:
    def teardown_method(self):
        init("en")

    def test_init_with_valid_locale(self):
        init("pt-br")
        assert get_locale() == "pt-br"

    def test_init_with_invalid_locale_falls_back(self):
        init("xx")
        assert get_locale() == "en"

    def test_init_with_none_uses_default(self):
        init(None)
        assert get_locale() == "en"

    def test_init_loads_translations(self):
        init("pt-br")
        assert t("nav.chat") != "nav.chat"


class TestSetLocale:
    def teardown_method(self):
        init("en")

    def test_set_valid_locale(self):
        assert set_locale("pt-br") is True
        assert get_locale() == "pt-br"

    def test_set_invalid_locale_returns_false(self):
        assert set_locale("xx") is False
        assert get_locale() == "en"  # Unchanged


class TestAvailableLocales:
    def test_returns_tuple(self):
        locs = available_locales()
        assert isinstance(locs, tuple)
        assert "en" in locs

    def test_includes_pt_br(self):
        assert "pt-br" in available_locales()


class TestT:
    def teardown_method(self):
        init("en")

    def test_resolves_dotted_key(self):
        value = t("nav.chat")
        assert isinstance(value, str)
        assert len(value) > 0

    def test_fallback_to_english(self):
        init("pt-br")
        value = t("nav.chat")
        assert isinstance(value, str)
        assert len(value) > 0

    def test_missing_key_returns_key(self):
        value = t("nonexistent.key.here")
        assert value == "nonexistent.key.here"

    def test_format_kwargs(self):
        value = t("chat.tool_calling", tool="web_search")
        assert "web_search" in value

    def test_format_kwargs_missing_variable_does_not_crash(self):
        """When kwargs are missing a variable in the template, it falls back gracefully."""

        _translations["test_greeting"] = "Hello {name}"
        result = t("test_greeting", wrong_var="foo")
        assert isinstance(result, str)
        del _translations["test_greeting"]

    def test_empty_translations_returns_key(self, monkeypatch):
        monkeypatch.setattr("hermes_mobile.locales._translations", {})
        assert t("nav.chat") == "nav.chat"

    def test_nested_key_resolution(self):
        value = t("nav.chat")
        assert value != "nav.chat"

    def test_common_save_is_translated_in_both_locales(self):
        init("en")
        assert t("common.save") == "Save"
        init("pt-br")
        assert t("common.save") == "Salvar"

    def test_gateway_connection_labels_resolve_in_both_locales(self):
        init("en")
        assert t("gateway.offline") == "Offline"
        assert t("gateway.online") == "Online"
        init("pt-br")
        assert t("gateway.offline") == "Offline"
        assert t("gateway.online") == "Online"


class TestTranslateDict:
    def teardown_method(self):
        init("en")

    def test_translates_values(self):
        data = {"title": "nav.chat"}
        result = translate_dict(data)
        assert result["title"] != "nav.chat"

    def test_passes_through_normal_strings(self):
        data = {"title": "Hello World"}
        result = translate_dict(data)
        assert result["title"] == "Hello World"

    def test_handles_nested_dicts(self):
        data = {"menu": {"item": "nav.chat"}}
        result = translate_dict(data)
        assert result["menu"]["item"] != "nav.chat"

    def test_handles_lists(self):
        data = {"items": ["nav.chat", "nav.settings"]}
        result = translate_dict(data)
        assert len(result["items"]) == 2
        assert all(i != k for i, k in zip(result["items"], ["nav.chat", "nav.settings"]))

    def test_skips_format_strings(self):
        """Strings starting with { are likely template fragments, not keys."""
        data = {"text": "{variable}"}
        result = translate_dict(data)
        assert result["text"] == "{variable}"

    def test_translate_dict_uses_locale(self):
        data = {"title": "nav.chat"}
        result = translate_dict(data, locale="pt-br")
        assert result["title"] != "nav.chat"
        assert get_locale() == "en"  # Original locale restored


class TestCountKeys:
    def test_counts_flat_dict(self):
        assert _count_keys({"a": 1, "b": 2}) == 2

    def test_counts_nested_keys(self):
        assert _count_keys({"a": {"b": 1, "c": 2}, "d": 3}) == 3
