"""Hermes Mobile Theme — visual identity matching Hermes Desktop "nous".

The desktop app (apps/desktop in hermes-agent) owns the visual contract. Its
canonical theme is "nous": glass neutrals with Nous blue accents in light mode
and a deep blue-violet surface with psyche cream foreground in dark mode.

This module maps that palette onto Flet's Material 3 ColorScheme so the mobile
app reads as the same product: same brand blue, same warm cream primary on the
dark surface, same flat, hairline-based chrome.
"""

from __future__ import annotations

import inspect

import flet as ft

HERMES_TEXT_FONT = "Rules"
HERMES_DISPLAY_FONT = "Sigurd"
HERMES_MONO_FONT = "Courier Prime"

# --- Canonical desktop tokens (src/themes/presets.ts) ---
NOUS_BLUE = "#0053FD"
PSYCHE_BLUE = "#1540B1"
PSYCHE_WARM = "#FFE6CB"


def _supported_kwargs(cls: type, values: dict) -> dict:
    """Return kwargs supported by a Flet dataclass-style control/theme class.

    Flet 0.86 exposes some constructors as (*args, **kwargs), so filtering only
    by inspect.signature() drops every real theme token. Prefer dataclass fields
    when present, and treat VAR_KEYWORD as support for all provided values.
    """
    fields = set(getattr(cls, "__dataclass_fields__", {}) or {})
    if fields:
        return {key: value for key, value in values.items() if key in fields}
    params = inspect.signature(cls).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return values
    return {key: value for key, value in values.items() if key in params}


DARK = {
    "background": "#0D2F86",
    "foreground": PSYCHE_WARM,
    "card": "#12378F",
    "card_foreground": PSYCHE_WARM,
    "muted": "#183F9A",
    "muted_foreground": "#B5C7F3",
    "popover": "#123A96",
    "primary": PSYCHE_WARM,
    "primary_foreground": "#0D2F86",
    "secondary": "#1B45A4",
    "secondary_foreground": "#E0E8FF",
    "accent": PSYCHE_BLUE,
    "accent_foreground": "#F0F4FF",
    "border": "#3158AD",
    "input": "#0B2566",
    "ring": PSYCHE_WARM,
    "destructive": "#C0473A",
    "destructive_foreground": "#FEF2F2",
    "sidebar": "#09286F",
    "sidebar_border": "#234A9C",
    "user_bubble": "#143B91",
    "user_bubble_border": "#3A63BD",
    "composer": "#0B286E",
    "composer_border": "#3158AD",
    "success": "#6EE7A8",
}

LIGHT = {
    "background": "#F8FAFF",
    "foreground": "#17171A",
    "card": "#FFFFFF",
    "card_foreground": "#17171A",
    "muted": "#EBF0FF",
    "muted_foreground": "#666678",
    "popover": "#FFFFFF",
    "primary": NOUS_BLUE,
    "primary_foreground": "#FCFCFC",
    "secondary": "#E8EEFF",
    "secondary_foreground": "#242432",
    "accent": "#E6EEFF",
    "accent_foreground": "#202030",
    "border": "#D7E2FC",
    "input": "#D3DFFA",
    "ring": NOUS_BLUE,
    "destructive": "#C72E4D",
    "destructive_foreground": "#FFFFFF",
    "sidebar": "#F3F7FF",
    "sidebar_border": "#E0E9FB",
    "user_bubble": "#EDF2FF",
    "user_bubble_border": "#D7E2FC",
    "composer": "#FFFFFF",
    "composer_border": "#D7E2FC",
    "success": "#16845B",
}


def _scheme(c: dict) -> ft.ColorScheme:
    """Map the desktop token dict onto a Material 3 ColorScheme."""
    values = dict(
        primary=c["primary"],
        on_primary=c["primary_foreground"],
        primary_container=c["accent"],
        on_primary_container=c["accent_foreground"],
        secondary=c["secondary"],
        on_secondary=c["secondary_foreground"],
        secondary_container=c["muted"],
        on_secondary_container=c["muted_foreground"],
        tertiary=c["ring"],
        on_tertiary=c["background"],
        error=c["destructive"],
        on_error=c["destructive_foreground"],
        error_container=c["destructive"],
        on_error_container=c["destructive_foreground"],
        surface=c["background"],
        on_surface=c["foreground"],
        on_surface_variant=c["muted_foreground"],
        surface_container_lowest=c["input"],
        surface_container_low=c["popover"],
        surface_container=c["muted"],
        surface_container_high=c["secondary"],
        surface_container_highest=c["card"],
        outline=c["border"],
        outline_variant=c["sidebar_border"],
        shadow="#000000",
        scrim="#000000",
        inverse_surface=c["foreground"],
        on_inverse_surface=c["background"],
        inverse_primary=c["accent"],
        surface_tint=c["primary"],
    )
    return ft.ColorScheme(**_supported_kwargs(ft.ColorScheme, values))


def build_theme(dark: bool = False) -> ft.Theme:
    """Build a Flet Theme for the requested color mode."""
    c = DARK if dark else LIGHT
    # The Hermes website ships Rules (UI), Sigurd (display), and Courier Prime
    # (mono). Flet accepts one family name here; register the actual font files
    # on page.fonts before applying this theme.
    font_family = HERMES_TEXT_FONT
    values = dict(
        color_scheme=_scheme(c),
        use_material3=True,
        font_family=font_family,
        scaffold_bgcolor=c["background"],
        canvas_color=c["background"],
        divider_color=c["border"],
        card_bgcolor=c["card"],
        navigation_bar_theme=ft.NavigationBarTheme(
            bgcolor=c["sidebar"],
            indicator_color=c["accent"],
            label_text_style=ft.TextStyle(
                color=c["foreground"], weight=ft.FontWeight.W_500, size=11
            ),
        ),
        navigation_rail_theme=ft.NavigationRailTheme(
            bgcolor=c["sidebar"],
            indicator_color=c["accent"],
            selected_label_text_style=ft.TextStyle(
                color=c["foreground"], weight=ft.FontWeight.W_600
            ),
            unselected_label_text_style=ft.TextStyle(color=c["muted_foreground"]),
        ),
        appbar_theme=ft.AppBarTheme(
            bgcolor=c["sidebar"],
            color=c["foreground"],
            center_title=False,
            elevation=0,
        ),
        card_theme=ft.CardTheme(
            color=c["card"],
            elevation=0,
            margin=0,
            shape=ft.RoundedRectangleBorder(radius=12),
        ),
        divider_theme=ft.DividerTheme(color=c["border"]),
        snackbar_theme=ft.SnackBarTheme(
            bgcolor=c["popover"],
            content_text_style=ft.TextStyle(color=c["foreground"]),
        ),
        text_theme=ft.TextTheme(
            body_medium=ft.TextStyle(color=c["foreground"], size=13, height=1.45),
            body_large=ft.TextStyle(color=c["foreground"], size=14, height=1.45),
            title_medium=ft.TextStyle(color=c["foreground"], size=13, weight=ft.FontWeight.W_600),
            title_large=ft.TextStyle(
                color=c["foreground"],
                size=18,
                weight=ft.FontWeight.W_700,
                font_family=HERMES_DISPLAY_FONT,
            ),
            label_medium=ft.TextStyle(color=c["muted_foreground"], size=11),
        ),
        list_tile_theme=ft.ListTileTheme(
            text_color=c["foreground"],
            icon_color=c["muted_foreground"],
        ),
    )
    supported = set(getattr(ft.Theme, "__dataclass_fields__", {}) or {})
    if not supported:
        supported = set(inspect.signature(ft.Theme).parameters)
    if "card_bgcolor" not in supported and "card_color" in supported:
        values["card_color"] = values.pop("card_bgcolor")
    return ft.Theme(**_supported_kwargs(ft.Theme, values))


def mode_colors(dark: bool = False) -> dict:
    """Expose the raw token dict for the current mode (UI helpers)."""
    return DARK if dark else LIGHT
