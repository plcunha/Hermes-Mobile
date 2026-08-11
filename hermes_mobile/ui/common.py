"""Shared visual primitives for Hermes Mobile.

The desktop contract is the source of truth: flat over boxed, tokens over
literals, one hairline per structural boundary, and the real Nous brand mark.
These helpers keep every Flet view on that contract instead of drifting into
Material-card defaults.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import flet as ft

from hermes_mobile.ui.theme import HERMES_DISPLAY_FONT, HERMES_MONO_FONT, mode_colors

# Official Hermes website mono face. Fallback is handled by Flutter if the
# asset cannot be loaded in a non-packaged test environment.
MONO_FONT = HERMES_MONO_FONT


def snack(page: ft.Page, text: str, error: bool = False):
    """Show a themed snack bar."""
    c = mode_colors(getattr(page, "theme_mode", None) == ft.ThemeMode.DARK)
    platform = str(
        getattr(getattr(page, "platform", None), "value", getattr(page, "platform", ""))
    ).lower()
    width = getattr(page, "width", None)
    is_phone = platform in {"android", "ios"} or (
        isinstance(width, (int, float)) and 0 < width <= 600
    )
    content = ft.Text(text, color=c["destructive"] if error else c["foreground"])
    sb = ft.SnackBar(
        content=content,
        bgcolor=c["popover"],
        behavior=ft.SnackBarBehavior.FLOATING,
        elevation=0,
        margin=ft.Margin.only(left=12, right=12, bottom=96 if is_phone else 12),
        shape=ft.RoundedRectangleBorder(radius=8),
    )
    sb.open = True
    page.overlay.append(sb)
    page.update()


def open_dialog(page: ft.Page, dialog: ft.AlertDialog):
    """Open a dialog using the current Flet API."""
    page.show_dialog(dialog)


def close_dialog(page: ft.Page, dialog: ft.AlertDialog):
    """Close a dialog using the current Flet API."""
    dialog.open = False
    page.update()


def brand_mark(size: int = 32) -> ft.Control:
    """Canonical Hermes application mark shared with the Desktop launcher."""
    return ft.Container(
        content=ft.Image(
            src="icon.png",
            width=size,
            height=size,
            fit=ft.BoxFit.CONTAIN,
        ),
        width=size,
        height=size,
        bgcolor="#FFFFFF",
        border_radius=ft.BorderRadius.all(max(4, size // 7)),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )


def hermes_mascot(size: int = 144) -> ft.Control:
    """Canonical Hermes pixel mascot for spacious identity moments."""
    return ft.Image(
        src="hermes-mascot.png",
        width=size,
        height=size,
        fit=ft.BoxFit.CONTAIN,
        filter_quality=ft.FilterQuality.NONE,
        semantics_label="Hermes",
    )


def hermes_welcome_art(size: int = 136) -> ft.Control:
    """Hermes mascot framed by official and secondary Mobile identity assets."""
    width = round(size * 1.75)
    return ft.Stack(
        [
            ft.Image(
                src="hermes-welcome-bg.webp",
                width=width,
                height=size,
                fit=ft.BoxFit.COVER,
                filter_quality=ft.FilterQuality.HIGH,
                opacity=0.18,
                semantics_label="Hermes Mobile abstract orbital background",
            ),
            ft.Image(
                src="hermes-mobile-sigil.svg",
                width=width,
                height=round(size * 0.72),
                fit=ft.BoxFit.CONTAIN,
                filter_quality=ft.FilterQuality.HIGH,
                opacity=0.78,
                semantics_label="Hermes Mobile messenger halo",
            ),
            hermes_mascot(round(size * 0.84)),
        ],
        width=width,
        height=size,
        alignment=ft.Alignment.CENTER,
    )


def status_dot(color: str, size: int = 7, tooltip: str = "") -> ft.Control:
    """Small operational presence indicator."""
    return ft.Container(
        width=size,
        height=size,
        border_radius=ft.BorderRadius.all(size),
        bgcolor=color,
        tooltip=tooltip or None,
    )


def hairline(dark: bool, vertical: bool = False) -> ft.Container:
    """A one-pixel structural divider."""
    c = mode_colors(dark)
    if vertical:
        return ft.Container(width=1, bgcolor=c["border"])
    return ft.Container(height=1, bgcolor=c["border"])


def section_label(dark: bool, text: str, trailing: str = "") -> ft.Control:
    """Compact mono section label used for data/navigation groupings."""
    c = mode_colors(dark)
    controls: List[ft.Control] = [
        ft.Text(
            text.upper(),
            size=10,
            weight=ft.FontWeight.W_700,
            color=c["muted_foreground"],
            font_family=MONO_FONT,
        )
    ]
    if trailing:
        controls.extend(
            [
                ft.Container(expand=True),
                ft.Text(
                    trailing,
                    size=10,
                    color=c["muted_foreground"],
                    font_family=MONO_FONT,
                ),
            ]
        )
    return ft.Row(controls, spacing=8)


def section_header(dark: bool, title: str, subtitle: str = "") -> ft.Control:
    """Flat section heading; whitespace groups content, not cards."""
    c = mode_colors(dark)
    controls: List[ft.Control] = [
        ft.Text(
            title,
            size=17,
            weight=ft.FontWeight.W_700,
            color=c["foreground"],
        )
    ]
    if subtitle:
        controls.append(
            ft.Text(
                subtitle,
                size=12,
                color=c["muted_foreground"],
            )
        )
    return ft.Column(controls, spacing=3)


def page_header(
    dark: bool,
    title: str,
    subtitle: str = "",
    action: Optional[ft.Control] = None,
) -> ft.Control:
    """Canonical page heading for durable destinations and operational views."""
    c = mode_colors(dark)
    text_controls: List[ft.Control] = [
        ft.Text(
            title,
            size=20,
            weight=ft.FontWeight.W_700,
            color=c["foreground"],
            font_family=HERMES_DISPLAY_FONT,
        )
    ]
    if subtitle:
        text_controls.append(ft.Text(subtitle, size=12, color=c["muted_foreground"]))
    row: List[ft.Control] = [ft.Column(text_controls, spacing=2, expand=True)]
    if action is not None:
        row.append(action)
    return ft.Container(
        content=ft.Row(row, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.Padding.only(left=16, right=10, top=14, bottom=12),
        border=ft.Border.only(bottom=ft.BorderSide(1, c["border"])),
    )


def empty_state(
    dark: bool,
    title: str,
    description: str,
    icon: Any = ft.Icons.INBOX_OUTLINED,
    action: Optional[ft.Control] = None,
    branded: bool = False,
) -> ft.Control:
    """One shared empty-state language for all views."""
    c = mode_colors(dark)
    lead = brand_mark(52) if branded else ft.Icon(icon, size=32, color=c["muted_foreground"])
    controls: List[ft.Control] = [
        lead,
        ft.Container(height=10),
        ft.Text(
            title,
            size=19,
            weight=ft.FontWeight.W_700,
            color=c["foreground"],
            text_align=ft.TextAlign.CENTER,
            font_family=HERMES_DISPLAY_FONT,
        ),
        ft.Text(
            description,
            size=13,
            color=c["muted_foreground"],
            text_align=ft.TextAlign.CENTER,
        ),
    ]
    if action is not None:
        controls.extend([ft.Container(height=8), action])
    return ft.Container(
        content=ft.Column(
            controls,
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        alignment=ft.Alignment.CENTER,
        expand=True,
        padding=ft.Padding.symmetric(horizontal=32, vertical=24),
    )


def flat_list_row(
    dark: bool,
    title: str,
    subtitle: str = "",
    leading: Optional[ft.Control] = None,
    trailing: Optional[ft.Control] = None,
    on_click=None,
) -> ft.Control:
    """Flat list row with one optional trailing action; never a nested card."""
    c = mode_colors(dark)
    text: List[ft.Control] = [
        ft.Text(title, size=14, weight=ft.FontWeight.W_500, color=c["foreground"])
    ]
    if subtitle:
        text.append(
            ft.Text(
                subtitle,
                size=11,
                color=c["muted_foreground"],
                max_lines=2,
                overflow=ft.TextOverflow.ELLIPSIS,
            )
        )
    row: List[ft.Control] = []
    if leading is not None:
        row.extend([leading, ft.Container(width=4)])
    row.append(ft.Column(text, spacing=2, expand=True))
    if trailing is not None:
        row.append(trailing)
    return ft.Container(
        content=ft.Row(row, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
        padding=ft.Padding.symmetric(horizontal=14, vertical=10),
        on_click=on_click,
        ink=on_click is not None,
        border_radius=ft.BorderRadius.all(6),
    )


def page_scaffold(controls: Iterable[ft.Control], dark: bool, padding: int = 16) -> ft.Control:
    """Standard scrollable page body with flat styling."""
    return ft.ListView(
        controls=list(controls),
        padding=ft.Padding.all(padding),
        spacing=18,
    )


def flat_button(
    text: str,
    icon: Any,
    on_click,
    dark: bool,
    destructive: bool = False,
    primary: bool = False,
) -> ft.Control:
    """Token-driven button with no Material elevation."""
    c = mode_colors(dark)
    if primary:
        foreground = c["primary_foreground"]
        background = c["primary"]
        side = ft.BorderSide(1, c["primary"])
    else:
        foreground = c["destructive"] if destructive else c["foreground"]
        background = None
        side = ft.BorderSide(1, c["destructive"] if destructive else c["border"])
    return ft.Button(
        content=text,
        icon=icon,
        on_click=on_click,
        elevation=0,
        style=ft.ButtonStyle(
            color=foreground,
            bgcolor=background,
            shape=ft.RoundedRectangleBorder(radius=7),
            side=side,
        ),
    )
