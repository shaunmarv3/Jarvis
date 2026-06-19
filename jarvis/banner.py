"""The cool 'JARVIS' banner (pyfiglet, with a static fallback)."""

from __future__ import annotations

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_STATIC = r"""     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝"""


def _art() -> str:
    try:
        from pyfiglet import figlet_format

        art = figlet_format("JARVIS", font="ansi_shadow").rstrip("\n")
        return art if art.strip() else _STATIC
    except Exception:
        return _STATIC


def render_banner(console: Console, backend: str, model: str) -> None:
    art = Text(_art(), style="bold cyan")
    subtitle = Text.from_markup(
        "[dim]personal research agent · "
        f"backend [bold green]{backend}[/] · model [bold green]{model}[/][/]"
    )
    body = Text("\n").join([art, Text(""), subtitle])
    console.print(
        Panel(
            Align.center(body),
            border_style="cyan",
            padding=(1, 4),
            subtitle="[dim]type a topic, or /help[/]",
        )
    )
