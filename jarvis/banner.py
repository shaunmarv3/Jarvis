"""The cool 'JARVIS' banner — holographic gradient art (pyfiglet + a static fallback)."""

from __future__ import annotations

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

_STATIC = r"""     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝"""

# Holographic gradient stops (top → middle → bottom): cyan → sky → violet.
_STOPS = [(94, 234, 252), (56, 130, 246), (139, 92, 246)]


def _art() -> str:
    try:
        from pyfiglet import figlet_format

        art = figlet_format("JARVIS", font="ansi_shadow").rstrip("\n")
        return art if art.strip() else _STATIC
    except Exception:
        return _STATIC


def _blend(t: float) -> str:
    """Sample the multi-stop gradient at 0..1, return a #rrggbb hex string."""
    if t <= 0:
        r, g, b = _STOPS[0]
    elif t >= 1:
        r, g, b = _STOPS[-1]
    else:
        seg = t * (len(_STOPS) - 1)
        i = int(seg)
        f = seg - i
        (r1, g1, b1), (r2, g2, b2) = _STOPS[i], _STOPS[i + 1]
        r, g, b = (int(r1 + (r2 - r1) * f), int(g1 + (g2 - g1) * f), int(b1 + (b2 - b1) * f))
    return f"#{r:02x}{g:02x}{b:02x}"


def _gradient_art(art: str) -> Text:
    """Color the ASCII art with a per-row vertical gradient (the hologram glow)."""
    lines = art.split("\n")
    span = max(len(lines) - 1, 1)
    out = Text(justify="center")
    for i, line in enumerate(lines):
        out.append(line, style=f"bold {_blend(i / span)}")
        if i < len(lines) - 1:
            out.append("\n")
    return out


def render_banner(console: Console, backend: str, model: str) -> None:
    from .config import subagent_ceiling

    art = _gradient_art(_art())

    # "Just A Rather Very Intelligent System" — the real Iron Man acronym, as an easter egg.
    acronym = Text.from_markup(
        "[italic #8b5cf6]Just A Rather Very Intelligent System[/]", justify="center"
    )

    dot = "[bright_green]●[/]" if backend == "ollama" else "[bright_cyan]●[/]"
    n = subagent_ceiling(backend)
    status = Text.from_markup(
        f"{dot} [bold #5eeafc]{backend}[/] [dim]·[/] [bold #5eeafc]{model}[/] "
        f"[dim]·[/] [bold #c084fc]⚡ {n} subagents[/]",
        justify="center",
    )

    body = Group(art, Text(""), acronym, Text(""), status)
    console.print(
        Panel(
            Align.center(body),
            box=ROUNDED,
            border_style="#3882f6",
            padding=(1, 4),
            title="[bold #5eeafc]✦[/] [dim]personal research agent[/] [bold #5eeafc]✦[/]",
            subtitle="[dim]type a topic, or [bold #5eeafc]/help[/][/]",
        )
    )
