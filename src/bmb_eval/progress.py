"""What the provider sees while setup runs: a header per step saying what is
happening and how long it usually takes, how long it actually took, and a
progress bar for long transfers. Bars redraw in place on a terminal and print
every 10% otherwise (logs, CI)."""
from __future__ import annotations

import sys
import time
from pathlib import Path


def duration(seconds: float) -> str:
    s = max(0, round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def step(text: str, usually: str | None = None) -> float:
    """Print a step header; returns its start time for done()."""
    bold, plain = ("\033[1m", "\033[0m") if sys.stdout.isatty() else ("", "")
    print(f"\n{bold}==> {text}{plain}" + (f"  (usually {usually})" if usually else ""), flush=True)
    return time.monotonic()


def done(t0: float, text: str) -> None:
    print(f"✓ {text}  (took {duration(time.monotonic() - t0)})", flush=True)


def size_of(path: Path) -> int:
    """Bytes in a file, or in everything under a folder."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human_bytes(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1000
    return ""


def transfer_estimate(n_bytes: int, mb_per_s: float = 20) -> str:
    """'about 3 minutes' at a typical office connection (~20 MB/s by default)."""
    s = n_bytes / (mb_per_s * 1e6)
    if s < 60:
        return "under a minute"
    return f"about {round(s / 60)} minute{'s' if round(s / 60) != 1 else ''}"


class Bar:
    """bar = Bar("Downloading", total_bytes); bar.update(n) per chunk; bar.close()."""

    def __init__(self, label: str, total: int | None, out=None):
        self.label, self.total, self.done = label, total or None, 0
        self.out = out or sys.stdout
        self.tty = self.out.isatty()
        self.t0 = time.monotonic()
        self._last_decile = -1
        self._last_draw = 0.0

    def update(self, n: int) -> None:
        self.done += n
        now = time.monotonic()
        if self.tty:
            if now - self._last_draw >= 0.2:
                self._last_draw = now
                self.out.write("\r" + self._line(now) + "\033[K")
                self.out.flush()
        elif self.total:
            decile = int(10 * self.done / self.total)
            if decile != self._last_decile:
                self._last_decile = decile
                print(self._line(now), file=self.out, flush=True)

    def _line(self, now: float) -> str:
        elapsed = max(now - self.t0, 1e-6)
        rate = self.done / elapsed
        if not self.total:
            return f"  {self.label}: {human_bytes(self.done)} ({human_bytes(rate)}/s)"
        frac = min(1.0, self.done / self.total)
        left = (self.total - self.done) / rate if rate > 0 else 0
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        return (f"  {self.label} [{bar}] {frac:4.0%}  {human_bytes(self.done)} of "
                f"{human_bytes(self.total)}, {duration(left)} left")

    def close(self) -> None:
        if self.tty:
            self.out.write("\r" + self._line(time.monotonic()) + "\033[K\n")
            self.out.flush()


def counter(label: str, i: int, n: int) -> None:
    """'  Security checks: 3 of 12' redrawn in place (a line per check off a terminal)."""
    if sys.stdout.isatty():
        sys.stdout.write(f"\r  {label}: {i} of {n}\033[K" + ("\n" if i == n else ""))
        sys.stdout.flush()
    elif i == n or i % max(1, n // 4) == 0:
        print(f"  {label}: {i} of {n}", flush=True)
