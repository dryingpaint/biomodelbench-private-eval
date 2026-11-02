"""Entrypoints run inside the sandboxes (predict, validate, score)."""
from __future__ import annotations

import os
import sys
import traceback


def run(main, errors: dict | None = None) -> None:
    """Run a stage's main(), then leave without interpreter teardown.

    Native libraries' thread pools can abort while the interpreter shuts down, after the stage
    has written everything ("terminate called without an active exception", exit -6): under
    CPU contention it failed a different scorer or predict on each CI run (2026-11-02), and it
    would fail a correct model on a busy node the same way. So the exit code is decided here,
    output is flushed, and the process ends with os._exit.

    `errors` maps an exception type to (exit code, print the traceback?) for exceptions that
    escape main(); any other one prints its traceback and exits 1, as Python would."""
    code = 0
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, int):
            code = e.code
        elif e.code is not None:
            print(e.code, file=sys.stderr)
            code = 1
    except BaseException as e:
        code, show = next(((c, p) for t, (c, p) in (errors or {}).items() if isinstance(e, t)), (1, True))
        if show:
            traceback.print_exc()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)
