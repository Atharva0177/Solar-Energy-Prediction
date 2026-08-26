---
name: env-conda-run-newline
description: On this machine `conda run -n solar python -c "<multiline>"` fails — write a temp script file instead
metadata:
  type: reference
---

`conda run -n solar python -c "..."` raises
`NotImplementedError: Support for scripts where arguments contain newlines not implemented`
whenever the `-c` argument contains newlines (conda 25.11.1 on this Windows box).

**Workaround:** Write the snippet to a `.py` file (use the session job tmp dir,
e.g. `C:\Users\manda\.claude\jobs\<job>\tmp`) and run
`conda run -n solar python <file.py>`. Single-line `-c` is fine.

Also: env is conda `solar` (`conda run -n solar ...`), Python 3.13, torch 2.13.0+cu132
installed manually by user (RTX 5070, sm_120). Related: [[project-state]].
