#!/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/bin/python
"""Run Python with locked PWM deps plus MJLab packages from the project env.

The locked original-PWM environment has the desired DFlex/PWM stack but does
not include MJLab.  This wrapper keeps locked torch/PWM/TorchRL loaded first,
then exposes the project env site-packages so MJLab and its MuJoCo runtime can
be imported without modifying either conda environment.
"""

from __future__ import annotations

import os
import runpy
import site
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PWM_SRC = PROJECT_ROOT / "baselines" / "PWM" / "src"
PROJECT_SRC = PROJECT_ROOT / "src"
PROJECT_ENV_SITE = Path(
    os.environ.get(
        "MJLAB_PROJECT_SITE_PACKAGES",
        "/storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm/lib/python3.10/site-packages",
    )
)

os.environ.setdefault("PYTHONNOUSERSITE", "1")
USER_SITE_ROOT = str(Path.home() / ".local" / "lib")
for path in list(sys.path):
    if path == site.getusersitepackages() or path.startswith(USER_SITE_ROOT):
        sys.path.remove(path)

for path in (str(PROJECT_SRC), str(PWM_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import locked-stack modules before exposing project-env site-packages.
import torch  # noqa: F401,E402
import tensordict  # noqa: F401,E402
import torchrl  # noqa: F401,E402
from pwm.algorithms.pwm import PWM  # noqa: F401,E402

if str(PROJECT_ENV_SITE) not in sys.path:
    sys.path.insert(0, str(PROJECT_ENV_SITE))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: locked_mjlab_python.py [-|-c|script.py] [args...]")

    target = sys.argv[1]
    if target == "-":
        code = sys.stdin.read()
        sys.argv = ["-"] + sys.argv[2:]
        exec(compile(code, "<stdin>", "exec"), {"__name__": "__main__", "__file__": "<stdin>"})
        return
    if target == "-c":
        if len(sys.argv) < 3:
            raise SystemExit("argument expected for -c")
        code = sys.argv[2]
        sys.argv = ["-c"] + sys.argv[3:]
        exec(compile(code, "<string>", "exec"), {"__name__": "__main__", "__file__": "<string>"})
        return
    if target == "-m":
        if len(sys.argv) < 3:
            raise SystemExit("argument expected for -m")
        module = sys.argv[2]
        sys.argv = [module] + sys.argv[3:]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return

    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
