#!/usr/bin/env python
"""Shadow Hunter launcher.

    python run.py api            FastAPI backend only (docs at /docs)
    python run.py deck           PySide6 instrument deck   [flagship]
    python run.py console        CustomTkinter field console
    python run.py control        Flet mission control
    python run.py web            NiceGUI browser observatory
    python run.py scope          DearPyGui telemetry scope
    python run.py all            backend + deck (default)

    python run.py train-rl  --steps 20000 --algo PPO
    python run.py train-cnn --scenes 24 --epochs 20
    python run.py synth     --count 8
    python run.py doctor         check which parts of the stack are installed
"""
from __future__ import annotations

import argparse
import sys

# Pin the numeric runtime before torch/pandas can disagree about it - see the
# import-order guard documented in shadowhunter/__init__.py.
import shadowhunter  # noqa: F401,E402

BANNER = r"""
   ____ _  _   _   ___   ___  _    _   _   _ _   _ _  _ _____ ___ ___
  / ___| || | /_\ |   \ / _ \| |  | | | | | | \ | | || |_   _| __| _ \
  \___ \ __ |/ _ \| |) | (_) | |/\| | | |_| |  \| | __ | | | | _||   /
  |____/_||_/_/ \_\___/ \___/|_/\_\_|  \___/|_|\__|_||_| |_| |___|_|_\
        deep RL zone selection  ·  CNN shadow metrology  ·  100% OSS
"""


def cmd_api(args: argparse.Namespace) -> int:
    import uvicorn

    from shadowhunter.core.config import SETTINGS

    uvicorn.run("shadowhunter.views.api.app:app",
                host=args.host or SETTINGS.server.host,
                port=args.port or SETTINGS.server.port,
                reload=args.reload, log_level="info")
    return 0


def cmd_deck(args: argparse.Namespace) -> int:
    from shadowhunter.views.desktop.pyside.app import main

    return main(base_url=args.url, embedded=not args.url)


def cmd_console(args: argparse.Namespace) -> int:
    from shadowhunter.views.desktop.ctk.app import main

    return main(base_url=args.url)


def cmd_control(args: argparse.Namespace) -> int:
    from shadowhunter.views.desktop.flet_app.app import main

    return main(base_url=args.url, web=args.web)


def cmd_web(args: argparse.Namespace) -> int:
    from shadowhunter.views.web.observatory import main

    return main(base_url=args.url, port=args.port or 8080)


def cmd_scope(args: argparse.Namespace) -> int:
    from shadowhunter.views.desktop.dpg.app import main

    return main(base_url=args.url)


def cmd_all(args: argparse.Namespace) -> int:
    return cmd_deck(args)


def cmd_train_rl(args: argparse.Namespace) -> int:
    from shadowhunter.core.config import SETTINGS, RLConfig
    from shadowhunter.models.rl.agent import train_agent

    cfg = RLConfig(algo=args.algo, total_timesteps=args.steps)
    out = SETTINGS.checkpoint_dir / f"shadow_hunter_{args.algo.lower()}.zip"
    result = train_agent(cfg, save_path=out)
    print(f"\npolicy -> {result['path']}")
    return 0


def cmd_train_cnn(args: argparse.Namespace) -> int:
    import numpy as np

    from shadowhunter.core.config import SETTINGS
    from shadowhunter.models.vision.height_cnn import save_model, train_regressor
    from shadowhunter.models.vision.preprocess import crop_dataset, synthesize_scene
    from shadowhunter.models.vision.shadow_ops import analyse_crop, shadow_mask

    images, heights, elevs, gsds, lens = [], [], [], [], []
    for i in range(args.scenes):
        scene = synthesize_scene(size=512, n_buildings=args.buildings, seed=1000 + i)
        X, y = crop_dataset(scene, crop=96, seed=i)
        for patch, h in zip(X, y):
            bgr = patch[:, :, :3]
            m = analyse_crop(bgr, scene.sun, shadow_mask(bgr))
            images.append(patch); heights.append(h)
            elevs.append(scene.sun.elevation_deg); gsds.append(scene.sun.gsd_m)
            lens.append(m.shadow_len_px)
        print(f"  scene {i + 1}/{args.scenes}  ·  {len(images)} crops", end="\r")
    print()
    if not images:
        print("no crops produced - increase --scenes")
        return 1

    model, meta = train_regressor(
        np.stack(images), np.asarray(heights, np.float32), np.asarray(elevs, np.float32),
        np.asarray(gsds, np.float32), np.asarray(lens, np.float32),
        epochs=args.epochs, batch_size=args.batch,
    )
    out = SETTINGS.checkpoint_dir / "height_cnn.pt"
    save_model(model, out, meta)
    print(f"\nregressor -> {out}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from shadowhunter.core.config import SETTINGS
    from shadowhunter.models.vision.preprocess import save_scene, synthesize_scene

    for i in range(args.count):
        scene = synthesize_scene(size=args.size, n_buildings=args.buildings,
                                 seed=args.seed + i, name=f"tile_{i:03d}")
        path = save_scene(scene, SETTINGS.data_dir / "synthetic")
        print(f"  {path.name}  ·  {len(scene.buildings)} buildings  ·  "
              f"sun {scene.sun.elevation_deg:.0f}° @ {scene.sun.azimuth_deg:.0f}°")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(BANNER)
    checks = [
        ("torch", "PyTorch"), ("stable_baselines3", "Stable-Baselines3"),
        ("gymnasium", "Gymnasium"), ("cv2", "OpenCV"), ("rasterio", "Rasterio"),
        ("fastapi", "FastAPI"), ("uvicorn", "Uvicorn"),
        ("customtkinter", "CustomTkinter"), ("PySide6", "PySide6"),
        ("nicegui", "NiceGUI"), ("flet", "Flet"), ("dearpygui", "DearPyGui"),
    ]
    missing = []
    for module, label in checks:
        try:
            __import__(module)
            print(f"  [ok]      {label}")
        except Exception as exc:
            missing.append(label)
            print(f"  [MISSING] {label}  ({type(exc).__name__})")

    try:
        import flet_web  # noqa: F401

        print("  [ok]      flet-web (browser mode for `run.py control --web`)")
    except Exception:
        print("  [opt]     flet-web not installed - `run.py control` still works "
              "as a native window; `pip install flet-web` adds --web")

    try:
        import torch

        print(f"\n  device: {'CUDA · ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except Exception:
        pass
    if missing:
        print(f"\n  install with:  pip install -r requirements.txt")
        return 1
    print("\n  stack complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="Shadow Hunter launcher",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--url", help="talk to a remote deck instead of embedding one")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--web", action="store_true", help="Flet: serve in a browser")
    sub = p.add_subparsers(dest="command")

    for name, fn in (("api", cmd_api), ("deck", cmd_deck), ("console", cmd_console),
                     ("control", cmd_control), ("web", cmd_web), ("scope", cmd_scope),
                     ("all", cmd_all), ("doctor", cmd_doctor)):
        sub.add_parser(name).set_defaults(func=fn)

    rl = sub.add_parser("train-rl")
    rl.add_argument("--steps", type=int, default=20_000)
    rl.add_argument("--algo", default="PPO", choices=["PPO", "DQN"])
    rl.set_defaults(func=cmd_train_rl)

    cnn = sub.add_parser("train-cnn")
    cnn.add_argument("--scenes", type=int, default=24)
    cnn.add_argument("--buildings", type=int, default=14)
    cnn.add_argument("--epochs", type=int, default=20)
    cnn.add_argument("--batch", type=int, default=32)
    cnn.set_defaults(func=cmd_train_cnn)

    syn = sub.add_parser("synth")
    syn.add_argument("--count", type=int, default=8)
    syn.add_argument("--size", type=int, default=512)
    syn.add_argument("--buildings", type=int, default=16)
    syn.add_argument("--seed", type=int, default=7)
    syn.set_defaults(func=cmd_synth)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        print(BANNER)
        parser.print_help()
        return 0
    if args.command not in {"doctor"}:
        print(BANNER)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
