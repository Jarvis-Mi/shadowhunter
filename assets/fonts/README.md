# Fonts

The design system names **Chakra Petch** (display) and **IBM Plex Sans / IBM Plex Mono**
(body and telemetry). Both are SIL Open Font License — free for any use, including
commercial, with no attribution requirement in the UI.

The desktop views fall back to **Bahnschrift**, **Segoe UI** and **Cascadia Mono**,
which ship with Windows 11, so the apps look deliberate on a bare machine. The
NiceGUI observatory pulls the real faces from Google Fonts at runtime.

To bundle them for offline desktop use, drop the `.ttf` files here:

```bash
python - <<'PY'
import io, urllib.request, zipfile, pathlib
out = pathlib.Path("assets/fonts"); out.mkdir(parents=True, exist_ok=True)
for name in ("Chakra_Petch", "IBM_Plex_Sans", "IBM_Plex_Mono"):
    url = f"https://fonts.google.com/download?family={name.replace('_', '%20')}"
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for member in z.namelist():
        if member.lower().endswith(".ttf") and "static" not in member.lower():
            (out / pathlib.Path(member).name).write_bytes(z.read(member))
            print("+", pathlib.Path(member).name)
PY
```

`views/desktop/pyside/app.py` and `views/desktop/dpg/app.py` register every
`.ttf`/`.otf` found in this directory at startup. Files here are git-ignored.
