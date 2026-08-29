"""PySide6 entry point - the flagship desktop deck."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from ....core.config import SETTINGS
from ....core.logging import get_logger
from ....services.client import DeckClient
from ....templates import theme
from .main_window import DeckWindow

log = get_logger(__name__)
FONT_DIR = Path(__file__).resolve().parents[4] / "assets" / "fonts"


def _install_fonts() -> list[str]:
    """Register bundled TTF/OTF files if present.

    The design names Chakra Petch and IBM Plex; the token stacks fall back to
    Bahnschrift and Cascadia Mono, both shipped with Windows, so the deck
    still looks deliberate on a bare machine.
    """
    families: list[str] = []
    if not FONT_DIR.exists():
        return families
    for path in sorted(FONT_DIR.glob("*.[to]tf")):
        fid = QFontDatabase.addApplicationFont(str(path))
        if fid != -1:
            families.extend(QFontDatabase.applicationFontFamilies(fid))
    if families:
        log.info("fonts registered: %s", ", ".join(sorted(set(families))))
    return families


def _dark_palette(app: QApplication) -> None:
    """Native dialogs and scrollbars follow the palette, not the stylesheet."""
    p = QPalette()
    ink, void, panel = theme.c("ink"), theme.c("void"), theme.c("panel")
    p.setColor(QPalette.Window, QColor(void))
    p.setColor(QPalette.WindowText, QColor(ink))
    p.setColor(QPalette.Base, QColor(panel))
    p.setColor(QPalette.AlternateBase, QColor(theme.c("elevated")))
    p.setColor(QPalette.Text, QColor(ink))
    p.setColor(QPalette.Button, QColor(theme.c("raised")))
    p.setColor(QPalette.ButtonText, QColor(ink))
    p.setColor(QPalette.Highlight, QColor(theme.c("solarWash")))
    p.setColor(QPalette.HighlightedText, QColor(theme.c("solar")))
    p.setColor(QPalette.ToolTipBase, QColor(theme.c("elevated")))
    p.setColor(QPalette.ToolTipText, QColor(ink))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(theme.c("inkFaint")))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(theme.c("inkFaint")))
    app.setPalette(p)


def main(base_url: str | None = None, embedded: bool = True) -> int:
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Shadow Hunter")
    app.setOrganizationName("Shadow Hunter Lab")
    app.setStyle("Fusion")

    _install_fonts()
    _dark_palette(app)
    app.setStyleSheet(theme.qss())

    if base_url:
        client = DeckClient(base_url=base_url)
    elif embedded:
        from ....services.supervisor import serve_in_thread

        client = serve_in_thread()
    else:
        client = DeckClient(base_url=SETTINGS.api_base)

    window = DeckWindow(client)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
