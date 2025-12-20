from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from src.gui import modern_ui  # ensure modern patches are applied
from src.gui import native_ui

app = QApplication([])
bar = native_ui.WindowControlBar()
# Ensure buttons exist
buttons = [bar.btn_minimize, bar.btn_maximize, bar.btn_close]

out_dir = Path.cwd() / '.tmp_icon_debug'
out_dir.mkdir(exist_ok=True)

for idx, btn in enumerate(buttons, start=1):
    icon = btn.icon()
    sizes = icon.availableSizes()
    if sizes:
        size = sizes[0]
        pix = icon.pixmap(size)
    else:
        # render with default 24x24
        pix = icon.pixmap(24,24)
    fname = out_dir / f'icon_{idx}.png'
    pix.save(str(fname))
    print('saved', fname)

app.quit()
