#!/usr/bin/env python3
"""
Turing Tumble Board Editor

Click-to-place GUI for creating Turing Tumble challenge boards.
Renders boards using the same matplotlib code as the reference images
(board_renderer.py), so what you see is exactly what you get.

Usage:
    python scripts/ui.py                     # New board
    python scripts/ui.py path/to/board.json  # Open existing board
"""

import json
import math
import sys
import os
from pathlib import Path
from typing import Optional
from copy import deepcopy

# -- matplotlib (must be set before any matplotlib import) --------------------
import matplotlib
matplotlib.use("QtAgg")

# -- project root on path so simulator imports work ---------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "simulator"))

# -- board renderer imports ---------------------------------------------------
from board_renderer import (
    COLOURS, BOARD_W, BOARD_H, CELL,
    MARGIN_SIDES, MARGIN_TOP, MARGIN_BOTTOM, FIG_W, FIG_H,
    _ax_coord, draw_peg_grid, draw_board_frame, draw_component,
    draw_hopper, draw_catcher,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# -- PyQt6 --------------------------------------------------------------------
from PyQt6.QtCore import (
    Qt, QRectF, QPointF, QSize, pyqtSignal,
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QIcon, QAction, QPixmap, QCursor,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QSplitter, QLabel,
    QDockWidget, QMenu, QFileDialog, QMessageBox,
    QSpinBox, QComboBox, QGroupBox, QFormLayout,
    QToolBar, QStatusBar, QLineEdit, QTextEdit,
    QDialog, QDialogButtonBox, QPushButton, QMenuBar,
    QSizePolicy, QFrame,
)

# ============================================================================
# Constants
# ============================================================================

# Colours pulled from board_renderer for the palette — per-component accent
PALETTE_COLOURS = {
    "ramp_right":   "#4FC3F7",
    "ramp_left":    "#81C784",
    "crossover":    "#CE93D8",
    "bit":          "#FFB74D",
    "gear_bit":     "#E57373",
    "gear":         "#90A4AE",
    "interceptor":  "#FFF176",
    "trigger":      "#4DD0E1",
}

COMPONENT_DEFS = {
    "ramp_right":   {"label": "Ramp →",   "has_state": False},
    "ramp_left":    {"label": "Ramp ←",   "has_state": False},
    "crossover":    {"label": "Crossover", "has_state": False},
    "bit":          {"label": "Bit",      "has_state": True,
                     "state": 0, "state_labels": ["← (0)", "→ (1)"]},
    "gear_bit":     {"label": "Gear Bit", "has_state": True,
                     "state": 0, "state_labels": ["← (0)", "→ (1)"],
                     "gear_group": -1},
    "gear":         {"label": "Gear",     "has_state": False},
    "interceptor":  {"label": "Intercept.", "has_state": True,
                     "state": "left", "state_labels": ["left", "right"]},
    "trigger":      {"label": "Trigger",  "has_state": True,
                     "state": "blue", "state_labels": ["blue", "red"]},
}

DEFAULT_BOARD_META = {
    "task_id": "tt-custom-001",
    "challenge_number": 0,
    "title": "Untitled Board",
    "objective": "",
    "hopper_entry_mode": "column",
    "blue_hopper":  {"x": 2, "y": -1, "count": 8},
    "red_hopper":   {"x": 8, "y": -1, "count": 8},
    "left_trigger":  {"x": 2, "y": 11},
    "right_trigger": {"x": 8, "y": 11},
    "available_parts": {k: 0 for k in COMPONENT_DEFS},
    "input_sequence": ["blue"] * 8,
    "expected_output_desc": "Where each marble ends up after execution",
    "expected_output_format": "left_catcher, right_catcher counts",
}


# ============================================================================
# Helpers
# ============================================================================

def ax_to_grid(ax_x: float, ax_y: float) -> Optional[tuple[int, int]]:
    """Convert matplotlib axes coordinates to (col, row) grid coords.

    Inverse of board_renderer._ax_coord.  Returns None if outside the
    valid 0..BOARD_W-1 × 0..BOARD_H-1 range.
    """
    col = round(ax_x - MARGIN_SIDES)
    row = BOARD_H - 1 - round(ax_y - MARGIN_BOTTOM)
    if 0 <= col < BOARD_W and 0 <= row < BOARD_H:
        return col, row
    return None


# ============================================================================
# BoardCanvas — click-to-place matplotlib board
# ============================================================================

class BoardCanvas(FigureCanvas):
    """Matplotlib-powered board view.  Click palette → click cell to place.

    Signals
    -------
    component_selected(dict | None)
        Emitted when the user selects / deselects a component on the board.
    board_changed()
        Emitted after any placement / move / removal.
    """

    component_selected = pyqtSignal(object)   # dict or None
    board_changed = pyqtSignal()               # fired after any change

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(7.0, 7.0 * FIG_H / FIG_W))
        self._ax  = self._fig.add_subplot(111)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(500, 500)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # -- state -----------------------------------------------------------
        self._components: list[dict] = []      # [{type, x, y, state?, gear_group?, side?}]
        self._selected_idx: Optional[int] = None
        self._placing_type: Optional[str] = None   # set by palette, cleared after placement
        self._board_meta = deepcopy(DEFAULT_BOARD_META)

        # undo / redo
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._undoing = False
        self.MAX_UNDO = 50

        # -- mouse events ----------------------------------------------------
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("key_press_event", self._on_key)

        # -- initial draw ----------------------------------------------------
        self._redraw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def board_meta(self) -> dict:
        return self._board_meta

    @board_meta.setter
    def board_meta(self, value: dict):
        self._board_meta = value
        self._redraw()

    def place_component(self, comp_type: str, col: int, row: int,
                        extra: dict | None = None) -> Optional[dict]:
        """Place a component at (col, row).  Returns None if occupied."""
        if self._component_at(col, row) is not None:
            return None
        self._push_undo()
        comp = {"type": comp_type, "x": col, "y": row}
        if extra:
            comp.update(extra)
        else:
            defn = COMPONENT_DEFS[comp_type]
            if defn.get("has_state"):
                comp["state"] = defn["state"]
            if comp_type == "gear_bit":
                comp["gear_group"] = -1
        self._components.append(comp)
        self._selected_idx = len(self._components) - 1
        self.board_changed.emit()
        self.component_selected.emit(comp)
        self._redraw()
        return comp

    def remove_component_at(self, col: int, row: int) -> bool:
        """Remove the component at (col, row).  Returns True if one was found."""
        idx = self._component_at(col, row)
        if idx is None:
            return False
        self._push_undo()
        removed = self._components.pop(idx)
        if self._selected_idx == idx or self._selected_idx is None:
            self._selected_idx = None
        elif self._selected_idx > idx:
            self._selected_idx -= 1
        self.board_changed.emit()
        self.component_selected.emit(None)
        self._redraw()
        return True

    def remove_selected(self) -> bool:
        """Remove the currently selected component."""
        if self._selected_idx is None:
            return False
        self._push_undo()
        self._components.pop(self._selected_idx)
        self._selected_idx = None
        self.board_changed.emit()
        self.component_selected.emit(None)
        self._redraw()
        return True

    def get_selected(self) -> Optional[dict]:
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._components):
            return self._components[self._selected_idx]
        return None

    def all_components(self) -> list[dict]:
        return self._components

    def set_placing_type(self, comp_type: Optional[str]):
        """Enter / exit component placement mode."""
        self._placing_type = comp_type
        if comp_type:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self._redraw()

    def clear_all(self):
        if not self._components:
            return
        self._push_undo()
        self._components.clear()
        self._selected_idx = None
        self.board_changed.emit()
        self.component_selected.emit(None)
        self._redraw()

    def update_component_state(self, idx: int, key: str, value):
        if 0 <= idx < len(self._components):
            self._components[idx][key] = value
            self.board_changed.emit()
            self._redraw()

    def toggle_selected_state(self):
        """Double-click equivalent — cycle state of selected component."""
        comp = self.get_selected()
        if comp is None:
            return
        defn = COMPONENT_DEFS.get(comp["type"], {})
        if not defn.get("has_state"):
            return
        t = comp["type"]
        if t in ("bit", "gear_bit"):
            comp["state"] = 1 if comp.get("state", 0) == 0 else 0
        elif t == "interceptor":
            comp["side"] = "right" if comp.get("side", "left") == "left" else "left"
        elif t == "trigger":
            comp["side"] = "red" if comp.get("side", "blue") == "blue" else "blue"
        self.board_changed.emit()
        self.component_selected.emit(comp)
        self._redraw()

    # -- serialisation ---------------------------------------------------

    def to_json(self):
        """Build the full challenge JSON dict from current state."""
        meta = self._board_meta
        fixed = sorted(self._components, key=lambda d: (d["y"], d["x"]))
        # Normalise keys for JSON export
        export = []
        for c in fixed:
            d = {"type": c["type"], "x": c["x"], "y": c["y"]}
            if c["type"] in ("bit", "gear_bit") and "state" in c:
                d["state"] = c["state"]
            if c["type"] == "gear_bit" and c.get("gear_group", -1) != -1:
                d["gear_group"] = c["gear_group"]
            if c["type"] == "interceptor" and c.get("side"):
                d["side"] = c["side"]
            if c["type"] == "trigger" and c.get("side"):
                d["side"] = c["side"]
            export.append(d)

        return {
            "task_id": meta["task_id"],
            "challenge_number": meta["challenge_number"],
            "title": meta["title"],
            "objective": meta["objective"],
            "board": {
                "width": BOARD_W,
                "height": BOARD_H,
                "fixed_components": export,
                "ball_hoppers": {
                    "blue": meta["blue_hopper"],
                    "red": meta["red_hopper"],
                },
                "trigger_levers": {
                    "left": meta["left_trigger"],
                    "right": meta["right_trigger"],
                },
            },
            "available_parts": meta.get("available_parts", {k: 0 for k in COMPONENT_DEFS}),
            "solution": {
                "placed_components": [],
                "explanation": "",
                "verified": False,
                "verifier_version": None,
                "position_verified": False,
                "final_marble_state": [],
            },
            "input_sequence": meta.get("input_sequence", ["blue"] * 8),
            "expected_output": {
                "description": meta.get("expected_output_desc", ""),
                "format": meta.get("expected_output_format", ""),
            },
        }

    def load_from_json(self, doc: dict):
        self.clear_all()
        board = doc.get("board", {})
        self._board_meta = {
            "task_id": doc.get("task_id", "tt-custom-001"),
            "challenge_number": doc.get("challenge_number", 0),
            "title": doc.get("title", "Untitled"),
            "objective": doc.get("objective", ""),
            "board_width": BOARD_W,
            "board_height": BOARD_H,
            "hopper_entry_mode": board.get("hopper_entry_mode", "column"),
            "blue_hopper": board.get("ball_hoppers", {}).get("blue",
                                   {"x": 2, "y": -1, "count": 8}),
            "red_hopper": board.get("ball_hoppers", {}).get("red",
                                  {"x": 8, "y": -1, "count": 8}),
            "left_trigger": board.get("trigger_levers", {}).get("left",
                                   {"x": 2, "y": 11}),
            "right_trigger": board.get("trigger_levers", {}).get("right",
                                     {"x": 8, "y": 11}),
            "available_parts": doc.get("available_parts",
                                       {k: 5 for k in COMPONENT_DEFS}),
            "input_sequence": doc.get("input_sequence", ["blue"] * 8),
            "expected_output_desc": doc.get("expected_output", {}).get("description", ""),
            "expected_output_format": doc.get("expected_output", {}).get("format", ""),
        }
        for fc in board.get("fixed_components", []):
            extra = {}
            if fc["type"] in ("bit", "gear_bit"):
                extra["state"] = fc.get("state", 0)
            if fc["type"] == "gear_bit":
                extra["gear_group"] = fc.get("gear_group", -1)
            if fc["type"] in ("interceptor", "trigger"):
                extra["side"] = fc.get("side", "left" if fc["type"] == "interceptor" else "blue")
            self.place_component(fc["type"], fc["x"], fc["y"], extra)
        self._selected_idx = None
        self.board_changed.emit()
        self._redraw()

    # -- undo / redo ------------------------------------------------------

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self._redraw()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self._redraw()
        return True

    def _clear_undo(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _push_undo(self):
        if self._undoing:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _snapshot(self):
        return {
            "components": deepcopy(self._components),
            "meta": deepcopy(self._board_meta),
        }

    def _restore(self, snap):
        self._undoing = True
        try:
            self._components = snap["components"]
            self._board_meta = snap["meta"]
            self._selected_idx = None
        finally:
            self._undoing = False

    # -- internals ------------------------------------------------------------

    def _component_at(self, col: int, row: int) -> Optional[int]:
        """Return the list index of the component at (col, row), or None."""
        for i, c in enumerate(self._components):
            if c["x"] == col and c["y"] == row:
                return i
        return None

    def _on_click(self, event):
        """Handle mouse clicks on the board."""
        if event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        grid = ax_to_grid(event.xdata, event.ydata)
        if grid is None:
            return
        col, row = grid

        if event.dblclick:
            self._on_double_click(col, row)
            return

        if self._placing_type is not None:
            # Place component
            existing = self._component_at(col, row)
            if existing is not None:
                self._flash_status(f"Cell ({col},{row}) already occupied")
            else:
                self.place_component(self._placing_type, col, row)
                self._flash_status(f"Placed {self._placing_type} at ({col},{row})")
                self.set_placing_type(None)  # exit placement mode after single placement
            return

        # Select / deselect
        idx = self._component_at(col, row)
        if idx == self._selected_idx:
            self._selected_idx = None
            self.component_selected.emit(None)
        else:
            self._selected_idx = idx
            if idx is not None:
                self.component_selected.emit(self._components[idx])
            else:
                self.component_selected.emit(None)
        self._redraw()

    def _on_double_click(self, col: int, row: int):
        """Double-click: toggle state if on a component."""
        idx = self._component_at(col, row)
        if idx is None:
            return
        self._selected_idx = idx
        self.toggle_selected_state()

    def _on_key(self, event):
        """Handle keyboard events (delete/backspace to remove)."""
        if event.key in ("delete", "backspace"):
            self.remove_selected()
            self._flash_status("Removed selected component")

    def _redraw(self):
        """Full redraw using board_renderer functions."""
        ax = self._ax
        ax.clear()
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, FIG_H)
        ax.set_aspect("equal")
        ax.axis("off")

        # Background
        ax.add_patch(plt.Rectangle((0, 0), FIG_W, FIG_H,
                                    color=COLOURS["bg"], zorder=0))

        # Grid & frame
        draw_peg_grid(ax)
        meta = self._board_meta
        title = meta.get("title", "")
        subtitle = "" if title == "Untitled Board" else "Board Editor"
        draw_board_frame(ax, title=title or "", subtitle="")

        # Hoppers
        blue_h = meta.get("blue_hopper", {})
        red_h  = meta.get("red_hopper", {})
        if blue_h.get("count", 0) > 0:
            draw_hopper(ax, blue_h["x"], "B", blue_h["count"], "blue",
                        blue_h.get("y", -1))
        if red_h.get("count", 0) > 0:
            draw_hopper(ax, red_h["x"], "R", red_h["count"], "red",
                        red_h.get("y", -1))

        # Catchers
        left_t  = meta.get("left_trigger", {})
        right_t = meta.get("right_trigger", {})
        if left_t:
            draw_catcher(ax, left_t["x"], "blue")
        if right_t:
            draw_catcher(ax, right_t["x"], "red")

        # Components
        for i, comp in enumerate(self._components):
            is_selected = (i == self._selected_idx)
            # Use the same color as board_renderer for "fixed" components
            draw_component(ax, comp, COLOURS["fixed"], zorder=5)
            # Selection highlight
            if is_selected:
                cx, cy = _ax_coord(comp["x"], comp["y"])
                sel = plt.Rectangle(
                    (cx - 0.48, cy - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="#2563EB", linewidth=2.8,
                    zorder=15, linestyle="--",
                )
                ax.add_patch(sel)

        # Placement preview — highlight cell under cursor if in placing mode
        if self._placing_type is not None:
            # Draw a subtle highlight on the board title area
            ax.text(
                FIG_W / 2, FIG_H - 1.8,
                f"Placing: {COMPONENT_DEFS[self._placing_type]['label']} — click a cell",
                ha="center", va="top", fontsize=10,
                color="#2563EB", fontweight="bold", zorder=20,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE",
                          edgecolor="#93C5FD", alpha=0.9),
            )

        self.draw()

    def _flash_status(self, msg: str):
        w = self.window()
        if w and hasattr(w, "statusBar"):
            w.statusBar().showMessage(msg, 3000)


# ============================================================================
# ComponentPalette  —  click to select, then click board to place
# ============================================================================

class ComponentPalette(QListWidget):
    """Side panel listing component types; click to enter placement mode."""

    def __init__(self, board_canvas: BoardCanvas, parent=None):
        super().__init__(parent)
        self._canvas = board_canvas
        self.setMaximumWidth(150)
        self.setMinimumWidth(130)
        self.setSpacing(2)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # header
        header = QListWidgetItem("— Components —")
        header.setFlags(Qt.ItemFlag.NoItemFlags)
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        self.addItem(header)

        for comp_type, defn in COMPONENT_DEFS.items():
            item = QListWidgetItem(f"  {defn['label']}")
            item.setData(Qt.ItemDataRole.UserRole, comp_type)
            item.setToolTip(f"Click to select, then click a cell on the board")
            pix = QPixmap(24, 24)
            pix.fill(QColor(PALETTE_COLOURS.get(comp_type, "#CCC")))
            item.setIcon(QIcon(pix))
            self.addItem(item)

        self.itemClicked.connect(self._on_item_clicked)

    def _on_item_clicked(self, item: QListWidgetItem):
        comp_type = item.data(Qt.ItemDataRole.UserRole)
        if comp_type is None:
            return
        # Toggle: if already placing this type, cancel; else start placing it
        if self._canvas._placing_type == comp_type:
            self._canvas.set_placing_type(None)
            self.clearSelection()
        else:
            self._canvas.set_placing_type(comp_type)
            self.setCurrentItem(item)

    def keyPressEvent(self, e):
        """Escape cancels placement mode."""
        if e.key() == Qt.Key.Key_Escape:
            self._canvas.set_placing_type(None)
            self.clearSelection()
        else:
            super().keyPressEvent(e)


# ============================================================================
# PropertiesPanel  —  edit selected component
# ============================================================================

class PropertiesPanel(QDockWidget):
    """Dock widget showing / editing the selected component's properties."""

    def __init__(self, canvas: BoardCanvas, parent=None):
        super().__init__("Properties", parent)
        self._canvas = canvas
        self._comp: Optional[dict] = None

        w = QWidget()
        layout = QVBoxLayout(w)

        self._lbl_type = QLabel("(none)")
        self._lbl_pos  = QLabel("")
        self._lbl_type.setWordWrap(True)
        layout.addWidget(QLabel("<b>Component</b>"))
        layout.addWidget(self._lbl_type)
        layout.addWidget(QLabel("<b>Position</b>"))
        layout.addWidget(self._lbl_pos)

        # state editor
        self._grp_state = QGroupBox("State")
        self._grp_state.setVisible(False)
        fl = QFormLayout()
        self._state_combo = QComboBox()
        self._state_combo.currentIndexChanged.connect(self._on_state_changed)
        fl.addRow("Value:", self._state_combo)
        self._gear_spin = QSpinBox()
        self._gear_spin.setRange(-1, 99)
        self._gear_spin.setValue(-1)
        self._gear_spin.valueChanged.connect(self._on_gear_changed)
        self._lbl_gear = QLabel("Gear group:")
        fl.addRow(self._lbl_gear, self._gear_spin)
        self._lbl_gear.setVisible(False)
        self._gear_spin.setVisible(False)
        self._grp_state.setLayout(fl)
        layout.addWidget(self._grp_state)

        # remove button
        btn = QPushButton("Remove")
        btn.clicked.connect(self._on_remove)
        layout.addWidget(btn)

        layout.addStretch()
        self.setWidget(w)
        self.setMaximumWidth(220)

        # connect
        canvas.component_selected.connect(self.set_component)

    def set_component(self, comp: Optional[dict]):
        self._comp = comp
        if comp is None:
            self._lbl_type.setText("(none)")
            self._lbl_pos.setText("")
            self._grp_state.setVisible(False)
            return
        defn = COMPONENT_DEFS[comp["type"]]
        self._lbl_type.setText(defn["label"])
        self._lbl_pos.setText(f"({comp['x']}, {comp['y']})")

        has_state = defn.get("has_state", False)
        self._grp_state.setVisible(has_state)
        if has_state:
            self._state_combo.blockSignals(True)
            self._state_combo.clear()
            self._state_combo.addItems(defn["state_labels"])
            idx = 0
            t = comp["type"]
            if t in ("bit", "gear_bit"):
                idx = comp.get("state", 0)
            elif t in ("interceptor", "trigger"):
                side = comp.get("side", "left" if t == "interceptor" else "blue")
                labels = defn["state_labels"]
                idx = labels.index(side) if side in labels else 0
            self._state_combo.setCurrentIndex(idx)
            self._state_combo.blockSignals(False)

            is_gearbit = t == "gear_bit"
            self._lbl_gear.setVisible(is_gearbit)
            self._gear_spin.setVisible(is_gearbit)
            if is_gearbit:
                self._gear_spin.blockSignals(True)
                self._gear_spin.setValue(comp.get("gear_group", -1))
                self._gear_spin.blockSignals(False)

    def _on_state_changed(self, idx: int):
        if self._comp is None:
            return
        t = self._comp["type"]
        defn = COMPONENT_DEFS[t]
        if t in ("bit", "gear_bit"):
            self._comp["state"] = idx
        elif t in ("interceptor", "trigger"):
            self._comp["side"] = defn["state_labels"][idx]
        self._canvas.board_changed.emit()
        self._canvas._redraw()

    def _on_gear_changed(self, val: int):
        if self._comp and self._comp.get("type") == "gear_bit":
            self._comp["gear_group"] = val
            self._canvas.board_changed.emit()
            self._canvas._redraw()

    def _on_remove(self):
        self._canvas.remove_selected()
        self._comp = None


# ============================================================================
# BoardConfigDialog  —  edit board metadata
# ============================================================================

class BoardConfigDialog(QDialog):
    """Modal dialog to configure board metadata, hoppers, triggers, etc."""

    def __init__(self, meta: dict, parent=None, canvas: Optional[BoardCanvas] = None):
        super().__init__(parent)
        self.setWindowTitle("Board Configuration")
        self.setMinimumWidth(420)
        self._canvas = canvas
        self._original_meta = deepcopy(meta)
        self._meta = deepcopy(meta)

        layout = QVBoxLayout(self)

        # -- task info -------------------------------------------------------
        grp = QGroupBox("Task Metadata")
        fl = QFormLayout()
        self._task_id = QLineEdit(meta.get("task_id", ""))
        self._ch_num = QSpinBox()
        self._ch_num.setRange(0, 9999)
        self._ch_num.setValue(meta.get("challenge_number", 0))
        self._title = QLineEdit(meta.get("title", ""))
        self._objective = QTextEdit()
        self._objective.setMaximumHeight(60)
        self._objective.setPlainText(meta.get("objective", ""))
        fl.addRow("Task ID:", self._task_id)
        fl.addRow("Challenge #:", self._ch_num)
        fl.addRow("Title:", self._title)
        fl.addRow("Objective:", self._objective)
        grp.setLayout(fl)
        layout.addWidget(grp)

        # -- board entry mode ------------------------------------------------
        grp2 = QGroupBox("Board")
        fl2 = QFormLayout()
        self._entry = QComboBox()
        self._entry.addItems(["column", "inward"])
        self._entry.setCurrentText(meta.get("hopper_entry_mode", "column"))
        fl2.addRow("Entry mode:", self._entry)
        grp2.setLayout(fl2)
        layout.addWidget(grp2)

        # -- hoppers ---------------------------------------------------------
        grp3 = QGroupBox("Ball Hoppers")
        fl3 = QFormLayout()
        bh = meta.get("blue_hopper", {"x": 2, "y": -1, "count": 8})
        rh = meta.get("red_hopper", {"x": 8, "y": -1, "count": 8})
        self._bh_x = QSpinBox(); self._bh_x.setRange(0, 10); self._bh_x.setValue(bh["x"])
        self._bh_cnt = QSpinBox(); self._bh_cnt.setRange(0, 99); self._bh_cnt.setValue(bh["count"])
        self._rh_x = QSpinBox(); self._rh_x.setRange(0, 10); self._rh_x.setValue(rh["x"])
        self._rh_cnt = QSpinBox(); self._rh_cnt.setRange(0, 99); self._rh_cnt.setValue(rh["count"])
        fl3.addRow("Blue X:", self._bh_x)
        fl3.addRow("Blue count:", self._bh_cnt)
        fl3.addRow("Red X:", self._rh_x)
        fl3.addRow("Red count:", self._rh_cnt)
        grp3.setLayout(fl3)
        layout.addWidget(grp3)

        # -- triggers --------------------------------------------------------
        grp4 = QGroupBox("Trigger Levers")
        fl4 = QFormLayout()
        lt = meta.get("left_trigger", {"x": 2, "y": 11})
        rt = meta.get("right_trigger", {"x": 8, "y": 11})
        self._lt_x = QSpinBox(); self._lt_x.setRange(0, 10); self._lt_x.setValue(lt["x"])
        self._rt_x = QSpinBox(); self._rt_x.setRange(0, 10); self._rt_x.setValue(rt["x"])
        fl4.addRow("Left X:", self._lt_x)
        fl4.addRow("Right X:", self._rt_x)
        grp4.setLayout(fl4)
        layout.addWidget(grp4)

        # -- available parts -------------------------------------------------
        grp5 = QGroupBox("Available Parts (for solver)")
        fl5 = QFormLayout()
        self._part_spins: dict[str, QSpinBox] = {}
        ap = meta.get("available_parts", {})
        for ct in COMPONENT_DEFS:
            sb = QSpinBox()
            sb.setRange(0, 99)
            sb.setValue(ap.get(ct, 5))
            self._part_spins[ct] = sb
            fl5.addRow(COMPONENT_DEFS[ct]["label"] + ":", sb)
        grp5.setLayout(fl5)
        layout.addWidget(grp5)

        # -- input sequence --------------------------------------------------
        grp6 = QGroupBox("Input Sequence")
        fl6 = QFormLayout()
        self._input_seq = QLineEdit(
            ", ".join(meta.get("input_sequence", ["blue"] * 8))
        )
        self._input_seq.setToolTip("Comma-separated: blue, blue, red, ...")
        self._exp_desc = QLineEdit(meta.get("expected_output_desc", ""))
        self._exp_fmt = QLineEdit(meta.get("expected_output_format", ""))
        fl6.addRow("Sequence:", self._input_seq)
        fl6.addRow("Output desc:", self._exp_desc)
        fl6.addRow("Output format:", self._exp_fmt)
        grp6.setLayout(fl6)
        layout.addWidget(grp6)

        # -- live preview ----------------------------------------------------
        if self._canvas is not None:
            for sb in [self._bh_x, self._bh_cnt, self._rh_x, self._rh_cnt,
                        self._lt_x, self._rt_x]:
                sb.valueChanged.connect(self._on_preview_changed)
            self._entry.currentTextChanged.connect(self._on_preview_changed)

        # -- buttons ---------------------------------------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self._on_reject)
        layout.addWidget(btns)

    def _on_preview_changed(self, *_):
        if self._canvas is None:
            return
        self._canvas.board_meta = self.get_meta()

    def _on_reject(self):
        if self._canvas is not None:
            self._canvas.board_meta = self._original_meta
        self.reject()

    def get_meta(self) -> dict:
        seq_raw = self._input_seq.text().strip()
        seq = [s.strip() for s in seq_raw.split(",") if s.strip()] if seq_raw else []
        return {
            "task_id": self._task_id.text(),
            "challenge_number": self._ch_num.value(),
            "title": self._title.text(),
            "objective": self._objective.toPlainText(),
            "hopper_entry_mode": self._entry.currentText(),
            "blue_hopper":  {"x": self._bh_x.value(), "y": -1, "count": self._bh_cnt.value()},
            "red_hopper":   {"x": self._rh_x.value(), "y": -1, "count": self._rh_cnt.value()},
            "left_trigger":  {"x": self._lt_x.value(), "y": 11},
            "right_trigger": {"x": self._rt_x.value(), "y": 11},
            "available_parts": {ct: sb.value() for ct, sb in self._part_spins.items()},
            "input_sequence": seq,
            "expected_output_desc": self._exp_desc.text(),
            "expected_output_format": self._exp_fmt.text(),
        }


# ============================================================================
# MainWindow
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self, open_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Turing Tumble Board Editor")
        self.resize(1100, 850)
        self._current_path: Optional[str] = None
        self._dirty = False

        # -- canvas ----------------------------------------------------------
        self._canvas = BoardCanvas(self)

        # -- palette ---------------------------------------------------------
        self._palette = ComponentPalette(self._canvas, self)

        # -- properties dock -------------------------------------------------
        self._props = PropertiesPanel(self._canvas, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._props)

        # -- central layout --------------------------------------------------
        central = QWidget()
        ch = QHBoxLayout(central)
        ch.setContentsMargins(4, 4, 4, 4)
        ch.setSpacing(6)
        ch.addWidget(self._palette)
        ch.addWidget(self._canvas, 1)
        self.setCentralWidget(central)

        # -- menus & toolbar -------------------------------------------------
        self._create_menus()
        self._create_toolbar()

        # -- status bar ------------------------------------------------------
        self.statusBar().showMessage(
            "Click a component in the palette, then click a cell to place it"
        )

        # -- signals ---------------------------------------------------------
        self._canvas.board_changed.connect(self._mark_dirty)

        # -- open file if given ----------------------------------------------
        if open_path:
            self._load_file(open_path)

    # -- menus ----------------------------------------------------------------

    def _create_menus(self):
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        act = file_menu.addAction("&New Board", "Ctrl+N")
        act.triggered.connect(self._new_board)
        act = file_menu.addAction("&Open...", "Ctrl+O")
        act.triggered.connect(self._open_file)
        file_menu.addSeparator()
        act = file_menu.addAction("&Save", "Ctrl+S")
        act.triggered.connect(self._save)
        act = file_menu.addAction("Save &As...", "Ctrl+Shift+S")
        act.triggered.connect(self._save_as)
        file_menu.addSeparator()
        act = file_menu.addAction("E&xit", "Ctrl+Q")
        act.triggered.connect(self.close)

        edit_menu = bar.addMenu("&Edit")
        act = edit_menu.addAction("&Board Settings...")
        act.triggered.connect(self._edit_board_config)
        act = edit_menu.addAction("&Clear Board")
        act.triggered.connect(self._clear_board)
        edit_menu.addSeparator()
        act = edit_menu.addAction("&Undo", "Ctrl+Z")
        act.triggered.connect(self._undo)
        act = edit_menu.addAction("&Redo", "Ctrl+Shift+Z")
        act.triggered.connect(self._redo)
        edit_menu.addSeparator()
        act = edit_menu.addAction("&Fit View")
        act.triggered.connect(self._fit_view)

    def _create_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        act = tb.addAction("New")
        act.triggered.connect(self._new_board)
        act = tb.addAction("Open")
        act.triggered.connect(self._open_file)
        act = tb.addAction("Save")
        act.triggered.connect(self._save)
        tb.addSeparator()
        act = tb.addAction("Clear")
        act.triggered.connect(self._clear_board)
        act = tb.addAction("Config")
        act.triggered.connect(self._edit_board_config)
        tb.addSeparator()
        act = tb.addAction("Undo")
        act.triggered.connect(self._undo)
        act = tb.addAction("Redo")
        act.triggered.connect(self._redo)
        act = tb.addAction("Fit")
        act.triggered.connect(self._fit_view)
        self.addToolBar(tb)

    # -- actions --------------------------------------------------------------

    def _new_board(self):
        if self._dirty and not self._confirm_discard():
            return
        self._canvas.clear_all()
        self._canvas.board_meta = deepcopy(DEFAULT_BOARD_META)
        self._current_path = None
        self._dirty = False
        self.setWindowTitle("Turing Tumble Board Editor")
        self._canvas._clear_undo()
        self.statusBar().showMessage("New board")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Challenge JSON",
            str(_PROJECT_ROOT / "tasks"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        if self._dirty and not self._confirm_discard():
            return
        self._load_file(path)

    def _load_file(self, path: str):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")
            return
        self._canvas.load_from_json(doc)
        self._canvas._clear_undo()
        self._current_path = path
        self._dirty = False
        self.setWindowTitle(f"TT Board Editor — {Path(path).name}")
        n = len(self._canvas.all_components())
        self.statusBar().showMessage(f"Loaded {Path(path).name} ({n} components)")

    def _save(self):
        if self._current_path:
            self._write_file(self._current_path)
        else:
            self._save_as()

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Challenge JSON",
            str(_PROJECT_ROOT / "tasks"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        self._write_file(path)

    def _write_file(self, path: str):
        doc = self._canvas.to_json()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(doc, f, indent=2)
            self._current_path = path
            self._dirty = False
            self.setWindowTitle(f"TT Board Editor — {Path(path).name}")
            self.statusBar().showMessage(
                f"Saved {len(doc['board']['fixed_components'])} components to {Path(path).name}"
            )
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not save:\n{e}")

    def _edit_board_config(self):
        dlg = BoardConfigDialog(self._canvas.board_meta, self, canvas=self._canvas)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._canvas.board_meta = dlg.get_meta()
            self._dirty = True

    def _clear_board(self):
        if self._canvas.all_components():
            reply = QMessageBox.question(
                self, "Clear Board",
                "Remove all placed components?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._canvas.clear_all()

    def _fit_view(self):
        self._canvas._fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def _undo(self):
        if self._canvas.undo():
            self.statusBar().showMessage("Undo", 2000)
        else:
            self.statusBar().showMessage("Nothing to undo", 2000)

    def _redo(self):
        if self._canvas.redo():
            self.statusBar().showMessage("Redo", 2000)
        else:
            self.statusBar().showMessage("Nothing to redo", 2000)

    def _mark_dirty(self):
        self._dirty = True
        title = self.windowTitle()
        if not title.endswith(" *"):
            self.setWindowTitle(title + " *")

    def _confirm_discard(self) -> bool:
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "Discard unsaved changes?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Discard

    def closeEvent(self, event):
        if self._dirty and not self._confirm_discard():
            event.ignore()
        else:
            event.accept()


# ============================================================================
# Entry point
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TT Board Editor")
    open_path = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(open_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
