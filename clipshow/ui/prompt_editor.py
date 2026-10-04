"""Reusable list-editor widget for semantic prompts."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

DEFAULT_PROMPTS = [
    "exciting moment",
    "people laughing",
    "beautiful scenery",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "boring static shot",
    "blank wall",
    "empty room",
    "black screen",
]


class _PromptRow(QWidget):
    """Single row in the prompt list: text label + remove button."""

    remove_clicked = Signal(int)

    def __init__(self, index: int, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        label = QLabel(text)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        remove_btn = QPushButton("\u00d7")  # multiplication sign as "x"
        remove_btn.setFixedSize(20, 20)
        remove_btn.setToolTip("Remove this prompt")
        remove_btn.setStyleSheet("QPushButton { border: none; font-weight: bold; }")
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(index))
        layout.addWidget(label, stretch=1)
        layout.addWidget(remove_btn)


class PromptEditor(QWidget):
    """Editable list of semantic text prompts.

    Emits ``prompts_changed`` whenever the list is modified (add, remove,
    or reset).
    """

    prompts_changed = Signal(list)

    def __init__(
        self,
        prompts: list[str] | None = None,
        parent: QWidget | None = None,
        default_prompts: list[str] | None = None,
    ):
        super().__init__(parent)
        self._default_prompts = list(default_prompts or DEFAULT_PROMPTS)
        self._prompts: list[str] = list(
            prompts if prompts is not None else self._default_prompts
        )
        self._setup_ui()
        self._connect_signals()
        self._refresh_list()

    # ── UI ────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        # Add row
        add_row = QHBoxLayout()
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("New prompt\u2026")
        self.add_button = QPushButton("Add")
        self.add_button.setAutoDefault(False)
        add_row.addWidget(self.line_edit)
        add_row.addWidget(self.add_button)
        layout.addLayout(add_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self.remove_button = QPushButton("Remove")
        self.remove_button.setAutoDefault(False)
        self.remove_button.setEnabled(False)
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.setAutoDefault(False)
        btn_row.addWidget(self.remove_button)
        btn_row.addStretch()
        btn_row.addWidget(self.reset_button)
        layout.addLayout(btn_row)

    def _connect_signals(self) -> None:
        self.add_button.clicked.connect(self._on_add)
        self.line_edit.returnPressed.connect(self._on_add)
        self.remove_button.clicked.connect(self._on_remove)
        self.reset_button.clicked.connect(self._on_reset)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_add(self) -> None:
        text = self.line_edit.text().strip()
        if not text:
            return
        if text in self._prompts:
            return
        self._prompts.append(text)
        self.line_edit.clear()
        self._refresh_list()
        self.prompts_changed.emit(list(self._prompts))

    def _on_remove(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._prompts):
            self._remove_at(row)

    def _remove_at(self, index: int) -> None:
        """Remove the prompt at the given index."""
        if 0 <= index < len(self._prompts):
            self._prompts.pop(index)
            self._refresh_list()
            self.prompts_changed.emit(list(self._prompts))

    def _on_reset(self) -> None:
        self._prompts = list(self._default_prompts)
        self._refresh_list()
        self.prompts_changed.emit(list(self._prompts))

    def clear_all(self) -> None:
        """Remove all prompts from the list."""
        self._prompts.clear()
        self._refresh_list()
        self.prompts_changed.emit(list(self._prompts))

    def _on_selection_changed(self, row: int) -> None:
        self.remove_button.setEnabled(row >= 0)

    # ── Helpers ───────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for i, prompt in enumerate(self._prompts):
            item = QListWidgetItem()
            self.list_widget.addItem(item)
            row_widget = _PromptRow(i, prompt)
            row_widget.remove_clicked.connect(self._remove_at)
            item.setSizeHint(row_widget.sizeHint())
            self.list_widget.setItemWidget(item, row_widget)

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    @prompts.setter
    def prompts(self, value: list[str]) -> None:
        self._prompts = list(value)
        self._refresh_list()
