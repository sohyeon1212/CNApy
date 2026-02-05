"""Filterable ComboBox widget for CNApy.

Provides a QComboBox with real-time text filtering capability.
Users can type text to filter items containing that text,
rather than just jumping to items starting with the typed text.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QCompleter


class FilterableComboBox(QComboBox):
    """QComboBox with real-time text filtering.

    This widget extends QComboBox to provide substring filtering:
    - User types text in the editable combo box
    - Items containing the typed text are shown in dropdown
    - Case-insensitive matching

    Example:
        combo = FilterableComboBox()
        combo.addItem("EX_glc__D_e - D-Glucose exchange", "EX_glc__D_e")
        combo.addItem("EX_lac__D_e - D-Lactate exchange", "EX_lac__D_e")
        # Typing "lac" will show "EX_lac__D_e - D-Lactate exchange"
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)

        # Setup completer for contains-matching
        self._completer = QCompleter(self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self.setCompleter(self._completer)

        # Update completer model when items change
        self.model().rowsInserted.connect(self._update_completer)
        self.model().rowsRemoved.connect(self._update_completer)
        self.model().modelReset.connect(self._update_completer)

    def _update_completer(self):
        """Update completer model to match combo box model."""
        self._completer.setModel(self.model())

    def showPopup(self):
        """Show popup and ensure completer model is synchronized."""
        self._completer.setModel(self.model())
        super().showPopup()
