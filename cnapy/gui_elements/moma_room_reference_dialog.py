from qtpy.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QVBoxLayout,
)

from cnapy.appdata import AppData

_MAX_SCENARIO_ROWS = 8  # show at most this many rows before scrolling kicks in


class MomaRoomReferenceDialog(QDialog):
    """Dialog to select reference fluxes for MOMA/ROOM.

    Shows:
    - Radio buttons to choose reference (previous solution or omics condition)
    - Read-only summary of the current scenario constraints that will be
      applied to the model automatically (KO / bounds).
    """

    def __init__(self, appdata: AppData, method_name: str, parent=None):
        QDialog.__init__(self, parent=parent)
        self.setWindowTitle(f"Select Reference for {method_name}")

        layout = QVBoxLayout()
        layout.setSpacing(8)

        # ── Reference selection ────────────────────────────────────────────
        ref_group = QGroupBox("Reference flux")
        ref_layout = QVBoxLayout()
        ref_layout.setSpacing(4)

        self._button_group = QButtonGroup(self)

        solution = appdata.project.solution
        solution_btn = QRadioButton("Previous Solution (FBA/MOMA/ROOM)")
        solution_btn.setEnabled(solution is not None)
        if solution is None:
            solution_btn.setToolTip("No previous solution available.")
        self._button_group.addButton(solution_btn, 0)
        ref_layout.addWidget(solution_btn)

        conditions = appdata.project.omics_conditions
        for i, condition in enumerate(conditions):
            btn = QRadioButton(f"Omics: {condition}")
            self._button_group.addButton(btn, i + 1)
            ref_layout.addWidget(btn)

        # Default selection: first omics condition (or solution if no omics)
        if conditions:
            self._button_group.button(1).setChecked(True)
        elif solution is not None:
            solution_btn.setChecked(True)

        self._conditions = conditions
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)

        # ── Current scenario summary ───────────────────────────────────────
        scen_values = appdata.project.scen_values
        if scen_values:
            model = appdata.project.cobra_py_model
            scen_group = QGroupBox("Current scenario")
            scen_layout = QVBoxLayout()
            scen_layout.setSpacing(4)

            scen_layout.addWidget(
                QLabel(f"{len(scen_values)} constraint(s) will be applied to the model automatically.")
            )

            list_widget = QListWidget()
            list_widget.setSelectionMode(QListWidget.NoSelection)
            list_widget.setFocusPolicy(list_widget.focusPolicy() & ~0x1)  # no Tab focus

            for rxn_id, (lb, ub) in scen_values.items():
                # Try to get a human-readable reaction name
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                    display_name = rxn.name if rxn.name else rxn_id
                except KeyError:
                    display_name = rxn_id

                if lb == 0 and ub == 0:
                    label = f"  {rxn_id}  ({display_name})  →  KO"
                else:
                    label = f"  {rxn_id}  ({display_name})  →  [{lb}, {ub}]"

                item = QListWidgetItem(label)
                list_widget.addItem(item)

            # Limit visible height so dialog doesn't grow too tall
            row_h = list_widget.sizeHintForRow(0)
            visible = min(len(scen_values), _MAX_SCENARIO_ROWS)
            list_widget.setFixedHeight(row_h * visible + 4)

            scen_layout.addWidget(list_widget)
            scen_group.setLayout(scen_layout)
            layout.addWidget(scen_group)

        # ── OK / Cancel ────────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_reference(self) -> tuple:
        """Returns ("solution", None) or ("omics", condition_name)."""
        checked_id = self._button_group.checkedId()
        if checked_id == 0:
            return ("solution", None)
        condition = self._conditions[checked_id - 1]
        return ("omics", condition)
