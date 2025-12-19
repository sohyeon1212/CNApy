#!/usr/bin/env python3
#
# Copyright 2022 CNApy organization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-

"""Flux response analysis dialog.

Scan a target reaction flux from min..max on the x-axis.
For each fixed value, maximize a product reaction (y-axis) and plot the result.

This module was enhanced as part of CNApy improvements.
"""

import numpy as np
import matplotlib.pyplot as plt

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from cnapy.appdata import AppData
from cnapy.utils import QComplReceivLineEdit


class FluxResponseDialog(QDialog):
    """Dialog to configure and run flux response analysis."""

    def __init__(self, appdata: AppData, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flux response analysis")
        self.appdata = appdata

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Target reaction (x-axis):"))
        self.target_edit = QComplReceivLineEdit(self, self.appdata.project.reaction_ids, self.appdata.is_in_dark_mode, check=False)
        self.target_edit.setPlaceholderText("e.g. BIOMASS or a pathway reaction id")
        layout.addWidget(self.target_edit)

        layout.addWidget(QLabel("Product reaction to maximize (y-axis):"))
        self.product_edit = QComplReceivLineEdit(self, self.appdata.project.reaction_ids, self.appdata.is_in_dark_mode, check=False)
        self.product_edit.setPlaceholderText("e.g. EX_product_e")
        layout.addWidget(self.product_edit)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Min x:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(6)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setValue(-10.0)
        range_row.addWidget(self.min_spin)
        range_row.addWidget(QLabel("Max x:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(6)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setValue(10.0)
        range_row.addWidget(self.max_spin)
        layout.addLayout(range_row)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 500)
        self.steps_spin.setValue(31)
        steps_row.addWidget(self.steps_spin)
        steps_row.addStretch()
        layout.addLayout(steps_row)

        self.use_scenario = QCheckBox("Use scenario flux values as model bounds")
        self.use_scenario.setChecked(True)
        layout.addWidget(self.use_scenario)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("Run && Plot")
        self.run_btn.clicked.connect(self.run)
        cancel_btn = QPushButton("Close")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.run_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        self.setLayout(layout)

    def _get_reaction_id(self, edit: QComplReceivLineEdit) -> str:
        txt = edit.text().strip()
        if "|" in txt:
            txt = txt.split("|", 1)[0].strip()
        return txt

    def run(self):
        target_id = self._get_reaction_id(self.target_edit)
        product_id = self._get_reaction_id(self.product_edit)
        if not target_id or not product_id:
            QMessageBox.warning(self, "Missing input", "Please enter both target and product reaction ids.")
            return

        model = self.appdata.project.cobra_py_model
        if target_id not in model.reactions or product_id not in model.reactions:
            QMessageBox.warning(self, "Invalid reaction", "Target and product reactions must exist in the current model.")
            return

        x_min = float(self.min_spin.value())
        x_max = float(self.max_spin.value())
        if x_max < x_min:
            x_min, x_max = x_max, x_min
            self.min_spin.setValue(x_min)
            self.max_spin.setValue(x_max)

        steps = int(self.steps_spin.value())
        xs = np.linspace(x_min, x_max, steps)
        ys = np.full_like(xs, np.nan, dtype=float)

        infeasible = 0
        with model as m:
            if self.use_scenario.isChecked():
                try:
                    self.appdata.project.load_scenario_into_model(m)
                except Exception:
                    # best-effort; do not fail analysis just because scenario couldn't load
                    pass

            target_rxn = m.reactions.get_by_id(target_id)
            product_rxn = m.reactions.get_by_id(product_id)

            # Ensure objective is the product reaction
            m.objective = product_rxn
            m.objective_direction = "max"

            for i, x in enumerate(xs):
                # Fix target reaction to x
                target_rxn.bounds = (x, x)
                sol = m.optimize()
                if sol.status != "optimal":
                    infeasible += 1
                    ys[i] = np.nan
                else:
                    try:
                        ys[i] = float(sol.fluxes[product_id])
                    except Exception:
                        ys[i] = np.nan

        plt.figure()
        plt.plot(xs, ys, marker="o", linewidth=1)
        plt.xlabel(f"{target_id} (fixed flux)")
        plt.ylabel(f"max {product_id} (optimal flux)")
        title = "Flux response analysis"
        if infeasible:
            title += f"  (infeasible: {infeasible}/{len(xs)})"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


