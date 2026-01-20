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

"""Flux Sampling Dialog for CNApy

Provides two sampling modes:
1. Random Sampling: Standard hit-and-run sampling from the feasible flux space
2. Predicted Flux-Based Sampling: Sampling centered around a reference flux solution
"""

from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from cnapy.appdata import AppData


class FluxSamplingDialog(QDialog):
    """Dialog for configuring flux sampling parameters."""

    def __init__(self, appdata: AppData):
        QDialog.__init__(self)
        self.setWindowTitle("Flux Sampling")
        self.setMinimumWidth(450)
        self.appdata = appdata

        self.layout = QVBoxLayout()

        # Sampling mode selection
        mode_group = QGroupBox("Sampling Mode")
        mode_layout = QVBoxLayout()

        self.mode_group = QButtonGroup(self)

        self.random_mode = QRadioButton("Random Sampling")
        self.random_mode.setToolTip(
            "Standard flux sampling using hit-and-run algorithm.\nSamples uniformly from the feasible flux space."
        )
        self.random_mode.setChecked(True)
        self.mode_group.addButton(self.random_mode, 0)
        mode_layout.addWidget(self.random_mode)

        self.predicted_mode = QRadioButton("Predicted Flux-Based Sampling")
        self.predicted_mode.setToolTip(
            "Sample around a reference flux distribution.\n"
            "Uses computed values (from FBA, MOMA, etc.) as the center.\n"
            "Useful for uncertainty quantification."
        )
        self.mode_group.addButton(self.predicted_mode, 1)
        mode_layout.addWidget(self.predicted_mode)

        # Check if computed values exist
        has_computed = len(self.appdata.project.comp_values) > 0
        if not has_computed:
            self.predicted_mode.setEnabled(False)
            self.predicted_mode.setToolTip(
                "No computed flux values available.\nRun FBA, MOMA, or other analysis first."
            )

        mode_group.setLayout(mode_layout)
        self.layout.addWidget(mode_group)

        # Predicted flux-based options (shown when predicted mode is selected)
        self.predicted_options_group = QGroupBox("Predicted Flux Options")
        pred_layout = QVBoxLayout()

        # Constraint mode
        constraint_row = QHBoxLayout()
        constraint_row.addWidget(QLabel("Constraint mode:"))
        self.constraint_combo = QComboBox()
        self.constraint_combo.addItems(
            ["Bounds (sample within range around reference)", "Free (use reference as starting point only)"]
        )
        self.constraint_combo.setToolTip(
            "Bounds: Constrain flux bounds to a range around reference values.\n"
            "Free: Don't constrain bounds, use reference only for initialization."
        )
        constraint_row.addWidget(self.constraint_combo)
        pred_layout.addLayout(constraint_row)

        # Range parameters
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Min fraction:"))
        self.min_fraction_spin = QDoubleSpinBox()
        self.min_fraction_spin.setRange(0.0, 1.0)
        self.min_fraction_spin.setValue(0.8)
        self.min_fraction_spin.setSingleStep(0.1)
        self.min_fraction_spin.setToolTip("Lower bound = reference * min_fraction")
        range_row.addWidget(self.min_fraction_spin)

        range_row.addWidget(QLabel("Max fraction:"))
        self.max_fraction_spin = QDoubleSpinBox()
        self.max_fraction_spin.setRange(1.0, 10.0)
        self.max_fraction_spin.setValue(1.2)
        self.max_fraction_spin.setSingleStep(0.1)
        self.max_fraction_spin.setToolTip("Upper bound = reference * max_fraction")
        range_row.addWidget(self.max_fraction_spin)
        pred_layout.addLayout(range_row)

        # Add noise option
        self.add_noise_check = QCheckBox("Add Gaussian noise to samples")
        self.add_noise_check.setToolTip(
            "Add Gaussian noise centered around the reference values.\nUseful for generating uncertainty estimates."
        )
        pred_layout.addWidget(self.add_noise_check)

        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise std (fraction of flux):"))
        self.noise_std_spin = QDoubleSpinBox()
        self.noise_std_spin.setRange(0.01, 1.0)
        self.noise_std_spin.setValue(0.1)
        self.noise_std_spin.setSingleStep(0.05)
        noise_row.addWidget(self.noise_std_spin)
        pred_layout.addLayout(noise_row)

        self.predicted_options_group.setLayout(pred_layout)
        self.predicted_options_group.setVisible(False)
        self.layout.addWidget(self.predicted_options_group)

        # Connect mode change to show/hide options
        self.random_mode.toggled.connect(self._on_mode_changed)
        self.predicted_mode.toggled.connect(self._on_mode_changed)

        # Basic parameters group
        params_group = QGroupBox("Sampling Parameters")
        params_layout = QVBoxLayout()

        # Number of samples
        l1 = QHBoxLayout()
        l1.addWidget(QLabel("Number of samples:"))
        self.n_samples = QSpinBox()
        self.n_samples.setRange(1, 1000000)
        self.n_samples.setValue(5000)
        l1.addWidget(self.n_samples)
        params_layout.addLayout(l1)

        # Thinning
        l2 = QHBoxLayout()
        l2.addWidget(QLabel("Thinning factor:"))
        self.thinning = QSpinBox()
        self.thinning.setRange(1, 10000)
        self.thinning.setValue(100)
        self.thinning.setToolTip("Keep every nth sample to reduce autocorrelation")
        l2.addWidget(self.thinning)
        params_layout.addLayout(l2)

        # Processes
        l3 = QHBoxLayout()
        l3.addWidget(QLabel("Processes (1 for single core):"))
        self.processes = QSpinBox()
        self.processes.setRange(1, 64)
        self.processes.setValue(4)
        l3.addWidget(self.processes)
        params_layout.addLayout(l3)

        params_group.setLayout(params_layout)
        self.layout.addWidget(params_group)

        # Buttons
        l_btns = QHBoxLayout()
        self.button = QPushButton("Start Sampling")
        self.button.setStyleSheet("font-size: 13px; padding: 8px;")
        self.cancel = QPushButton("Cancel")
        l_btns.addWidget(self.button)
        l_btns.addWidget(self.cancel)
        self.layout.addLayout(l_btns)

        self.setLayout(self.layout)

        self.cancel.clicked.connect(self.reject)
        self.button.clicked.connect(self.accept)

    @Slot(bool)
    def _on_mode_changed(self, checked: bool):
        """Handle sampling mode change."""
        self.predicted_options_group.setVisible(self.predicted_mode.isChecked())
        self.adjustSize()

    def get_sampling_mode(self) -> str:
        """Get the selected sampling mode."""
        if self.predicted_mode.isChecked():
            return "predicted"
        return "random"

    def get_constraint_mode(self) -> str:
        """Get the constraint mode for predicted sampling."""
        if "Bounds" in self.constraint_combo.currentText():
            return "bounds"
        return "free"

    def get_reference_fluxes(self) -> dict:
        """Get reference fluxes from computed values."""
        reference = {}
        for rid, (lb, ub) in self.appdata.project.comp_values.items():
            # Use mean of bounds as reference
            reference[rid] = (lb + ub) / 2
        return reference
