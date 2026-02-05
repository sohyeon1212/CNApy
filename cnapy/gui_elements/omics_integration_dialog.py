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

"""Omics integration dialog for transcriptome-based flux prediction.

This module provides two methods for integrating gene expression data:

1. LAD (Least Absolute Deviation) method:
   - Predicts flux distributions by minimizing the absolute deviation
     between predicted fluxes and expression-based targets.

2. E-Flux2 method:
   - Constrains reaction upper bounds based on gene expression levels.
   - Performs FBA with minimization of L2 norm for unique flux solution.
   - Reference: Kim & Lun (2016), PLOS Computational Biology.

This module was added/modified as part of CNApy enhancements.
"""

import cobra
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cnapy.appdata import AppData


def read_expression_data(filename: str) -> dict[str, float]:
    """
    Read gene expression data from a file.

    Supported formats:
    - CSV/TSV: First column = gene ID, second column = expression value
    - Excel: First column = gene ID, second column = expression value

    Parameters:
    -----------
    filename : str
        Path to the expression data file

    Returns:
    --------
    Dict[str, float]
        Dictionary mapping gene IDs to expression values
    """
    expression_info = {}

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(filename, header=0)
    elif filename.endswith(".csv"):
        df = pd.read_csv(filename, header=0)
    else:
        # Try tab-separated
        df = pd.read_csv(filename, sep="\t", header=0)

    # Assume first column is gene ID, second column is expression value
    if len(df.columns) >= 2:
        for idx, row in df.iterrows():
            gene_id = str(row.iloc[0]).strip()
            try:
                value = float(row.iloc[1])
                expression_info[gene_id] = value
            except (ValueError, TypeError):
                continue

    return expression_info


def read_multi_condition_expression_data(filename: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Read gene expression data with multiple conditions from a file.

    Supported formats:
    - CSV/TSV: First column = gene ID, remaining columns = expression values per condition
    - Excel: First column = gene ID, remaining columns = expression values per condition

    Parameters:
    -----------
    filename : str
        Path to the expression data file

    Returns:
    --------
    tuple[list[str], dict[str, dict[str, float]]]
        - conditions: List of condition names (column headers)
        - data: Dictionary mapping gene IDs to dict of {condition: expression_value}

    Example file format:
        gene_id,WT,Drug1,Drug2
        b0001,5.23,3.12,4.56
        b0002,8.45,7.89,6.01
    """
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(filename, header=0, index_col=0)
    elif filename.endswith(".csv"):
        df = pd.read_csv(filename, header=0, index_col=0)
    else:
        # Try tab-separated
        df = pd.read_csv(filename, sep="\t", header=0, index_col=0)

    conditions = list(df.columns)
    data = {}

    for gene_id in df.index:
        gene_id_str = str(gene_id).strip()
        gene_values = {}
        for col in conditions:
            try:
                value = float(df.loc[gene_id, col])
                if not np.isnan(value):
                    gene_values[col] = value
            except (ValueError, TypeError):
                continue
        if gene_values:
            data[gene_id_str] = gene_values

    return conditions, data


def gene_expression_to_reaction_weights(
    model: cobra.Model, gene_expression: dict[str, float], aggregation_method: str = "min"
) -> dict[str, float]:
    """
    Convert gene expression values to reaction weights using GPR rules.

    Parameters:
    -----------
    model : cobra.Model
        The metabolic model
    gene_expression : Dict[str, float]
        Dictionary mapping gene IDs to expression values
    aggregation_method : str
        How to aggregate gene expression for reactions with multiple genes:
        - 'min': Use minimum expression (AND logic)
        - 'max': Use maximum expression (OR logic)
        - 'mean': Use average expression
        - 'sum': Use sum of expression

    Returns:
    --------
    Dict[str, float]
        Dictionary mapping reaction IDs to weights
    """
    reaction_weights = {}

    for reaction in model.reactions:
        if not reaction.genes:
            # No genes associated, skip
            continue

        # Collect expression values for genes in this reaction
        gene_values = []
        for gene in reaction.genes:
            gene_id = gene.id
            # Try different ID formats
            if gene_id in gene_expression:
                gene_values.append(gene_expression[gene_id])
            elif gene_id.upper() in gene_expression:
                gene_values.append(gene_expression[gene_id.upper()])
            elif gene_id.lower() in gene_expression:
                gene_values.append(gene_expression[gene_id.lower()])

        if not gene_values:
            continue

        # Aggregate gene expression values
        if aggregation_method == "min":
            weight = min(gene_values)
        elif aggregation_method == "max":
            weight = max(gene_values)
        elif aggregation_method == "mean":
            weight = np.mean(gene_values)
        elif aggregation_method == "sum":
            weight = sum(gene_values)
        else:
            weight = min(gene_values)  # default to min

        reaction_weights[reaction.id] = weight

    return reaction_weights


def run_lad_fitting(
    model: cobra.Model,
    reaction_weights: dict[str, float],
    flux_constraints: dict[str, tuple[float, float]] | None = None,
    weight_threshold: float = 0.01,
    scaling_factor: float = 1.0,
) -> tuple[str, float | None, dict[str, float] | None]:
    """
    Run LAD (Least Absolute Deviation) fitting to predict fluxes from expression data.

    Parameters:
    -----------
    model : cobra.Model
        The metabolic model
    reaction_weights : Dict[str, float]
        Dictionary mapping reaction IDs to target weights (from expression data)
    flux_constraints : Dict[str, Tuple[float, float]], optional
        Additional flux constraints as {reaction_id: (lb, ub)}
    weight_threshold : float
        Minimum weight to include a reaction in the fitting
    scaling_factor : float
        Scaling factor for target fluxes

    Returns:
    --------
    Tuple[str, Optional[float], Optional[Dict[str, float]]]
        (status, objective_value, flux_distribution)
    """
    from optlang.symbolics import Add

    # Filter reactions by weight threshold
    target_reactions = {
        rid: w for rid, w in reaction_weights.items() if rid in model.reactions and abs(w) > weight_threshold
    }

    if not target_reactions:
        return "no_targets", None, None

    with model as m:
        # Apply flux constraints if provided
        if flux_constraints:
            for rid, (lb, ub) in flux_constraints.items():
                if rid in m.reactions:
                    rxn = m.reactions.get_by_id(rid)
                    rxn.bounds = (lb, ub)

        # Create deviation variables for target reactions
        vars_to_add = []
        cons_to_add = []

        # Store deviation variables for later access
        delta_pos_vars = {}
        delta_neg_vars = {}

        for rid, weight in target_reactions.items():
            rxn = m.reactions.get_by_id(rid)
            target_flux = abs(weight) * scaling_factor

            # Create positive and negative deviation variables
            # flux = target + delta_pos - delta_neg
            # Total deviation = delta_pos + delta_neg
            delta_pos = m.problem.Variable(f"delta_pos_{rid}", lb=0)
            delta_neg = m.problem.Variable(f"delta_neg_{rid}", lb=0)
            delta_pos_vars[rid] = delta_pos
            delta_neg_vars[rid] = delta_neg
            vars_to_add.extend([delta_pos, delta_neg])

            # flux = target + delta_pos - delta_neg
            cons_decomp = m.problem.Constraint(
                rxn.flux_expression - delta_pos + delta_neg, lb=target_flux, ub=target_flux, name=f"lad_decomp_{rid}"
            )
            cons_to_add.append(cons_decomp)

        m.add_cons_vars(vars_to_add)
        m.add_cons_vars(cons_to_add)

        # Objective: minimize sum of deviations
        objective_terms = []
        for rid in target_reactions:
            objective_terms.extend([delta_pos_vars[rid], delta_neg_vars[rid]])

        m.objective = m.problem.Objective(Add(*objective_terms), direction="min")

        # Optimize
        solution = m.optimize()

        if solution.status == "optimal":
            flux_distribution = {rxn.id: solution.fluxes[rxn.id] for rxn in m.reactions}
            return "optimal", solution.objective_value, flux_distribution
        else:
            return solution.status, None, None


def run_eflux2(
    model: cobra.Model,
    reaction_weights: dict[str, float],
    flux_constraints: dict[str, tuple[float, float]] | None = None,
    normalize: bool = True,
    weight_threshold: float = 0.01,
    min_flux_bound: float = 0.001,
) -> tuple[str, float | None, dict[str, float] | None]:
    """
    Run E-Flux2 analysis to predict fluxes from expression data.

    E-Flux2 constrains reaction upper bounds based on gene expression levels,
    then performs FBA with L2-norm minimization for a unique solution.

    Reference: Kim & Lun (2016). E-Flux2 and SPOT: Validated Methods for
    Inferring Intracellular Metabolic Fluxes. PLOS Computational Biology.

    Parameters:
    -----------
    model : cobra.Model
        The metabolic model
    reaction_weights : Dict[str, float]
        Dictionary mapping reaction IDs to expression-based weights
    flux_constraints : Dict[str, Tuple[float, float]], optional
        Additional flux constraints as {reaction_id: (lb, ub)}
    normalize : bool
        Whether to normalize expression values (0-1 range)
    weight_threshold : float
        Minimum weight to constrain a reaction
    min_flux_bound : float
        Minimum flux bound to apply (prevents zero bounds)

    Returns:
    --------
    Tuple[str, Optional[float], Optional[Dict[str, float]]]
        (status, objective_value, flux_distribution)
    """
    from optlang.symbolics import Add

    if not reaction_weights:
        return "no_targets", None, None

    # Filter reactions by weight threshold
    valid_weights = {rid: w for rid, w in reaction_weights.items() if rid in model.reactions and w > weight_threshold}

    if not valid_weights:
        return "no_targets", None, None

    # Normalize weights to 0-1 range if requested
    if normalize and valid_weights:
        max_weight = max(valid_weights.values())
        min_weight = min(valid_weights.values())
        weight_range = max_weight - min_weight
        if weight_range > 0:
            normalized_weights = {rid: (w - min_weight) / weight_range for rid, w in valid_weights.items()}
        else:
            normalized_weights = {rid: 1.0 for rid in valid_weights}
    else:
        normalized_weights = valid_weights.copy()

    with model as m:
        # Apply flux constraints if provided (e.g., from scenario)
        if flux_constraints:
            for rid, (lb, ub) in flux_constraints.items():
                if rid in m.reactions:
                    rxn = m.reactions.get_by_id(rid)
                    rxn.bounds = (lb, ub)

        # E-Flux: Constrain upper bounds based on expression levels
        # Higher expression = higher allowed flux
        for rxn in m.reactions:
            if rxn.id in normalized_weights:
                weight = normalized_weights[rxn.id]
                # Scale the upper bound by the normalized expression
                # Keep a minimum bound to allow some flux
                new_ub = max(min_flux_bound, weight * abs(rxn.upper_bound) if rxn.upper_bound else weight * 1000)
                new_lb = -new_ub if rxn.reversibility else max(rxn.lower_bound, 0)

                # Don't make bounds more restrictive than existing scenario constraints
                if flux_constraints and rxn.id in flux_constraints:
                    continue

                rxn.upper_bound = new_ub
                rxn.lower_bound = new_lb
            elif rxn.id not in normalized_weights and rxn.genes:
                # Reactions with genes but no expression data: keep original bounds
                # or apply minimal constraint
                pass

        # E-Flux2: Add L2-norm minimization (quadratic objective)
        # We use a two-step approach:
        # 1. First FBA to get optimal objective value
        # 2. Then minimize sum of squared fluxes while maintaining optimal objective

        # Step 1: Standard FBA
        try:
            solution1 = m.optimize()
        except Exception as e:
            return f"optimization_error: {str(e)}", None, None

        if solution1.status != "optimal":
            return solution1.status, None, None

        optimal_value = solution1.objective_value

        # Step 2: Minimize L2 norm while maintaining objective
        # Fix objective to optimal value (with small tolerance)
        tolerance = abs(optimal_value) * 0.001 if optimal_value != 0 else 0.001

        obj_constraint = m.problem.Constraint(
            m.objective.expression,
            lb=optimal_value - tolerance,
            ub=optimal_value + tolerance,
            name="eflux2_obj_constraint",
        )
        m.add_cons_vars(obj_constraint)

        # Create squared flux variables and constraints
        # For linear solvers, we approximate by minimizing sum of absolute values
        squared_vars = []
        squared_cons = []

        for rxn in m.reactions:
            # Create auxiliary variable for |flux|
            abs_var = m.problem.Variable(f"abs_{rxn.id}", lb=0)
            squared_vars.append(abs_var)

            # |v| >= v and |v| >= -v
            cons1 = m.problem.Constraint(abs_var - rxn.flux_expression, lb=0, name=f"abs_pos_{rxn.id}")
            cons2 = m.problem.Constraint(abs_var + rxn.flux_expression, lb=0, name=f"abs_neg_{rxn.id}")
            squared_cons.extend([cons1, cons2])

        m.add_cons_vars(squared_vars)
        m.add_cons_vars(squared_cons)

        # Minimize sum of |fluxes| (L1 norm as approximation to L2)
        m.objective = m.problem.Objective(Add(*squared_vars), direction="min")

        try:
            solution2 = m.optimize()
        except Exception:
            # Fall back to first solution if L2 minimization fails
            flux_distribution = {rxn.id: solution1.fluxes[rxn.id] for rxn in m.reactions}
            return "optimal", optimal_value, flux_distribution

        if solution2.status == "optimal":
            flux_distribution = {rxn.id: solution2.fluxes[rxn.id] for rxn in m.reactions}
            return "optimal", optimal_value, flux_distribution
        else:
            # Fall back to first solution
            flux_distribution = {rxn.id: solution1.fluxes[rxn.id] for rxn in m.reactions}
            return "optimal", optimal_value, flux_distribution


class OmicsIntegrationDialog(QDialog):
    """Dialog for Omics integration using LAD or E-Flux2 methods.

    Supports both single-condition and multi-condition expression data files.
    Multi-condition files allow batch analysis across multiple experimental conditions.
    """

    def __init__(self, appdata: AppData, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Omics Integration - Multi-Condition Flux Prediction")
        self.setMinimumWidth(900)
        self.setMinimumHeight(700)
        self.appdata = appdata

        # Single-condition data (legacy support)
        self.expression_data: dict[str, float] = {}
        self.reaction_weights: dict[str, float] = {}

        # Multi-condition data
        self.conditions: list[str] = []
        self.multi_expression_data: dict[str, dict[str, float]] = {}
        self.multi_results: dict[str, dict[str, float]] = {}
        self.is_multi_condition = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # File selection group
        file_group = QGroupBox("Expression Data")
        file_layout = QVBoxLayout()

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Expression file:"))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select CSV, TSV, or Excel file with gene expression data")
        self.file_edit.setReadOnly(True)
        file_row.addWidget(self.file_edit)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.browse_btn)
        file_layout.addLayout(file_row)

        info_row = QHBoxLayout()
        self.gene_count_label = QLabel("Genes loaded: 0")
        info_row.addWidget(self.gene_count_label)
        self.matched_count_label = QLabel("Matched to model: 0")
        info_row.addWidget(self.matched_count_label)
        self.conditions_label = QLabel("Conditions: 0")
        info_row.addWidget(self.conditions_label)
        info_row.addStretch()
        file_layout.addLayout(info_row)

        # Example file format display
        example_label = QLabel(
            "<b>Expected File Format (Multi-Condition):</b><br>"
            "<pre style='background-color: #f0f0f0; padding: 8px; font-family: monospace;'>"
            "gene_id,WT,Drug1,Drug2\n"
            "b0001,5.23,3.12,4.56\n"
            "b0002,8.45,7.89,6.01\n"
            "...</pre>"
            "<b>Requirements:</b><br>"
            "• <b>Column 1</b>: Gene ID (must match model gene IDs)<br>"
            "• <b>Column 2+</b>: Expression values for each condition<br>"
            "• <b>Header row</b>: Condition names (WT, Drug1, Drug2, etc.)"
        )
        example_label.setWordWrap(True)
        example_label.setStyleSheet(
            "QLabel { background-color: #fafafa; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }"
        )
        file_layout.addWidget(example_label)

        # Data preview label (shown after loading)
        self.data_preview_label = QLabel("")
        self.data_preview_label.setWordWrap(True)
        self.data_preview_label.setVisible(False)
        self.data_preview_label.setStyleSheet(
            "QLabel { background-color: #e8f5e9; padding: 10px; border: 1px solid #c8e6c9; border-radius: 4px; }"
        )
        file_layout.addWidget(self.data_preview_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Main content area with splitter
        content_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Conditions and Parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Condition selection group
        condition_group = QGroupBox("Conditions to Analyze")
        condition_layout = QVBoxLayout()

        self.condition_list = QListWidget()
        self.condition_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.condition_list.setMaximumHeight(120)
        condition_layout.addWidget(self.condition_list)

        cond_btn_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all_conditions)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self._deselect_all_conditions)
        cond_btn_row.addWidget(self.select_all_btn)
        cond_btn_row.addWidget(self.deselect_all_btn)
        condition_layout.addLayout(cond_btn_row)

        condition_group.setLayout(condition_layout)
        left_layout.addWidget(condition_group)

        # Method selection group
        method_group = QGroupBox("Integration Method")
        method_layout = QVBoxLayout()

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["LAD (Least Absolute Deviation)", "E-Flux2 (Expression-based FBA)"])
        self.method_combo.setToolTip(
            "LAD: Minimizes absolute deviation between fluxes and expression targets.\n"
            "E-Flux2: Constrains upper bounds based on expression, then optimizes growth."
        )
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self.method_combo)
        method_layout.addLayout(method_row)

        self.method_desc_label = QLabel(
            "LAD minimizes the deviation between predicted fluxes and expression-derived targets."
        )
        self.method_desc_label.setWordWrap(True)
        self.method_desc_label.setStyleSheet("color: gray; font-style: italic;")
        method_layout.addWidget(self.method_desc_label)

        method_group.setLayout(method_layout)
        left_layout.addWidget(method_group)

        # Parameters group
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QVBoxLayout()

        agg_row = QHBoxLayout()
        agg_row.addWidget(QLabel("Gene aggregation:"))
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(["min (AND logic)", "max (OR logic)", "mean", "sum"])
        agg_row.addWidget(self.agg_combo)
        params_layout.addLayout(agg_row)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Weight threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setRange(0, 1000)
        self.threshold_spin.setValue(0.01)
        threshold_row.addWidget(self.threshold_spin)
        params_layout.addLayout(threshold_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scaling factor:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0001, 10000)
        self.scale_spin.setValue(1.0)
        scale_row.addWidget(self.scale_spin)
        params_layout.addLayout(scale_row)

        self.use_scenario_check = QCheckBox("Apply scenario constraints")
        self.use_scenario_check.setChecked(True)
        params_layout.addWidget(self.use_scenario_check)

        # E-Flux2 specific parameters
        self.eflux2_params_widget = QGroupBox("E-Flux2 Options")
        eflux2_layout = QVBoxLayout()

        self.normalize_check = QCheckBox("Normalize expression (0-1)")
        self.normalize_check.setChecked(True)
        eflux2_layout.addWidget(self.normalize_check)

        minflux_row = QHBoxLayout()
        minflux_row.addWidget(QLabel("Min flux bound:"))
        self.minflux_spin = QDoubleSpinBox()
        self.minflux_spin.setDecimals(4)
        self.minflux_spin.setRange(0.0, 100)
        self.minflux_spin.setValue(0.001)
        minflux_row.addWidget(self.minflux_spin)
        eflux2_layout.addLayout(minflux_row)

        self.eflux2_params_widget.setLayout(eflux2_layout)
        self.eflux2_params_widget.setVisible(False)
        params_layout.addWidget(self.eflux2_params_widget)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        left_layout.addStretch()
        content_splitter.addWidget(left_panel)

        # Right panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()

        self.results_tabs = QTabWidget()

        # Comparison table tab
        self.comparison_table = QTableWidget()
        self.comparison_table.setAlternatingRowColors(True)
        self.results_tabs.addTab(self.comparison_table, "Comparison")

        results_layout.addWidget(self.results_tabs)

        # Result actions
        result_actions = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Selected to Main View")
        self.apply_btn.setToolTip("Apply the selected condition's flux values to the main view")
        self.apply_btn.clicked.connect(self._apply_selected_to_main)
        self.apply_btn.setEnabled(False)
        result_actions.addWidget(self.apply_btn)

        self.export_btn = QPushButton("Export Results...")
        self.export_btn.setToolTip("Export all results to CSV file")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        result_actions.addWidget(self.export_btn)

        results_layout.addLayout(result_actions)

        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([350, 550])

        layout.addWidget(content_splitter)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Bottom buttons
        btn_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self._run_multi_analysis)
        self.run_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self._on_method_changed(self.method_combo.currentText())

    def _on_method_changed(self, method_text: str):
        """Handle method selection change."""
        is_eflux2 = "E-Flux2" in method_text
        self.eflux2_params_widget.setVisible(is_eflux2)

        if is_eflux2:
            self.method_desc_label.setText(
                "E-Flux2 constrains reaction bounds based on expression, then optimizes growth."
            )
            self.run_btn.setText("Run E-Flux2 Analysis")
        else:
            self.method_desc_label.setText(
                "LAD minimizes the deviation between predicted fluxes and expression targets."
            )
            self.run_btn.setText("Run LAD Analysis")

    def _browse_file(self):
        """Open file dialog to select expression data file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Expression Data File",
            self.appdata.work_directory,
            "Data files (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*.*)",
        )

        if filename:
            self.file_edit.setText(filename)
            self._load_expression_data(filename)

    def _load_expression_data(self, filename: str):
        """Load expression data from the selected file (auto-detect single or multi-condition)."""
        try:
            # Try to load as multi-condition first
            self.conditions, self.multi_expression_data = read_multi_condition_expression_data(filename)

            # Check if it's really multi-condition or single-condition
            if len(self.conditions) == 1:
                # Single condition - convert to legacy format for backward compatibility
                self.is_multi_condition = False
                self.expression_data = {
                    gene: values[self.conditions[0]]
                    for gene, values in self.multi_expression_data.items()
                }
            else:
                self.is_multi_condition = True
                # Also create legacy single-condition data using first condition
                if self.conditions:
                    self.expression_data = {
                        gene: values.get(self.conditions[0], 0.0)
                        for gene, values in self.multi_expression_data.items()
                    }

            # Update condition list widget
            self.condition_list.clear()
            for cond in self.conditions:
                item = QListWidgetItem(cond)
                item.setSelected(True)
                self.condition_list.addItem(item)

            # Update info labels
            self.gene_count_label.setText(f"Genes loaded: {len(self.multi_expression_data)}")
            self.conditions_label.setText(f"Conditions: {len(self.conditions)}")

            # Count genes that match model genes
            model_genes = {g.id for g in self.appdata.project.cobra_py_model.genes}
            matched = sum(1 for g in self.multi_expression_data if g in model_genes)
            self.matched_count_label.setText(f"Matched to model: {matched}")

            # Show data preview
            if len(self.multi_expression_data) > 0:
                preview_html = (
                    f"<b>Data Preview:</b><br>"
                    f"Total genes: {len(self.multi_expression_data)}<br>"
                    f"Matched to model: {matched}<br>"
                    f"Conditions: {', '.join(self.conditions[:5])}"
                    f"{'...' if len(self.conditions) > 5 else ''}<br><br>"
                    f"<b>Sample data (first 3 genes):</b><br>"
                )
                for i, (gene_id, values) in enumerate(list(self.multi_expression_data.items())[:3]):
                    match_status = "+" if gene_id in model_genes else "-"
                    val_str = ", ".join(f"{c}:{values.get(c, 0):.2f}" for c in self.conditions[:3])
                    preview_html += f"&nbsp;&nbsp;{match_status} {gene_id}: {val_str}<br>"

                self.data_preview_label.setText(preview_html)
                self.data_preview_label.setVisible(True)
            else:
                self.data_preview_label.setVisible(False)

            # Enable run button if we have data
            self.run_btn.setEnabled(len(self.multi_expression_data) > 0 and matched > 0)

            if len(self.multi_expression_data) == 0:
                QMessageBox.warning(self, "No data", "No expression data could be loaded from the file.")
            elif matched == 0:
                QMessageBox.warning(
                    self,
                    "No matches",
                    "No genes from the expression data match genes in the model.\n"
                    "Check that gene IDs use the same format.",
                )

        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load expression data:\n{str(e)}")
            self.multi_expression_data = {}
            self.conditions = []
            self.gene_count_label.setText("Genes loaded: 0")
            self.matched_count_label.setText("Matched to model: 0")
            self.conditions_label.setText("Conditions: 0")
            self.data_preview_label.setVisible(False)
            self.run_btn.setEnabled(False)

    def _select_all_conditions(self):
        """Select all conditions in the list."""
        for i in range(self.condition_list.count()):
            self.condition_list.item(i).setSelected(True)

    def _deselect_all_conditions(self):
        """Deselect all conditions in the list."""
        for i in range(self.condition_list.count()):
            self.condition_list.item(i).setSelected(False)

    def _get_aggregation_method(self) -> str:
        """Get the selected aggregation method."""
        agg_text = self.agg_combo.currentText()
        if "min" in agg_text:
            return "min"
        elif "max" in agg_text:
            return "max"
        elif "mean" in agg_text:
            return "mean"
        else:
            return "sum"

    def _is_eflux2(self) -> bool:
        """Check if E-Flux2 method is selected."""
        return "E-Flux2" in self.method_combo.currentText()

    def _run_multi_analysis(self):
        """Run analysis for all selected conditions."""
        selected_conditions = [
            self.condition_list.item(i).text()
            for i in range(self.condition_list.count())
            if self.condition_list.item(i).isSelected()
        ]

        if not selected_conditions:
            QMessageBox.warning(self, "No conditions", "Please select at least one condition to analyze.")
            return

        if not self.multi_expression_data:
            QMessageBox.warning(self, "No data", "Please load expression data first.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(selected_conditions))
        self.run_btn.setEnabled(False)

        is_eflux2 = self._is_eflux2()
        method_name = "E-Flux2" if is_eflux2 else "LAD"
        agg_method = self._get_aggregation_method()
        threshold = self.threshold_spin.value()

        # Get scenario constraints if enabled
        flux_constraints = {}
        if self.use_scenario_check.isChecked():
            for rid, (lb, ub) in self.appdata.project.scen_values.items():
                flux_constraints[rid] = (lb, ub)

        self.multi_results.clear()
        failed_conditions = []
        successful_conditions = []

        for i, condition in enumerate(selected_conditions):
            self.progress_bar.setValue(i)
            self.progress_bar.setFormat(f"Analyzing {condition}... (%p%)")

            try:
                # Extract expression data for this condition
                condition_expression = {
                    gene: values[condition]
                    for gene, values in self.multi_expression_data.items()
                    if condition in values
                }

                # Compute reaction weights
                reaction_weights = gene_expression_to_reaction_weights(
                    self.appdata.project.cobra_py_model, condition_expression, agg_method
                )

                if not reaction_weights:
                    failed_conditions.append((condition, "No reaction weights computed"))
                    continue

                # Run analysis
                model = self.appdata.project.cobra_py_model.copy()

                if is_eflux2:
                    normalize = self.normalize_check.isChecked()
                    min_flux_bound = self.minflux_spin.value()
                    status, obj_value, flux_dist = run_eflux2(
                        model,
                        reaction_weights,
                        flux_constraints,
                        normalize=normalize,
                        weight_threshold=threshold,
                        min_flux_bound=min_flux_bound,
                    )
                else:
                    scaling = self.scale_spin.value()
                    status, obj_value, flux_dist = run_lad_fitting(
                        model, reaction_weights, flux_constraints, threshold, scaling
                    )

                if status == "optimal" and flux_dist:
                    self.multi_results[condition] = flux_dist
                    successful_conditions.append(condition)
                else:
                    failed_conditions.append((condition, f"Optimization status: {status}"))

            except Exception as e:
                failed_conditions.append((condition, str(e)))

        self.progress_bar.setValue(len(selected_conditions))
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        # Update results display
        self._update_results_display()

        # Store results in appdata for main view access
        self.appdata.project.omics_results = self.multi_results.copy()
        self.appdata.project.omics_conditions = list(self.multi_results.keys())

        # Update central widget if available
        if hasattr(self.parent(), "centralWidget"):
            central = self.parent().centralWidget()
            if hasattr(central, "update_omics_selector"):
                central.update_omics_selector()

        # Show summary
        msg = f"{method_name} analysis completed.\n\n"
        msg += f"Successful: {len(successful_conditions)} conditions\n"
        if successful_conditions:
            msg += f"  ({', '.join(successful_conditions[:5])}"
            if len(successful_conditions) > 5:
                msg += f"... +{len(successful_conditions)-5} more"
            msg += ")\n"

        if failed_conditions:
            msg += f"\nFailed: {len(failed_conditions)} conditions\n"
            for cond, reason in failed_conditions[:3]:
                msg += f"  - {cond}: {reason}\n"
            if len(failed_conditions) > 3:
                msg += f"  ... and {len(failed_conditions)-3} more\n"

        if successful_conditions:
            msg += "\nResults are available in the comparison table."
            QMessageBox.information(self, "Analysis Complete", msg)
        else:
            QMessageBox.warning(self, "Analysis Failed", msg)

    def _update_results_display(self):
        """Update the comparison table with multi-condition results."""
        if not self.multi_results:
            self.comparison_table.setRowCount(0)
            self.comparison_table.setColumnCount(0)
            self.apply_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            return

        conditions = list(self.multi_results.keys())

        # Collect all reactions
        all_reactions = set()
        for flux_dist in self.multi_results.values():
            all_reactions.update(flux_dist.keys())
        reactions = sorted(all_reactions)

        # Setup table
        self.comparison_table.setRowCount(len(reactions))
        self.comparison_table.setColumnCount(len(conditions) + 1)
        self.comparison_table.setHorizontalHeaderLabels(["Reaction"] + conditions)

        # Fill table
        for row, rxn_id in enumerate(reactions):
            # Reaction ID
            item = QTableWidgetItem(rxn_id)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.comparison_table.setItem(row, 0, item)

            # Flux values for each condition
            for col, cond in enumerate(conditions, 1):
                flux = self.multi_results[cond].get(rxn_id, 0.0)
                item = QTableWidgetItem(f"{flux:.4f}")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.comparison_table.setItem(row, col, item)

        # Resize columns
        self.comparison_table.resizeColumnsToContents()

        # Enable action buttons
        self.apply_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _apply_selected_to_main(self):
        """Apply the first selected condition's results to the main view."""
        if not self.multi_results:
            return

        conditions = list(self.multi_results.keys())
        if not conditions:
            return

        # Show condition selection dialog
        from qtpy.QtWidgets import QInputDialog

        condition, ok = QInputDialog.getItem(
            self,
            "Select Condition",
            "Choose condition to apply to main view:",
            conditions,
            0,
            False,
        )

        if not ok or condition not in self.multi_results:
            return

        # Apply to comp_values
        self.appdata.project.comp_values.clear()
        for rxn_id, flux in self.multi_results[condition].items():
            self.appdata.project.comp_values[rxn_id] = (flux, flux)

        # Update central widget
        if hasattr(self.parent(), "centralWidget"):
            central = self.parent().centralWidget()
            central.update()
            if hasattr(central, "set_current_omics_condition"):
                central.set_current_omics_condition(condition)

        QMessageBox.information(
            self,
            "Applied",
            f"Flux values from condition '{condition}' have been applied to the main view.",
        )

    def _export_results(self):
        """Export all results to a CSV file."""
        if not self.multi_results:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            self.appdata.work_directory,
            "CSV files (*.csv);;All files (*.*)",
        )

        if not filename:
            return

        if not filename.endswith(".csv"):
            filename += ".csv"

        try:
            conditions = list(self.multi_results.keys())
            all_reactions = set()
            for flux_dist in self.multi_results.values():
                all_reactions.update(flux_dist.keys())
            reactions = sorted(all_reactions)

            # Create DataFrame
            data = {"Reaction": reactions}
            for cond in conditions:
                data[cond] = [self.multi_results[cond].get(rxn, 0.0) for rxn in reactions]

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported to:\n{filename}\n\n"
                f"Exported {len(reactions)} reactions across {len(conditions)} conditions.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
