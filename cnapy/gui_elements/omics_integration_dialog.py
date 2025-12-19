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

"""Omics integration dialog for transcriptome-based flux prediction using LAD.

LAD (Least Absolute Deviation) method predicts flux distributions
based on gene expression data, assuming that flux through a reaction
correlates with the expression level of its associated genes.

This module was added/modified as part of CNApy enhancements.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
)

import cobra
from cnapy.appdata import AppData


def read_expression_data(filename: str) -> Dict[str, float]:
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
    
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename, header=0)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename, header=0)
    else:
        # Try tab-separated
        df = pd.read_csv(filename, sep='\t', header=0)
    
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


def gene_expression_to_reaction_weights(
    model: cobra.Model,
    gene_expression: Dict[str, float],
    aggregation_method: str = 'min'
) -> Dict[str, float]:
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
        if aggregation_method == 'min':
            weight = min(gene_values)
        elif aggregation_method == 'max':
            weight = max(gene_values)
        elif aggregation_method == 'mean':
            weight = np.mean(gene_values)
        elif aggregation_method == 'sum':
            weight = sum(gene_values)
        else:
            weight = min(gene_values)  # default to min
        
        reaction_weights[reaction.id] = weight
    
    return reaction_weights


def run_lad_fitting(
    model: cobra.Model,
    reaction_weights: Dict[str, float],
    flux_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    weight_threshold: float = 0.01,
    scaling_factor: float = 1.0
) -> Tuple[str, Optional[float], Optional[Dict[str, float]]]:
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
        rid: w for rid, w in reaction_weights.items()
        if rid in model.reactions and abs(w) > weight_threshold
    }
    
    if not target_reactions:
        return 'no_targets', None, None
    
    with model as m:
        # Apply flux constraints if provided
        if flux_constraints:
            for rid, (lb, ub) in flux_constraints.items():
                if rid in m.reactions:
                    rxn = m.reactions.get_by_id(rid)
                    rxn.bounds = (lb, ub)
        
        # Create deviation variables for target reactions
        deviation_vars = {}
        vars_to_add = []
        cons_to_add = []
        
        for rid, weight in target_reactions.items():
            rxn = m.reactions.get_by_id(rid)
            target_flux = abs(weight) * scaling_factor
            
            # Create positive deviation variable
            delta = m.problem.Variable(f"delta_{rid}", lb=0)
            deviation_vars[rid] = delta
            vars_to_add.append(delta)
            
            # Use absolute flux value for comparison
            # |v| = f + b where v = f - b, f >= 0, b >= 0
            # For simplicity, we'll use flux_expression and add constraints
            
            # Constraint: delta >= flux - target
            cons1 = m.problem.Constraint(
                rxn.flux_expression - delta,
                ub=target_flux,
                name=f"lad_upper_{rid}"
            )
            
            # Constraint: delta >= target - flux  
            cons2 = m.problem.Constraint(
                rxn.flux_expression + delta,
                lb=target_flux - 2 * abs(rxn.upper_bound if rxn.upper_bound else 1000),
                name=f"lad_lower_{rid}"
            )
            
            # Actually we need: delta >= |flux - target|
            # This requires: delta >= flux - target AND delta >= -(flux - target)
            # Which is: delta >= flux - target AND delta >= target - flux
            
            # Reformulating:
            # flux - target <= delta
            # target - flux <= delta
            
            delta_pos = m.problem.Variable(f"delta_pos_{rid}", lb=0)
            delta_neg = m.problem.Variable(f"delta_neg_{rid}", lb=0)
            vars_to_add.extend([delta_pos, delta_neg])
            
            # flux = target + delta_pos - delta_neg
            cons_decomp = m.problem.Constraint(
                rxn.flux_expression - delta_pos + delta_neg,
                lb=target_flux,
                ub=target_flux,
                name=f"lad_decomp_{rid}"
            )
            cons_to_add.append(cons_decomp)
        
        m.add_cons_vars(vars_to_add)
        m.add_cons_vars(cons_to_add)
        
        # Objective: minimize sum of deviations
        objective_terms = []
        for rid in target_reactions:
            delta_pos = m.problem.variables[f"delta_pos_{rid}"]
            delta_neg = m.problem.variables[f"delta_neg_{rid}"]
            objective_terms.extend([delta_pos, delta_neg])
        
        m.objective = m.problem.Objective(Add(*objective_terms), direction='min')
        
        # Optimize
        solution = m.optimize()
        
        if solution.status == 'optimal':
            flux_distribution = {rxn.id: solution.fluxes[rxn.id] for rxn in m.reactions}
            return 'optimal', solution.objective_value, flux_distribution
        else:
            return solution.status, None, None


class OmicsIntegrationDialog(QDialog):
    """Dialog for Omics integration using LAD method."""

    def __init__(self, appdata: AppData, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Omics Integration - LAD Flux Prediction")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.appdata = appdata
        
        self.expression_data: Dict[str, float] = {}
        self.reaction_weights: Dict[str, float] = {}
        
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
        info_row.addStretch()
        file_layout.addLayout(info_row)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Parameters group
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QVBoxLayout()
        
        agg_row = QHBoxLayout()
        agg_row.addWidget(QLabel("Gene aggregation method:"))
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(['min (AND logic)', 'max (OR logic)', 'mean', 'sum'])
        self.agg_combo.setToolTip(
            "How to aggregate expression values when multiple genes are associated with a reaction:\n"
            "- min: Use minimum expression (models AND logic in GPR)\n"
            "- max: Use maximum expression (models OR logic in GPR)\n"
            "- mean: Use average expression\n"
            "- sum: Use sum of expression values"
        )
        agg_row.addWidget(self.agg_combo)
        agg_row.addStretch()
        params_layout.addLayout(agg_row)
        
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Weight threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setRange(0, 1000)
        self.threshold_spin.setValue(0.01)
        self.threshold_spin.setToolTip("Minimum expression weight to include a reaction in the fitting")
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        params_layout.addLayout(threshold_row)
        
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scaling factor:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0001, 10000)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setToolTip("Scaling factor applied to expression values")
        scale_row.addWidget(self.scale_spin)
        scale_row.addStretch()
        params_layout.addLayout(scale_row)
        
        self.use_scenario_check = QCheckBox("Apply scenario constraints")
        self.use_scenario_check.setChecked(True)
        self.use_scenario_check.setToolTip("Apply flux constraints from the current scenario")
        params_layout.addWidget(self.use_scenario_check)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Preview group
        preview_group = QGroupBox("Reaction Weights Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(3)
        self.preview_table.setHorizontalHeaderLabels(['Reaction ID', 'Weight', 'Genes'])
        self.preview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.preview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.preview_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_table)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.compute_weights_btn = QPushButton("Compute Reaction Weights")
        self.compute_weights_btn.clicked.connect(self._compute_weights)
        btn_layout.addWidget(self.compute_weights_btn)
        
        self.run_btn = QPushButton("Run LAD Analysis")
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def _browse_file(self):
        """Open file dialog to select expression data file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Expression Data File",
            self.appdata.work_directory,
            "Data files (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*.*)"
        )
        
        if filename:
            self.file_edit.setText(filename)
            self._load_expression_data(filename)
    
    def _load_expression_data(self, filename: str):
        """Load expression data from the selected file."""
        try:
            self.expression_data = read_expression_data(filename)
            self.gene_count_label.setText(f"Genes loaded: {len(self.expression_data)}")
            
            # Count genes that match model genes
            model_genes = {g.id for g in self.appdata.project.cobra_py_model.genes}
            matched = sum(1 for g in self.expression_data if g in model_genes)
            self.matched_count_label.setText(f"Matched to model: {matched}")
            
            if len(self.expression_data) == 0:
                QMessageBox.warning(self, "No data", "No expression data could be loaded from the file.")
            elif matched == 0:
                QMessageBox.warning(self, "No matches", 
                    "No genes from the expression data match genes in the model.\n"
                    "Check that gene IDs use the same format.")
                
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load expression data:\n{str(e)}")
            self.expression_data = {}
            self.gene_count_label.setText("Genes loaded: 0")
            self.matched_count_label.setText("Matched to model: 0")
    
    def _compute_weights(self):
        """Compute reaction weights from gene expression data."""
        if not self.expression_data:
            QMessageBox.warning(self, "No data", "Please load expression data first.")
            return
        
        # Get aggregation method
        agg_text = self.agg_combo.currentText()
        if 'min' in agg_text:
            agg_method = 'min'
        elif 'max' in agg_text:
            agg_method = 'max'
        elif 'mean' in agg_text:
            agg_method = 'mean'
        else:
            agg_method = 'sum'
        
        try:
            self.reaction_weights = gene_expression_to_reaction_weights(
                self.appdata.project.cobra_py_model,
                self.expression_data,
                agg_method
            )
            
            # Update preview table
            self.preview_table.setRowCount(0)
            threshold = self.threshold_spin.value()
            
            # Sort by weight (descending)
            sorted_weights = sorted(self.reaction_weights.items(), key=lambda x: -x[1])
            
            for rid, weight in sorted_weights:
                if abs(weight) < threshold:
                    continue
                    
                row = self.preview_table.rowCount()
                self.preview_table.insertRow(row)
                
                self.preview_table.setItem(row, 0, QTableWidgetItem(rid))
                self.preview_table.setItem(row, 1, QTableWidgetItem(f"{weight:.4f}"))
                
                # Get associated genes
                try:
                    rxn = self.appdata.project.cobra_py_model.reactions.get_by_id(rid)
                    genes = ', '.join(g.id for g in rxn.genes)
                    self.preview_table.setItem(row, 2, QTableWidgetItem(genes))
                except:
                    self.preview_table.setItem(row, 2, QTableWidgetItem(""))
            
            self.run_btn.setEnabled(len(self.reaction_weights) > 0)
            
            if len(self.reaction_weights) == 0:
                QMessageBox.warning(self, "No weights", 
                    "No reaction weights could be computed.\n"
                    "Make sure expression data genes match model genes.")
            else:
                QMessageBox.information(self, "Weights computed", 
                    f"Computed weights for {len(self.reaction_weights)} reactions.\n"
                    f"Showing {self.preview_table.rowCount()} reactions above threshold.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute reaction weights:\n{str(e)}")
    
    def _run_analysis(self):
        """Run LAD analysis to predict fluxes."""
        if not self.reaction_weights:
            QMessageBox.warning(self, "No weights", "Please compute reaction weights first.")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.run_btn.setEnabled(False)
        
        try:
            model = self.appdata.project.cobra_py_model.copy()
            
            # Apply scenario constraints if checked
            flux_constraints = {}
            if self.use_scenario_check.isChecked():
                for rid, (lb, ub) in self.appdata.project.scen_values.items():
                    flux_constraints[rid] = (lb, ub)
            
            threshold = self.threshold_spin.value()
            scaling = self.scale_spin.value()
            
            status, obj_value, flux_dist = run_lad_fitting(
                model,
                self.reaction_weights,
                flux_constraints,
                threshold,
                scaling
            )
            
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            
            if status == 'optimal':
                # Store results in appdata
                self.appdata.project.comp_values.clear()
                for rid, flux in flux_dist.items():
                    self.appdata.project.comp_values[rid] = (flux, flux)
                
                # Update display
                if hasattr(self.parent(), 'centralWidget'):
                    self.parent().centralWidget().update()
                
                QMessageBox.information(self, "Analysis complete", 
                    f"LAD flux prediction completed successfully.\n"
                    f"Objective value (total deviation): {obj_value:.4f}\n\n"
                    f"Results have been loaded into the computed values.")
                
            elif status == 'no_targets':
                QMessageBox.warning(self, "No targets", 
                    "No reaction targets above the weight threshold.\n"
                    "Try lowering the threshold value.")
            else:
                QMessageBox.warning(self, "Optimization failed", 
                    f"LAD optimization failed with status: {status}")
                
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Error", f"LAD analysis failed:\n{str(e)}")

