"""Batch MOMA/ROOM Analysis Dialog for CNApy

This dialog performs batch MOMA (Minimization of Metabolic Adjustment) or
ROOM (Regulatory On/Off Minimization) analysis across multiple gene or
reaction knockouts.

Features:
- Batch analysis for all genes or reactions
- Scatter plot visualization comparing WT vs KO growth
- CSV/XLSX export functionality
"""

import csv

from qtpy.QtCore import Qt, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cnapy.appdata import AppData
from cnapy.moma import has_milp_solver, linear_moma, room

# Check for openpyxl availability for XLSX export
try:
    import openpyxl

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Check for matplotlib availability for scatter plot
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class BatchMomaRoomWorkerThread(QThread):
    """Worker thread for running batch MOMA/ROOM analysis in background."""

    progress_update = Signal(int, int)  # (current, total)
    result_ready = Signal(object)  # dict with results
    error_occurred = Signal(str)

    def __init__(
        self,
        model,
        appdata: AppData,
        analysis_type: str,  # "moma" or "room"
        target_type: str,  # "genes" or "reactions"
        threshold: float,
        room_delta: float = 0.03,
        room_epsilon: float = 0.001,
    ):
        """Initialize the worker thread.

        Args:
            model: COBRApy model (copy)
            appdata: Application data for scenario loading
            analysis_type: "moma" or "room"
            target_type: "genes" or "reactions"
            threshold: Growth ratio threshold for essentiality
            room_delta: ROOM delta parameter
            room_epsilon: ROOM epsilon parameter
        """
        super().__init__()
        self.model = model
        self.appdata = appdata
        self.analysis_type = analysis_type
        self.target_type = target_type
        self.threshold = threshold
        self.room_delta = room_delta
        self.room_epsilon = room_epsilon
        self._cancel_requested = False

    def request_cancel(self):
        """Request cancellation of the analysis."""
        self._cancel_requested = True

    def run(self):
        """Execute the batch MOMA/ROOM analysis."""
        try:
            model = self.model

            # Load scenario into model
            if self.appdata and self.appdata.project:
                self.appdata.project.load_scenario_into_model(model)

            # Get wild-type growth using FBA
            wt_solution = model.optimize()
            if wt_solution.status != "optimal":
                self.error_occurred.emit("Wild-type optimization failed. Check model constraints.")
                return

            wt_growth = wt_solution.objective_value

            if wt_growth < 1e-9:
                self.error_occurred.emit("Wild-type has no growth. Cannot perform analysis.")
                return

            # Get reference fluxes from wild-type solution
            reference_fluxes = {rxn.id: wt_solution.fluxes[rxn.id] for rxn in model.reactions}

            results = []

            if self.target_type == "genes":
                targets = list(model.genes)
                total = len(targets)
                self.progress_update.emit(0, total)

                for idx, gene in enumerate(targets):
                    if self._cancel_requested:
                        return

                    ko_growth = self._analyze_gene_knockout(model, gene, reference_fluxes)

                    growth_ratio = ko_growth / wt_growth if wt_growth > 1e-9 else 0.0
                    is_essential = growth_ratio < self.threshold

                    results.append(
                        {
                            "id": gene.id,
                            "name": gene.name if gene.name else "",
                            "type": "gene",
                            "wt_growth": wt_growth,
                            "ko_growth": ko_growth,
                            "growth_ratio": growth_ratio,
                            "is_essential": is_essential,
                        }
                    )

                    self.progress_update.emit(idx + 1, total)

            else:  # reactions
                # Filter to reactions that can be knocked out (not exchange, not essential constraints)
                targets = [r for r in model.reactions if not r.id.startswith("EX_")]
                total = len(targets)
                self.progress_update.emit(0, total)

                for idx, reaction in enumerate(targets):
                    if self._cancel_requested:
                        return

                    ko_growth = self._analyze_reaction_knockout(model, reaction, reference_fluxes)

                    growth_ratio = ko_growth / wt_growth if wt_growth > 1e-9 else 0.0
                    is_essential = growth_ratio < self.threshold

                    results.append(
                        {
                            "id": reaction.id,
                            "name": reaction.name if reaction.name else "",
                            "type": "reaction",
                            "wt_growth": wt_growth,
                            "ko_growth": ko_growth,
                            "growth_ratio": growth_ratio,
                            "is_essential": is_essential,
                        }
                    )

                    self.progress_update.emit(idx + 1, total)

            # Sort by growth ratio (essential first)
            results.sort(key=lambda x: x["growth_ratio"])

            n_essential = sum(1 for r in results if r["is_essential"])

            self.result_ready.emit(
                {
                    "results": results,
                    "wt_growth": wt_growth,
                    "n_essential": n_essential,
                    "n_total": len(results),
                    "threshold": self.threshold,
                    "analysis_type": self.analysis_type,
                    "target_type": self.target_type,
                }
            )

        except Exception as e:
            self.error_occurred.emit(f"Analysis failed: {str(e)}")

    def _analyze_gene_knockout(self, model, gene, reference_fluxes) -> float:
        """Analyze a single gene knockout."""
        try:
            with model:
                # Knock out the gene
                gene.knock_out()

                # Run MOMA or ROOM
                if self.analysis_type == "moma":
                    solution = linear_moma(model, reference_fluxes)
                else:
                    solution = room(model, reference_fluxes, self.room_delta, self.room_epsilon)

                if solution.status == "optimal":
                    return solution.objective_value
                else:
                    return 0.0
        except Exception:
            return 0.0

    def _analyze_reaction_knockout(self, model, reaction, reference_fluxes) -> float:
        """Analyze a single reaction knockout."""
        try:
            with model:
                # Knock out the reaction
                original_bounds = reaction.bounds
                reaction.bounds = (0, 0)

                # Run MOMA or ROOM
                if self.analysis_type == "moma":
                    solution = linear_moma(model, reference_fluxes)
                else:
                    solution = room(model, reference_fluxes, self.room_delta, self.room_epsilon)

                reaction.bounds = original_bounds

                if solution.status == "optimal":
                    return solution.objective_value
                else:
                    return 0.0
        except Exception:
            return 0.0


class BatchMomaRoomDialog(QDialog):
    """Dialog for batch MOMA/ROOM analysis."""

    def __init__(self, appdata: AppData, central_widget=None):
        super().__init__()
        self.setWindowTitle("Batch MOMA/ROOM Analysis")
        self.setMinimumSize(1000, 700)

        self.appdata = appdata
        self.central_widget = central_widget
        self.worker_thread: BatchMomaRoomWorkerThread | None = None
        self.last_results: dict | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        main_layout = QVBoxLayout()

        # Description
        desc_label = QLabel(
            "Batch MOMA/ROOM Analysis performs knockout analysis using MOMA (Minimization of "
            "Metabolic Adjustment) or ROOM (Regulatory On/Off Minimization) for all genes or reactions.\n"
            "MOMA minimizes L1 distance to wild-type flux. ROOM minimizes the number of flux changes (requires MILP solver)."
        )
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)

        # Parameters group
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QVBoxLayout()

        # Analysis type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Analysis Type:"))

        self.type_group = QButtonGroup(self)
        self.moma_radio = QRadioButton("MOMA (Linear)")
        self.moma_radio.setChecked(True)
        self.moma_radio.setToolTip("Minimization of Metabolic Adjustment - minimizes L1 distance to wild-type")
        self.type_group.addButton(self.moma_radio, 0)
        type_layout.addWidget(self.moma_radio)

        self.room_radio = QRadioButton("ROOM (MILP)")
        self.room_radio.setToolTip(
            "Regulatory On/Off Minimization - minimizes number of flux changes (requires MILP solver)"
        )
        self.type_group.addButton(self.room_radio, 1)
        type_layout.addWidget(self.room_radio)

        # Check MILP solver availability
        has_milp, milp_msg = has_milp_solver()
        if not has_milp:
            self.room_radio.setEnabled(False)
            self.room_radio.setToolTip(f"ROOM requires MILP solver. {milp_msg}")

        type_layout.addStretch()
        params_layout.addLayout(type_layout)

        # Target type selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target:"))

        self.target_group = QButtonGroup(self)
        self.genes_radio = QRadioButton("Genes")
        self.genes_radio.setChecked(True)
        self.genes_radio.setToolTip("Analyze gene knockouts")
        self.target_group.addButton(self.genes_radio, 0)
        target_layout.addWidget(self.genes_radio)

        self.reactions_radio = QRadioButton("Reactions")
        self.reactions_radio.setToolTip("Analyze reaction knockouts (excluding exchange reactions)")
        self.target_group.addButton(self.reactions_radio, 1)
        target_layout.addWidget(self.reactions_radio)

        target_layout.addStretch()
        params_layout.addLayout(target_layout)

        # Threshold setting
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Essentiality threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0001, 1.0)
        self.threshold_spin.setValue(0.01)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setToolTip("Growth ratio below which target is considered essential (default: 0.01 = 1%)")
        threshold_layout.addWidget(self.threshold_spin)

        threshold_layout.addStretch()

        # Compute button
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.setStyleSheet("font-size: 14px; padding: 8px 16px; background-color: #4CAF50; color: white;")
        self.compute_btn.clicked.connect(self._run_analysis)
        threshold_layout.addWidget(self.compute_btn)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        threshold_layout.addWidget(self.cancel_btn)

        params_layout.addLayout(threshold_layout)
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        # Results splitter (table and plot side by side)
        splitter = QSplitter(Qt.Horizontal)

        # Results table
        table_widget = QWidget()
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(
            ["ID", "Name", "WT Growth", "KO Growth", "Growth Ratio", "Essential"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.results_table)

        table_widget.setLayout(table_layout)
        splitter.addWidget(table_widget)

        # Scatter plot
        if MATPLOTLIB_AVAILABLE:
            plot_widget = QWidget()
            plot_layout = QVBoxLayout()
            plot_layout.setContentsMargins(0, 0, 0, 0)

            self.figure = Figure(figsize=(5, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            plot_layout.addWidget(self.canvas)

            # Export plot button
            export_plot_btn = QPushButton("Export Plot...")
            export_plot_btn.clicked.connect(self._export_plot)
            plot_layout.addWidget(export_plot_btn)

            plot_widget.setLayout(plot_layout)
            splitter.addWidget(plot_widget)
        else:
            # Placeholder if matplotlib not available
            no_plot_label = QLabel("Install matplotlib for scatter plot visualization")
            no_plot_label.setAlignment(Qt.AlignCenter)
            splitter.addWidget(no_plot_label)

        splitter.setSizes([500, 400])
        main_layout.addWidget(splitter)

        # Summary and export panel
        summary_layout = QHBoxLayout()

        # Summary label
        self.summary_label = QLabel("No results yet. Click 'Compute' to start analysis.")
        summary_layout.addWidget(self.summary_label)

        summary_layout.addStretch()

        # Export buttons
        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.clicked.connect(self._export_csv)
        summary_layout.addWidget(export_csv_btn)

        export_xlsx_btn = QPushButton("Export XLSX")
        export_xlsx_btn.clicked.connect(self._export_xlsx)
        if not OPENPYXL_AVAILABLE:
            export_xlsx_btn.setEnabled(False)
            export_xlsx_btn.setToolTip("openpyxl not installed. Install with: pip install openpyxl")
        summary_layout.addWidget(export_xlsx_btn)

        main_layout.addLayout(summary_layout)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    @Slot()
    def _run_analysis(self):
        """Start the batch MOMA/ROOM analysis."""
        model = self.appdata.project.cobra_py_model

        target_type = "genes" if self.genes_radio.isChecked() else "reactions"

        if target_type == "genes" and len(model.genes) == 0:
            QMessageBox.warning(self, "No Genes", "The model has no gene annotations.")
            return

        if target_type == "reactions" and len(model.reactions) == 0:
            QMessageBox.warning(self, "No Reactions", "The model has no reactions.")
            return

        analysis_type = "moma" if self.moma_radio.isChecked() else "room"

        # Disable compute, enable cancel
        self.compute_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)

        total = len(model.genes) if target_type == "genes" else len(model.reactions)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Starting {analysis_type.upper()} analysis on {target_type}...")

        # Create a copy of the model for the worker thread
        model_copy = model.copy()

        # Start worker thread
        self.worker_thread = BatchMomaRoomWorkerThread(
            model_copy,
            self.appdata,
            analysis_type,
            target_type,
            self.threshold_spin.value(),
        )
        self.worker_thread.progress_update.connect(self._on_progress)
        self.worker_thread.result_ready.connect(self._on_results)
        self.worker_thread.error_occurred.connect(self._on_error)
        self.worker_thread.finished.connect(self._on_finished)
        self.worker_thread.start()

    @Slot()
    def _cancel_analysis(self):
        """Cancel the running analysis."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.request_cancel()
            self.status_label.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setValue(current)
        target_type = "genes" if self.genes_radio.isChecked() else "reactions"
        self.status_label.setText(f"Analyzing {target_type[:-1]} {current}/{total}...")

    @Slot(object)
    def _on_results(self, data: dict):
        """Handle analysis results."""
        self.last_results = data
        results = data["results"]

        # Populate table
        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            # ID
            self.results_table.setItem(row, 0, QTableWidgetItem(result["id"]))

            # Name
            self.results_table.setItem(row, 1, QTableWidgetItem(result["name"]))

            # WT Growth
            item = QTableWidgetItem(f"{result['wt_growth']:.6f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(row, 2, item)

            # KO Growth
            item = QTableWidgetItem(f"{result['ko_growth']:.6f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(row, 3, item)

            # Growth Ratio
            item = QTableWidgetItem(f"{result['growth_ratio'] * 100:.2f}%")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(row, 4, item)

            # Is Essential
            essential_text = "Yes" if result["is_essential"] else "No"
            item = QTableWidgetItem(essential_text)
            item.setTextAlignment(Qt.AlignCenter)
            if result["is_essential"]:
                item.setBackground(Qt.red)
                item.setForeground(Qt.white)
            self.results_table.setItem(row, 5, item)

        # Update scatter plot
        if MATPLOTLIB_AVAILABLE:
            self._update_plot(results, data["wt_growth"])

        # Update summary
        analysis_type = data["analysis_type"].upper()
        target_type = data["target_type"]
        self.summary_label.setText(
            f"{analysis_type}: Found {data['n_essential']} essential {target_type} out of {data['n_total']} total "
            f"(WT growth: {data['wt_growth']:.4f}, threshold: {data['threshold'] * 100:.2f}%)"
        )
        self.status_label.setText("Analysis complete.")

    def _update_plot(self, results: list, wt_growth: float):
        """Update the scatter plot with results."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Separate essential and non-essential
        essential_wt = []
        essential_ko = []
        non_essential_wt = []
        non_essential_ko = []

        for result in results:
            if result["is_essential"]:
                essential_wt.append(result["wt_growth"])
                essential_ko.append(result["ko_growth"])
            else:
                non_essential_wt.append(result["wt_growth"])
                non_essential_ko.append(result["ko_growth"])

        # Plot non-essential (green)
        if non_essential_wt:
            ax.scatter(non_essential_wt, non_essential_ko, c="green", alpha=0.6, label="Non-essential", s=30)

        # Plot essential (red)
        if essential_wt:
            ax.scatter(essential_wt, essential_ko, c="red", alpha=0.6, label="Essential", s=30)

        # Reference line (y = x)
        max_val = max(wt_growth, max(r["ko_growth"] for r in results) if results else 0) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="No change (y=x)")

        # Threshold line
        threshold_y = wt_growth * self.threshold_spin.value()
        ax.axhline(
            y=threshold_y,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"Threshold ({self.threshold_spin.value()*100:.0f}%)",
        )

        ax.set_xlabel("Wild-type Growth")
        ax.set_ylabel("Knockout Growth")
        ax.set_title("MOMA/ROOM Knockout Analysis")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

        self.canvas.draw()

    @Slot(str)
    def _on_error(self, error: str):
        """Handle analysis error."""
        QMessageBox.warning(self, "Analysis Error", error)
        self.status_label.setText(f"Error: {error}")

    @Slot()
    def _on_finished(self):
        """Handle worker thread completion."""
        self.compute_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    @Slot()
    def _export_csv(self):
        """Export results to CSV."""
        if not self.last_results:
            QMessageBox.warning(self, "No Results", "No results to export. Run analysis first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export to CSV", self.appdata.work_directory, "CSV files (*.csv)"
        )

        if not filename:
            return

        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name", "WT Growth", "KO Growth", "Growth Ratio", "Is Essential"])

                for result in self.last_results["results"]:
                    writer.writerow(
                        [
                            result["id"],
                            result["name"],
                            result["wt_growth"],
                            result["ko_growth"],
                            result["growth_ratio"],
                            "Yes" if result["is_essential"] else "No",
                        ]
                    )

            QMessageBox.information(self, "Exported", f"Results exported to {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {str(e)}")

    @Slot()
    def _export_xlsx(self):
        """Export results to XLSX."""
        if not self.last_results:
            QMessageBox.warning(self, "No Results", "No results to export. Run analysis first.")
            return

        if not OPENPYXL_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "openpyxl is required for XLSX export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export to XLSX", self.appdata.work_directory, "Excel files (*.xlsx)"
        )

        if not filename:
            return

        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "MOMA-ROOM Analysis"

            # Headers
            headers = ["ID", "Name", "WT Growth", "KO Growth", "Growth Ratio", "Is Essential"]
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.font = openpyxl.styles.Font(bold=True)

            # Data
            for row, result in enumerate(self.last_results["results"], 2):
                ws.cell(row=row, column=1, value=result["id"])
                ws.cell(row=row, column=2, value=result["name"])
                ws.cell(row=row, column=3, value=result["wt_growth"])
                ws.cell(row=row, column=4, value=result["ko_growth"])
                ws.cell(row=row, column=5, value=result["growth_ratio"])
                ws.cell(row=row, column=6, value="Yes" if result["is_essential"] else "No")

                # Highlight essential
                if result["is_essential"]:
                    for col in range(1, 7):
                        ws.cell(row=row, column=col).fill = openpyxl.styles.PatternFill(
                            start_color="FFCCCC", end_color="FFCCCC", fill_type="solid"
                        )

            # Add summary sheet
            summary_ws = wb.create_sheet("Summary")
            summary_ws.cell(row=1, column=1, value="Metric")
            summary_ws.cell(row=1, column=2, value="Value")
            summary_ws.cell(row=1, column=1).font = openpyxl.styles.Font(bold=True)
            summary_ws.cell(row=1, column=2).font = openpyxl.styles.Font(bold=True)

            summary_data = [
                ("Analysis Type", self.last_results["analysis_type"].upper()),
                ("Target Type", self.last_results["target_type"]),
                ("Wild-type Growth", self.last_results["wt_growth"]),
                ("Threshold", f"{self.last_results['threshold'] * 100:.2f}%"),
                ("Total", self.last_results["n_total"]),
                ("Essential", self.last_results["n_essential"]),
                ("Non-essential", self.last_results["n_total"] - self.last_results["n_essential"]),
            ]

            for row, (metric, value) in enumerate(summary_data, 2):
                summary_ws.cell(row=row, column=1, value=metric)
                summary_ws.cell(row=row, column=2, value=value)

            wb.save(filename)
            QMessageBox.information(self, "Exported", f"Results exported to {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {str(e)}")

    @Slot()
    def _export_plot(self):
        """Export the scatter plot to a file."""
        if not MATPLOTLIB_AVAILABLE or not self.last_results:
            QMessageBox.warning(self, "No Plot", "No plot to export. Run analysis first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            self.appdata.work_directory,
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)",
        )

        if not filename:
            return

        try:
            self.figure.savefig(filename, dpi=150, bbox_inches="tight")
            QMessageBox.information(self, "Exported", f"Plot exported to {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {str(e)}")
