"""Plot Customization Dialog for CNApy

Shared dialog that allows customizing plot properties (title, axis labels,
axis scale, axis limits) for dialogs using FigureCanvasQTAgg.
"""

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class PlotCustomizationDialog(QDialog):
    """Dialog for customizing matplotlib plot properties.

    Supports figures with 1 or 2 axes. Changes are applied directly to the
    canvas without re-running the underlying simulation.
    """

    def __init__(self, parent, figure, canvas):
        """Initialize the dialog.

        Args:
            parent: Parent widget
            figure: matplotlib Figure object
            canvas: FigureCanvasQTAgg object
        """
        super().__init__(parent)
        self.setWindowTitle("Customize Plot")
        self.setMinimumWidth(400)

        self.figure = figure
        self.canvas = canvas
        self.axes = figure.get_axes()

        # Capture state at open time (for Reset)
        self.original_state = self._capture_state()

        # axis_widgets[idx] holds the form widgets for each axis
        self.axis_widgets: dict = {}

        self._setup_ui()
        self._load_state(self.original_state)

    def _capture_state(self) -> list[dict]:
        """Capture the current state of all axes."""
        state = []
        for ax in self.axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            state.append(
                {
                    "title": ax.get_title(),
                    "xlabel": ax.get_xlabel(),
                    "ylabel": ax.get_ylabel(),
                    "xscale": ax.get_xscale(),
                    "yscale": ax.get_yscale(),
                    "xmin": f"{xlim[0]:.6g}",
                    "xmax": f"{xlim[1]:.6g}",
                    "ymin": f"{ylim[0]:.6g}",
                    "ymax": f"{ylim[1]:.6g}",
                }
            )
        return state

    def _build_axis_form(self, idx: int) -> QWidget:
        """Build a form widget for one axis.

        Args:
            idx: Index of the axis in self.axes

        Returns:
            QWidget containing the form layout
        """
        widget = QWidget()
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        title_edit = QLineEdit()
        form.addRow("Title:", title_edit)

        xlabel_edit = QLineEdit()
        form.addRow("X Label:", xlabel_edit)

        ylabel_edit = QLineEdit()
        form.addRow("Y Label:", ylabel_edit)

        xscale_combo = QComboBox()
        xscale_combo.addItems(["linear", "log", "symlog", "logit"])
        xscale_combo.setToolTip(
            "linear: standard scale\n"
            "log: log scale (positive values only)\n"
            "symlog: symmetric log (handles zero and negative values)\n"
            "logit: logit scale (values strictly between 0 and 1)"
        )
        form.addRow("X Scale:", xscale_combo)

        yscale_combo = QComboBox()
        yscale_combo.addItems(["linear", "log", "symlog", "logit"])
        yscale_combo.setToolTip(
            "linear: standard scale\n"
            "log: log scale (positive values only)\n"
            "symlog: symmetric log (handles zero and negative values)\n"
            "logit: logit scale (values strictly between 0 and 1)"
        )
        form.addRow("Y Scale:", yscale_combo)

        # X range row
        xrange_widget = QWidget()
        xrange_layout = QHBoxLayout()
        xrange_layout.setContentsMargins(0, 0, 0, 0)
        xmin_edit = QLineEdit()
        xmax_edit = QLineEdit()
        xrange_layout.addWidget(QLabel("Min:"))
        xrange_layout.addWidget(xmin_edit)
        xrange_layout.addWidget(QLabel("Max:"))
        xrange_layout.addWidget(xmax_edit)
        xrange_widget.setLayout(xrange_layout)
        form.addRow("X Range:", xrange_widget)

        # Y range row
        yrange_widget = QWidget()
        yrange_layout = QHBoxLayout()
        yrange_layout.setContentsMargins(0, 0, 0, 0)
        ymin_edit = QLineEdit()
        ymax_edit = QLineEdit()
        yrange_layout.addWidget(QLabel("Min:"))
        yrange_layout.addWidget(ymin_edit)
        yrange_layout.addWidget(QLabel("Max:"))
        yrange_layout.addWidget(ymax_edit)
        yrange_widget.setLayout(yrange_layout)
        form.addRow("Y Range:", yrange_widget)

        widget.setLayout(form)

        self.axis_widgets[idx] = {
            "title": title_edit,
            "xlabel": xlabel_edit,
            "ylabel": ylabel_edit,
            "xscale": xscale_combo,
            "yscale": yscale_combo,
            "xmin": xmin_edit,
            "xmax": xmax_edit,
            "ymin": ymin_edit,
            "ymax": ymax_edit,
        }

        return widget

    def _setup_ui(self):
        """Setup the dialog UI."""
        main_layout = QVBoxLayout()

        if not self.axes:
            main_layout.addWidget(QLabel("No axes found in figure."))
        elif len(self.axes) == 1:
            form_widget = self._build_axis_form(0)
            main_layout.addWidget(form_widget)
        else:
            tab_widget = QTabWidget()
            for i in range(len(self.axes)):
                form_widget = self._build_axis_form(i)
                tab_widget.addTab(form_widget, f"Plot {i + 1}")
            main_layout.addWidget(tab_widget)

        # Button row
        btn_layout = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset to the state when this dialog was opened")
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def _load_state(self, state: list[dict]):
        """Populate form fields from a state snapshot.

        Args:
            state: List of per-axis state dicts as returned by _capture_state()
        """
        for idx, ax_state in enumerate(state):
            if idx not in self.axis_widgets:
                continue
            widgets = self.axis_widgets[idx]

            widgets["title"].setText(ax_state["title"])
            widgets["xlabel"].setText(ax_state["xlabel"])
            widgets["ylabel"].setText(ax_state["ylabel"])

            xscale_idx = widgets["xscale"].findText(ax_state["xscale"])
            if xscale_idx >= 0:
                widgets["xscale"].setCurrentIndex(xscale_idx)
            else:
                # Rare scale (e.g. "asinh", "function") not in list — fall back to linear
                widgets["xscale"].setCurrentIndex(0)

            yscale_idx = widgets["yscale"].findText(ax_state["yscale"])
            if yscale_idx >= 0:
                widgets["yscale"].setCurrentIndex(yscale_idx)
            else:
                # Rare scale not in list — fall back to linear
                widgets["yscale"].setCurrentIndex(0)

            widgets["xmin"].setText(ax_state["xmin"])
            widgets["xmax"].setText(ax_state["xmax"])
            widgets["ymin"].setText(ax_state["ymin"])
            widgets["ymax"].setText(ax_state["ymax"])

    def _apply(self):
        """Apply form values to the figure axes and redraw.

        If canvas.draw() fails (e.g. logit scale with out-of-range data),
        all axes are rolled back to their pre-apply state and a warning is shown.
        """
        # Snapshot axes state before making any changes (for rollback)
        pre_apply_state = self._capture_state()

        for idx, ax in enumerate(self.axes):
            if idx not in self.axis_widgets:
                continue
            widgets = self.axis_widgets[idx]

            ax.set_title(widgets["title"].text())
            ax.set_xlabel(widgets["xlabel"].text())
            ax.set_ylabel(widgets["ylabel"].text())

            xscale = widgets["xscale"].currentText()
            try:
                ax.set_xscale(xscale)
            except Exception:
                pass

            yscale = widgets["yscale"].currentText()
            try:
                ax.set_yscale(yscale)
            except Exception:
                pass

            xmin_str = widgets["xmin"].text().strip()
            xmax_str = widgets["xmax"].text().strip()
            if xmin_str and xmax_str:
                try:
                    ax.set_xlim(float(xmin_str), float(xmax_str))
                except ValueError:
                    pass

            ymin_str = widgets["ymin"].text().strip()
            ymax_str = widgets["ymax"].text().strip()
            if ymin_str and ymax_str:
                try:
                    ax.set_ylim(float(ymin_str), float(ymax_str))
                except ValueError:
                    pass

        try:
            self.canvas.draw()
        except Exception as e:
            # Rendering failed — roll back axes to pre-apply state
            self._restore_axes(pre_apply_state)
            self.canvas.draw()
            # Reflect rolled-back state in the form fields
            self._load_state(pre_apply_state)
            QMessageBox.warning(
                self,
                "Scale Not Applicable",
                f"Could not apply the selected scale to this plot:\n\n{e}\n\n"
                "Common causes:\n"
                "• log / logit: requires all data values > 0 (logit also requires < 1)\n"
                "• symlog: usually safe, but check your data range\n\n"
                "Changes have been reverted.",
            )

    def _restore_axes(self, state: list[dict]):
        """Directly restore axes properties from a state snapshot (no canvas.draw).

        Used for rollback when canvas.draw() fails after applying changes.
        """
        for idx, ax in enumerate(self.axes):
            if idx >= len(state):
                break
            s = state[idx]
            ax.set_title(s["title"])
            ax.set_xlabel(s["xlabel"])
            ax.set_ylabel(s["ylabel"])
            try:
                ax.set_xscale(s["xscale"])
            except Exception:
                ax.set_xscale("linear")
            try:
                ax.set_yscale(s["yscale"])
            except Exception:
                ax.set_yscale("linear")
            try:
                ax.set_xlim(float(s["xmin"]), float(s["xmax"]))
            except (ValueError, TypeError):
                pass
            try:
                ax.set_ylim(float(s["ymin"]), float(s["ymax"]))
            except (ValueError, TypeError):
                pass

    def _reset(self):
        """Reset fields and figure to the state captured at dialog open."""
        self._load_state(self.original_state)
        self._apply()
