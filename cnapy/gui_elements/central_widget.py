"""The central widget"""

import json
import os
from enum import IntEnum

import cobra
import numpy
from qtpy.QtCore import QSignalBlocker, Qt, Signal, Slot
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cnapy.appdata import AppData, CnaMap, ModelItemType, parse_scenario
from cnapy.gui_elements.escher_map_view import EscherMapView
from cnapy.gui_elements.gene_list import GeneList
from cnapy.gui_elements.map_view import MapView
from cnapy.gui_elements.metabolite_list import MetaboliteList
from cnapy.gui_elements.mode_navigator import ModeNavigator
from cnapy.gui_elements.model_info import ModelInfo
from cnapy.gui_elements.reactions_list import ReactionList, ReactionListColumn
from cnapy.gui_elements.scenario_tab import ScenarioTab
from cnapy.utils import SignalThrottler


class ModelTabIndex(IntEnum):
    Reactions = 0
    Metabolites = 1
    Genes = 2
    Scenario = 3
    Model = 4


class CentralWidget(QWidget):
    """The PyNetAnalyzer central widget"""

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.appdata: AppData = parent.appdata
        self.map_counter = 0

        searchbar_layout = QHBoxLayout()
        self.searchbar = QLineEdit()
        self.searchbar.setPlaceholderText("Enter search term")
        self.searchbar.setClearButtonEnabled(True)
        searchbar_layout.addWidget(self.searchbar)
        searchbar_layout.addSpacing(1)
        self.search_annotations = QCheckBox("+Annotations")
        self.search_annotations.setChecked(False)
        searchbar_layout.addWidget(self.search_annotations)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        searchbar_layout.addWidget(line)
        searchbar_layout.addSpacing(10)
        self.model_item_history = QComboBox()
        self.model_item_history.setToolTip("Recently viewed model items")
        self.model_item_history.activated.connect(self.select_item_from_history)
        self.model_item_history.setMaxCount(30)
        self.model_item_history.setMinimumContentsLength(25)
        self.model_item_history.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        searchbar_layout.addWidget(self.model_item_history)
        model_item_history_clear = QPushButton("Clear")
        model_item_history_clear.setFixedWidth(model_item_history_clear.fontMetrics().horizontalAdvance("Clear") + 10)
        searchbar_layout.addWidget(model_item_history_clear)
        model_item_history_clear.clicked.connect(self.clear_model_item_history)

        self.throttler = SignalThrottler(300)
        self.searchbar.textChanged.connect(self.throttler.throttle)
        self.throttler.triggered.connect(self.update_selected)
        self.search_annotations.clicked.connect(self.update_selected)

        self.tabs = QTabWidget()
        self.reaction_list = ReactionList(self)
        self.metabolite_list = MetaboliteList(self)
        self.scenario_tab = ScenarioTab(self)
        self.gene_list = GeneList(self)
        self.model_info = ModelInfo(self.appdata)
        self.tabs.addTab(self.reaction_list, "Reactions")
        self.tabs.addTab(self.metabolite_list, "Metabolites")
        self.tabs.addTab(self.gene_list, "Genes")
        self.tabs.addTab(self.scenario_tab, "Scenario")
        self.tabs.addTab(self.model_info, "Model")

        self.map_tabs = QTabWidget()
        self.map_tabs.setTabsClosable(True)
        self.map_tabs.setMovable(True)
        # Set white background for empty map area
        self.map_tabs.setStyleSheet("QTabWidget::pane { background-color: white; }")

        # Map toolbar (add reaction, save map, load map)
        map_toolbar = QHBoxLayout()
        map_toolbar.setContentsMargins(0, 0, 0, 0)
        map_toolbar.setSpacing(6)

        self.add_reaction_btn = QPushButton("Add reaction")
        self.add_reaction_btn.setToolTip("Add a model reaction to the current map")
        self.add_reaction_btn.clicked.connect(self.add_reaction_to_current_map)
        map_toolbar.addWidget(self.add_reaction_btn)

        self.save_map_btn = QPushButton("Save map")
        self.save_map_btn.setToolTip("Save current map to file (*.cmap.json)")
        self.save_map_btn.clicked.connect(self.save_current_map_to_file)
        map_toolbar.addWidget(self.save_map_btn)

        self.load_map_btn = QPushButton("Load map")
        self.load_map_btn.setToolTip("Load a map file (*.cmap.json)")
        self.load_map_btn.clicked.connect(self.load_map_from_file)
        map_toolbar.addWidget(self.load_map_btn)

        map_toolbar.addStretch()

        map_container = QWidget()
        map_container_layout = QVBoxLayout()
        map_container_layout.setContentsMargins(0, 0, 0, 0)
        map_container_layout.setSpacing(2)
        map_container_layout.addLayout(map_toolbar)
        map_container_layout.addWidget(self.map_tabs)
        map_container.setLayout(map_container_layout)

        self.splitter = QSplitter()
        self.splitter2 = QSplitter()
        self.splitter2.addWidget(map_container)
        self.mode_navigator = ModeNavigator(self.appdata, self)
        self.mode_navigator.setVisible(False)  # Hidden by default until modes are loaded
        self.splitter2.addWidget(self.mode_navigator)
        self.splitter2.setOrientation(Qt.Vertical)
        self.splitter.addWidget(self.splitter2)
        self.splitter.addWidget(self.tabs)

        layout = QVBoxLayout()
        layout.addItem(searchbar_layout)
        layout.addWidget(self.splitter)
        self.setLayout(layout)
        margins = self.layout().contentsMargins()
        margins.setBottom(0)  # otherwise the distance to the status bar appears too large
        self.layout().setContentsMargins(margins)

        self.tabs.currentChanged.connect(self.tabs_changed)
        self.reaction_list.jumpToMap.connect(self.jump_to_map)
        self.reaction_list.jumpToMetabolite.connect(self.jump_to_metabolite)
        self.reaction_list.reactionChanged.connect(self.handle_changed_reaction)
        self.reaction_list.reactionDeleted.connect(self.handle_deleted_reaction)
        self.metabolite_list.metaboliteChanged.connect(self.handle_changed_metabolite)
        self.metabolite_list.jumpToReaction.connect(self.jump_to_reaction)
        self.metabolite_list.metabolite_mask.metaboliteChanged.connect(
            self.reaction_list.reaction_mask.update_reaction_string
        )
        self.metabolite_list.metabolite_mask.metaboliteDeleted.connect(
            self.reaction_list.reaction_mask.update_reaction_string
        )
        self.metabolite_list.metabolite_mask.metaboliteDeleted.connect(self.handle_changed_metabolite)
        self.metabolite_list.metabolite_mask.metaboliteDeleted.connect(self.remove_top_item_history_entry)
        self.gene_list.geneChanged.connect(self.handle_changed_gene)
        self.gene_list.jumpToReaction.connect(self.jump_to_reaction)
        self.gene_list.jumpToMetabolite.connect(self.jump_to_metabolite)
        self.model_info.globalObjectiveChanged.connect(self.handle_changed_global_objective)
        self.scenario_tab.objectiveSetupChanged.connect(self.handle_changed_objective_setup)
        self.scenario_tab.scenarioChanged.connect(self.parent.update_scenario_file_name)
        self.map_tabs.tabCloseRequested.connect(self.delete_map)
        self.mode_navigator.changedCurrentMode.connect(self.update_mode)
        self.mode_navigator.modeNavigatorClosed.connect(self.update)
        self.mode_navigator.reaction_participation_button.clicked.connect(self.reaction_participation)

        self.mode_normalization_reaction = ""

        self.update()

    def fit_mapview(self):
        if isinstance(self.map_tabs.currentWidget(), MapView):
            self.map_tabs.currentWidget().fit()

    def handle_changed_reaction(self, previous_id: str, reaction: cobra.Reaction):
        self.parent.unsaved_changes()
        reaction_has_box = False
        escher_map_present = False
        for mmap in self.appdata.project.maps:
            if previous_id in self.appdata.project.maps[mmap]["boxes"].keys():
                self.appdata.project.maps[mmap]["boxes"][reaction.id] = self.appdata.project.maps[mmap]["boxes"].pop(
                    previous_id
                )
                reaction_has_box = True
            if self.appdata.project.maps[mmap].get("view", "") == "escher":
                escher_map_present = True
        if reaction_has_box or escher_map_present:
            self.update_reaction_on_maps(previous_id, reaction.id, reaction_has_box, escher_map_present)
        if reaction.id != previous_id:
            self.appdata.project.reaction_ids.replace_entry(previous_id, reaction.id)
        self.update_item_in_history(previous_id, reaction.id, reaction.name, ModelItemType.Reaction)

    def handle_deleted_reaction(self, reaction: cobra.Reaction):
        self.appdata.project.cobra_py_model.remove_reactions([reaction], remove_orphans=True)
        self.appdata.project.scen_values.pop(reaction.id, None)
        self.appdata.project.scen_values.objective_coefficients.pop(reaction.id, None)
        self.remove_top_item_history_entry()

        self.parent.unsaved_changes()
        for mmap in self.appdata.project.maps:
            if reaction.id in self.appdata.project.maps[mmap]["boxes"].keys():
                self.appdata.project.maps[mmap]["boxes"].pop(reaction.id)
        self.delete_reaction_on_maps(reaction.id)
        self.appdata.project.update_reaction_id_lists()

        if self.appdata.auto_fba:
            self.parent.run_auto_analysis()

    @Slot(cobra.Metabolite, object, str)
    def handle_changed_metabolite(self, metabolite: cobra.Metabolite, affected_reactions, previous_id: str):
        self.parent.unsaved_changes()
        for reaction in affected_reactions:  # only updates CNApy maps
            self.update_reaction_on_maps(reaction.id, reaction.id)
        for idx in range(0, self.map_tabs.count()):
            m = self.map_tabs.widget(idx)
            if isinstance(m, EscherMapView):
                m.change_metabolite_id(previous_id, metabolite.id)
        self.update_item_in_history(previous_id, metabolite.id, metabolite.name, ModelItemType.Metabolite)

    def handle_changed_gene(self, previous_id: str, gene: cobra.Gene):
        self.parent.unsaved_changes()
        # TODO update only relevant reaction boxes on maps
        self.update_maps()
        self.update_item_in_history(previous_id, gene.id, gene.name, ModelItemType.Gene)

    @Slot()
    def handle_changed_global_objective(self):
        self.parent.unsaved_changes()
        if self.appdata.auto_fba and not self.appdata.project.scen_values.use_scenario_objective:
            self.parent.run_auto_analysis()

    @Slot()
    def handle_changed_objective_setup(self):
        if self.appdata.auto_fba:
            self.parent.run_auto_analysis()

    def switch_to_reaction(self, reaction: str):
        with QSignalBlocker(self.tabs):  # set_current_item will update
            self.tabs.setCurrentIndex(ModelTabIndex.Reactions)
        if self.tabs.width() == 0:
            (left, _) = self.splitter.sizes()
            self.splitter.setSizes([left, 1])
        self.reaction_list.set_current_item(reaction)

    def minimize_reaction(self, reaction: str):
        self.parent.fba_optimize_reaction(reaction, mmin=True)

    def maximize_reaction(self, reaction: str):
        self.parent.fba_optimize_reaction(reaction, mmin=False)

    @Slot(str)
    def set_scen_value(self, reaction: str):
        self.appdata.set_comp_value_as_scen_value(reaction)
        self.update()

    def update_reaction_value(self, reaction: str, value: str, update_reaction_list=True):
        if value == "":
            self.appdata.scen_values_pop(reaction)
            self.appdata.project.comp_values.pop(reaction, None)
        else:
            self.appdata.scen_values_set(reaction, parse_scenario(value))
        if update_reaction_list:
            self.reaction_list.update(rebuild=False)

    def update_reaction_maps(self, _reaction: str):
        self.parent.unsaved_changes()
        self.reaction_list.reaction_mask.update_state()

    def handle_mapChanged(self, _reaction: str):
        self.parent.unsaved_changes()

    def tabs_changed(self, idx):
        if idx == ModelTabIndex.Reactions:
            self.reaction_list.update()
        elif idx == ModelTabIndex.Metabolites:
            self.metabolite_list.update()
        elif idx == ModelTabIndex.Genes:
            self.gene_list.update()
        elif idx == ModelTabIndex.Scenario:
            self.scenario_tab.update()
        elif idx == ModelTabIndex.Model:
            self.model_info.update()

    def connect_map_view_signals(self, mmap: MapView):
        mmap.switchToReactionMask.connect(self.switch_to_reaction)
        mmap.minimizeReaction.connect(self.minimize_reaction)
        mmap.maximizeReaction.connect(self.maximize_reaction)
        mmap.setScenValue.connect(self.set_scen_value)
        mmap.reactionValueChanged.connect(self.update_reaction_value)
        mmap.reactionRemoved.connect(self.update_reaction_maps)
        mmap.reactionAdded.connect(self.update_reaction_maps)
        mmap.mapChanged.connect(self.handle_mapChanged)

    def connect_escher_map_view_signals(self, mmap: EscherMapView):
        mmap.cnapy_bridge.reactionValueChanged.connect(self.update_reaction_value)
        mmap.cnapy_bridge.switchToReactionMask.connect(self.switch_to_reaction)
        mmap.cnapy_bridge.jumpToMetabolite.connect(self.jump_to_metabolite)

    @Slot()
    def add_map(self, base_name="Map", escher=False):
        if base_name == "Map" or (base_name in self.appdata.project.maps.keys()):
            while True:
                name = base_name + " " + str(self.map_counter)
                if name not in self.appdata.project.maps.keys():
                    break
                self.map_counter += 1
        else:
            name = base_name
        m = CnaMap(name)
        self.appdata.project.maps[name] = m
        if escher:
            mmap: EscherMapView = EscherMapView(self, name)
            self.connect_escher_map_view_signals(mmap)
            self.appdata.project.maps[name][EscherMapView] = mmap
            self.appdata.project.maps[name]["view"] = "escher"
            self.appdata.project.maps[name]["pos"] = '{"x":0,"y":0}'
            self.appdata.project.maps[name]["zoom"] = "1"
            # mmap.loadFinished.connect(self.finish_add_escher_map)
            # mmap.cnapy_bridge.reactionValueChanged.connect(self.update_reaction_value) # connection is not made?!
            # self.appdata.qapp.processEvents() # does not help
            idx = self.map_tabs.addTab(mmap, m["name"])
            # Try to also create a CNApy map from existing Escher JSON so CNApy view is available by default
            self.maybe_create_cnapy_from_escher(name)
        else:
            mmap = MapView(self.appdata, self, name)
            self.connect_map_view_signals(mmap)
            idx = self.map_tabs.addTab(mmap, m["name"])
            self.update_maps()  # only update mmap?
        self.map_tabs.setCurrentIndex(idx)
        self.parent.unsaved_changes()

        return name, idx

    def delete_map(self, idx: int):
        name = self.map_tabs.tabText(idx)
        diag = ConfirmMapDeleteDialog(self, idx, name)
        diag.exec()

    def update_selected(self):
        string = self.searchbar.text()

        idx = self.tabs.currentIndex()
        map_idx = self.map_tabs.currentIndex()

        with_annotations = self.search_annotations.isChecked() and self.search_annotations.isEnabled()
        QApplication.setOverrideCursor(Qt.BusyCursor)
        QApplication.processEvents()  # to put the change above into effect
        if idx == ModelTabIndex.Reactions:
            found_ids = self.reaction_list.update_selected(string, with_annotations)
            found_reaction_ids = found_ids
        elif idx == ModelTabIndex.Metabolites:
            found_ids = self.metabolite_list.update_selected(string, with_annotations)
            if map_idx >= 0:
                found_reaction_ids = []
                for found_id in found_ids:
                    metabolite = self.appdata.project.cobra_py_model.metabolites.get_by_id(found_id)
                    found_reaction_ids += [x.id for x in metabolite.reactions]
            else:
                found_reaction_ids = found_ids
        elif idx == ModelTabIndex.Genes:
            found_ids = self.gene_list.update_selected(string, with_annotations)
            if map_idx >= 0:
                found_reaction_ids = []
                for found_id in found_ids:
                    gene = self.appdata.project.cobra_py_model.genes.get_by_id(found_id)
                    found_reaction_ids += [x.id for x in gene.reactions]
            else:
                found_reaction_ids = found_ids
        else:
            if len(string) == 0:
                # needed to reset selection on map
                found_reaction_ids = self.appdata.project.cobra_py_model.reactions.list_attr("id")
            else:
                QApplication.restoreOverrideCursor()
                return

        if map_idx >= 0:
            m = self.map_tabs.widget(map_idx)
            if isinstance(m, EscherMapView):
                m.update_selected(string)
            else:
                m.update_selected(found_reaction_ids)
        QApplication.restoreOverrideCursor()

    def update_mode(self):
        if self.mode_navigator.mode_type <= 1:
            if len(self.appdata.project.modes) > self.mode_navigator.current:
                values = self.appdata.project.modes[self.mode_navigator.current]
                if self.mode_navigator.mode_type == 0 and not self.appdata.project.modes.is_integer_vector_rounded(
                    self.mode_navigator.current, self.appdata.rounding
                ):
                    # normalize non-integer EFM for better display
                    mean = sum(abs(v) for v in values.values()) / len(values)
                    for r, v in values.items():
                        values[r] = v / mean
                if self.mode_normalization_reaction != "":
                    if self.mode_normalization_reaction in values.keys():
                        normalization_value = values[self.mode_normalization_reaction]
                        if normalization_value != 0.0:
                            for r, v in values.items():
                                values[r] = v / normalization_value

                # set values
                self.appdata.project.comp_values.clear()
                self.parent.clear_status_bar()
                for i in values:
                    if self.mode_navigator.mode_type == 1:
                        if values[i] < 0:
                            values[i] = 0.0  # display KOs as zero flux
                    self.appdata.project.comp_values[i] = (values[i], values[i])
                self.appdata.project.comp_values_type = 0

            self.appdata.modes_coloring = True
            self.update()
            self.appdata.modes_coloring = False

        elif self.mode_navigator.mode_type == 2:
            if len(self.appdata.project.modes) > self.mode_navigator.current:
                # clear previous coloring
                self.appdata.project.comp_values.clear()
                self.parent.clear_status_bar()
                self.appdata.project.comp_values_type = 0
                # Set values
                bnd_dict = self.appdata.project.modes[self.mode_navigator.current]
                for k, v in bnd_dict.items():
                    if numpy.any(numpy.isnan(v)):
                        self.appdata.project.comp_values[k] = (0, 0)
                    else:
                        mod_bnds = self.appdata.project.cobra_py_model.reactions.get_by_id(k).bounds
                        self.appdata.project.comp_values[k] = (
                            numpy.max((v[0], mod_bnds[0])),
                            numpy.min((v[1], mod_bnds[1])),
                        )
                self.appdata.modes_coloring = True
                self.update()
                self.appdata.modes_coloring = False
                idx = self.appdata.window.centralWidget().tabs.currentIndex()
                if idx == ModelTabIndex.Reactions and self.appdata.project.comp_values_type == 0:
                    view = self.appdata.window.centralWidget().reaction_list
                    view.reaction_list.blockSignals(True)  # block itemChanged while recoloring
                    root = view.reaction_list.invisibleRootItem()
                    child_count = root.childCount()
                    for i in range(child_count):
                        item = root.child(i)
                        if item.text(0) in bnd_dict:
                            v = bnd_dict[item.text(0)]
                            if numpy.any(numpy.isnan(v)):
                                item.setBackground(ReactionListColumn.Flux, self.appdata.special_color_1)
                            elif (v[0] < 0 and v[1] >= 0) or (v[0] <= 0 and v[1] > 0):
                                item.setBackground(ReactionListColumn.Flux, self.appdata.special_color_2)
                            elif v[0] == 0.0 and v[1] == 0.0:
                                item.setBackground(ReactionListColumn.Flux, QColor.fromRgb(255, 0, 0))
                            elif (v[0] < 0 and v[1] < 0) or (v[0] > 0 and v[1] > 0):
                                item.setBackground(ReactionListColumn.Flux, self.appdata.special_color_1)
                        else:
                            item.setBackground(ReactionListColumn.Flux, QColor.fromRgb(255, 255, 255))
                    view.reaction_list.blockSignals(False)
                idx = self.appdata.window.centralWidget().map_tabs.currentIndex()
                if idx < 0:
                    return
                name = self.appdata.window.centralWidget().map_tabs.tabText(idx)
                view = self.appdata.window.centralWidget().map_tabs.widget(idx)
                for key in self.appdata.project.maps[name]["boxes"]:
                    if key in bnd_dict:
                        v = bnd_dict[key]
                        if numpy.any(numpy.isnan(v)):
                            view.reaction_boxes[key].set_color(self.appdata.special_color_1)
                        elif (v[0] < 0 and v[1] >= 0) or (v[0] <= 0 and v[1] > 0):
                            view.reaction_boxes[key].set_color(self.appdata.special_color_2)
                        elif v[0] == 0.0 and v[1] == 0.0:
                            view.reaction_boxes[key].set_color(QColor.fromRgb(255, 0, 0))
                        elif (v[0] < 0 and v[1] < 0) or (v[0] > 0 and v[1] > 0):
                            view.reaction_boxes[key].set_color(self.appdata.special_color_1)
                    else:
                        view.reaction_boxes[key].set_color(QColor.fromRgb(255, 255, 255))
                if self.appdata.window.sd_sols and self.appdata.window.sd_sols.__weakref__:  # if dialog exists
                    self.mode_navigator.current
                    for i in range(self.appdata.window.sd_sols.sd_table.rowCount()):
                        if (
                            self.mode_navigator.current
                            == int(self.appdata.window.sd_sols.sd_table.item(i, 0).text()) - 1
                        ):
                            self.appdata.window.sd_sols.sd_table.item(i, 0).setBackground(QBrush(QColor(230, 230, 230)))
                            self.appdata.window.sd_sols.sd_table.item(i, 1).setBackground(QBrush(QColor(230, 230, 230)))
                            if self.appdata.window.sd_sols.sd_table.columnCount() == 3:
                                self.appdata.window.sd_sols.sd_table.item(i, 2).setBackground(
                                    QBrush(QColor(230, 230, 230))
                                )
                        else:
                            self.appdata.window.sd_sols.sd_table.item(i, 0).setBackground(QBrush(QColor(255, 255, 255)))
                            self.appdata.window.sd_sols.sd_table.item(i, 1).setBackground(QBrush(QColor(255, 255, 255)))
                            if self.appdata.window.sd_sols.sd_table.columnCount() == 3:
                                self.appdata.window.sd_sols.sd_table.item(i, 2).setBackground(
                                    QBrush(QColor(255, 255, 255))
                                )
        self.mode_navigator.current_flux_values = self.appdata.project.comp_values.copy()

    def reaction_participation(self):
        self.appdata.project.comp_values.clear()
        self.parent.clear_status_bar()
        if self.appdata.window.centralWidget().mode_navigator.mode_type <= 1:
            relative_participation = (
                numpy.sum(self.appdata.project.modes.fv_mat[self.mode_navigator.selection, :] != 0, axis=0)
                / self.mode_navigator.num_selected
            )
            if isinstance(
                relative_participation, numpy.matrix
            ):  # numpy.sum returns a matrix with one row when fv_mat is scipy.sparse
                relative_participation = relative_participation.A1  # flatten into 1D array
            self.appdata.project.comp_values = {
                r: (relative_participation[i], relative_participation[i])
                for i, r in enumerate(self.appdata.project.modes.reac_id)
            }
        elif self.appdata.window.centralWidget().mode_navigator.mode_type == 2:
            reacs = self.appdata.project.cobra_py_model.reactions.list_attr("id")
            abund = [0 for _ in reacs]
            for i, r in enumerate(reacs):
                for s in [self.appdata.project.modes[l] for l, t in enumerate(self.mode_navigator.selection) if t]:
                    if r in s:
                        if not numpy.any(numpy.isnan(s[r])) or numpy.all(s[r] == 0):
                            abund[i] += 1
            relative_participation = [a / self.mode_navigator.num_selected for a in abund]
            self.appdata.project.comp_values = {r: (p, p) for r, p in zip(reacs, relative_participation, strict=False)}
        if isinstance(
            relative_participation, numpy.matrix
        ):  # numpy.sum returns a matrix with one row when fv_mat is scipy.sparse
            relative_participation = relative_participation.A1  # flatten into 1D array
        self.appdata.project.comp_values_type = 0
        self.update()
        self.parent.set_heaton()

    def update(self, rebuild_all_tabs=False):
        # use rebuild_all_tabs=True to rebuild all tabs when the model changes
        if len(self.appdata.project.modes) == 0:
            self.mode_navigator.hide()
            self.mode_navigator.current = 0
        else:
            self.mode_navigator.show()
            self.mode_navigator.update()

        if rebuild_all_tabs:
            self.reaction_list.update(rebuild=True)
            self.metabolite_list.update()
            self.gene_list.update()
            self.scenario_tab.recreate_scenario_items_needed = True
            self.scenario_tab.update()
            self.model_info.update()
        else:
            idx = self.tabs.currentIndex()
            if idx == ModelTabIndex.Reactions:
                self.reaction_list.update()
            elif idx == ModelTabIndex.Metabolites:
                self.metabolite_list.update()
            elif idx == ModelTabIndex.Genes:
                self.gene_list.update()
            elif idx == ModelTabIndex.Scenario:
                self.scenario_tab.update()
            elif idx == ModelTabIndex.Model:
                self.model_info.update()

        idx = self.map_tabs.currentIndex()
        if idx >= 0:
            m = self.map_tabs.widget(idx)
            m.update()

        self.__recolor_map()

    def update_map(self, idx):
        m = self.map_tabs.widget(idx)
        if m is not None:
            m.update()
        self.__recolor_map()

    def update_reaction_on_maps(
        self,
        old_reaction_id: str,
        new_reaction_id: str,
        update_cnapy_maps: bool = True,
        update_escher_maps: bool = False,
    ):
        for idx in range(0, self.map_tabs.count()):
            m = self.map_tabs.widget(idx)
            if update_cnapy_maps and isinstance(m, MapView):
                m.update_reaction(old_reaction_id, new_reaction_id)
            elif update_escher_maps and isinstance(m, EscherMapView):
                if old_reaction_id != new_reaction_id:
                    m.change_reaction_id(old_reaction_id, new_reaction_id)
                else:
                    m.update_reaction_stoichiometry(old_reaction_id)

    def delete_reaction_on_maps(self, reation_id: str):
        for idx in range(0, self.map_tabs.count()):
            m = self.map_tabs.widget(idx)
            if isinstance(m, MapView):
                m.delete_box(reation_id)
            else:
                m.delete_reaction(reation_id)

    def update_maps(self):
        for idx in range(0, self.map_tabs.count()):
            m = self.map_tabs.widget(idx)
            m.update()
        self.__recolor_map()

    def jump_to_map(self, identifier: str, reaction: str):
        for idx in range(0, self.map_tabs.count()):
            name = self.map_tabs.tabText(idx)
            if name == identifier:
                m = self.map_tabs.widget(idx)
                self.map_tabs.setCurrentIndex(idx)

                m.update()
                m.focus_reaction(reaction)
                self.__recolor_map()
                m.highlight_reaction(reaction)
                break

    def reaction_selected(self, reac_id: str):
        for idx in range(0, self.map_tabs.count()):
            self.map_tabs.widget(idx).select_single_reaction(reac_id)

    @Slot()
    def add_reaction_to_current_map(self):
        idx = self.map_tabs.currentIndex()
        if idx < 0:
            return
        map_widget = self.map_tabs.widget(idx)
        if not isinstance(map_widget, MapView):
            if isinstance(map_widget, EscherMapView):
                dialog = MapReactionAddDialog(self, self.appdata)
                if dialog.exec() != QDialog.Accepted:
                    return
                reac_id = dialog.reaction_id()
                if not reac_id:
                    return
                if reac_id not in self.appdata.project.cobra_py_model.reactions:
                    QMessageBox.warning(self, "Add reaction", f"Reaction not in model: {reac_id}")
                    return
                # ensure editing mode then display Escher search bar for the reaction
                map_widget.enable_editing(True)
                map_widget.cnapy_bridge.displaySearchBarFor.emit(reac_id)
                map_widget.setFocus()
            else:
                QMessageBox.information(self, "Map", "You can add reactions only on CNApy or Escher maps.")
            return

        dialog = MapReactionAddDialog(self, self.appdata)
        if dialog.exec() != QDialog.Accepted:
            return
        reac_id = dialog.reaction_id()
        if reac_id is None or len(reac_id.strip()) == 0:
            return
        if reac_id not in self.appdata.project.cobra_py_model.reactions:
            QMessageBox.warning(self, "Add reaction", f"Reaction not in model: {reac_id}")
            return

        name = self.map_tabs.tabText(idx)
        boxes = self.appdata.project.maps[name]["boxes"]
        if reac_id in boxes:
            map_widget.select_single_reaction(reac_id)
            map_widget.focus_reaction(reac_id)
            return

        scene_pos = map_widget.mapToScene(map_widget.viewport().rect().center())
        boxes[reac_id] = (scene_pos.x(), scene_pos.y())
        map_widget.rebuild_scene()
        map_widget.update()
        map_widget.select_single_reaction(reac_id)
        map_widget.focus_reaction(reac_id)
        self.reaction_list.reaction_mask.update_state()
        self.parent.unsaved_changes()

    @Slot()
    def save_current_map_to_file(self):
        idx = self.map_tabs.currentIndex()
        if idx < 0:
            return
        name = self.map_tabs.tabText(idx)
        data = self.appdata.project.maps.get(name, None)
        if data is None:
            return
        filename = QFileDialog.getSaveFileName(
            self, "Save map", self.appdata.work_directory, "CNApy Map (*.cmap.json)"
        )[0]
        if not filename:
            return
        if not filename.endswith(".cmap.json"):
            filename += ".cmap.json"
        try:
            with open(filename, "w") as fp:
                json.dump(data, fp, skipkeys=True)
        except OSError as exc:
            QMessageBox.critical(self, "Save map failed", str(exc))
            return
        QMessageBox.information(self, "Save map", f"Saved: {filename}")

    @Slot()
    def load_map_from_file(self):
        filename = QFileDialog.getOpenFileName(
            self, "Load map", self.appdata.work_directory, "CNApy Map (*.cmap.json)"
        )[0]
        if not filename or not os.path.exists(filename):
            return
        try:
            with open(filename) as fp:
                data = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load map failed", str(exc))
            return

        if not isinstance(data, dict) or "boxes" not in data:
            QMessageBox.critical(self, "Load map failed", "Invalid map data.")
            return

        base_name = data.get("name", os.path.basename(filename).replace(".cmap.json", ""))
        name = base_name
        counter = 1
        while name in self.appdata.project.maps:
            name = f"{base_name}_{counter}"
            counter += 1
        data["name"] = name
        data.setdefault("view", "cnapy")
        data.setdefault("bg-size", 1)
        data.setdefault("box-size", 1)
        data.setdefault("zoom", 0)
        data.setdefault("pos", (0, 0))
        data.setdefault("escher_map_data", "")

        self.appdata.project.maps[name] = data
        mmap = MapView(self.appdata, self, name)
        self.connect_map_view_signals(mmap)
        idx = self.map_tabs.addTab(mmap, name)
        self.map_tabs.setCurrentIndex(idx)
        mmap.update()
        self.reaction_list.reaction_mask.update_state()
        self.parent.unsaved_changes()

    @Slot()
    def load_cnapy_map_from_escher_json(self):
        """Create a CNApy map from Escher JSON and/or image files.

        Supports two modes:
        1. JSON + image: Full map with reaction positions from JSON
        2. Image only: Empty map with image background, reactions can be added manually
        """
        # Create dialog for file selection
        dialog = EscherMapFileDialog(self, self.appdata.work_directory)
        if dialog.exec() != QDialog.Accepted:
            return

        json_filename = dialog.json_file
        image_filename = dialog.image_file

        if not image_filename or not os.path.exists(image_filename):
            QMessageBox.warning(
                self, "Image file required", "Image file (PNG or SVG) is required to display the map background."
            )
            return

        # Handle PNG/SVG-only mode (no JSON)
        if not json_filename or not os.path.exists(json_filename):
            return self._create_map_from_image_only(image_filename)

        # Full mode: JSON + Image
        # Read JSON file
        try:
            with open(json_filename) as fp:
                escher_json = fp.read()
        except OSError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        data = self._parse_escher_json(escher_json)
        if data is None:
            QMessageBox.warning(self, "Invalid file", "Could not parse Escher JSON.")
            return

        # Get map name from JSON if available
        map_name = None
        try:
            parsed_json = json.loads(escher_json)
            if isinstance(parsed_json, list) and len(parsed_json) > 0:
                if isinstance(parsed_json[0], dict) and "map_name" in parsed_json[0]:
                    map_name = parsed_json[0]["map_name"]
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            pass

        if isinstance(data, dict):
            if "map_name" in data:
                map_name = data["map_name"]
            elif isinstance(data.get("map"), dict) and "map_name" in data["map"]:
                map_name = data["map"]["map_name"]

        base_name = map_name if map_name else "CNApy from Escher"
        name = base_name
        suffix = 1
        while name in self.appdata.project.maps:
            name = f"{base_name} {suffix}"
            suffix += 1

        # Parse reactions from JSON
        reactions = data.get("reactions", {})
        if not isinstance(reactions, dict) or len(reactions) == 0:
            # try alternative structure if available
            records = data.get("reaction_data", [])
            if isinstance(records, dict):
                records = records.values()
            if isinstance(records, list):
                for r in records:
                    if isinstance(r, dict) and "bigg_id" in r and "label_x" in r and "label_y" in r:
                        reactions[r["bigg_id"]] = r
            if len(reactions) == 0:
                QMessageBox.warning(self, "No reactions", "No reaction entries found in JSON.")
                return

        # Get canvas offset from JSON (for PNG files, we use JSON canvas info)
        offset_x, offset_y = 0, 0
        canvas_info = data.get("canvas", {})
        if isinstance(canvas_info, dict):
            offset_x = canvas_info.get("x", 0)
            offset_y = canvas_info.get("y", 0)

        # Create boxes from JSON reactions
        boxes = {}
        missing_in_model = []
        reactions_with_labels = 0
        for _, r in reactions.items():
            rid = r.get("bigg_id")
            lx = r.get("label_x")
            ly = r.get("label_y")
            if rid and lx is not None and ly is not None:
                reactions_with_labels += 1
                # Adjust coordinates by canvas offset
                adjusted_x = lx - offset_x
                adjusted_y = ly - offset_y
                # Add all reactions with label positions to boxes, even if not in model
                boxes[rid] = (adjusted_x, adjusted_y)
                if rid not in self.appdata.project.cobra_py_model.reactions:
                    missing_in_model.append(rid)

        if len(boxes) == 0:
            QMessageBox.warning(self, "No positions", "No reaction label positions found in the JSON file.")
            return

        if missing_in_model:
            msg = (
                f"Successfully loaded {len(boxes)} reaction positions.\n\n"
                f"Note: {len(missing_in_model)} reactions from the JSON file are not in the current model "
                f"but were added to the map anyway."
            )
            if len(missing_in_model) <= 10:
                msg += f"\nMissing reactions: {', '.join(missing_in_model)}"
            else:
                msg += f"\nFirst few missing reactions: {', '.join(missing_in_model[:10])}..."
            QMessageBox.information(self, "Map loaded", msg)

        # Create CNApy map view (not EscherMapView)
        cmap = CnaMap(name)
        cmap["boxes"] = boxes
        cmap["background"] = image_filename
        cmap["escher_map_data"] = escher_json  # Store JSON for reference
        self.appdata.project.maps[name] = cmap

        mmap = MapView(self.appdata, self, name)
        self.connect_map_view_signals(mmap)
        idx = self.map_tabs.addTab(mmap, name)
        self.map_tabs.setCurrentIndex(idx)
        mmap.update()
        self.reaction_list.reaction_mask.update_state()
        self.parent.unsaved_changes()

    def _create_map_from_image_only(self, image_filename: str):
        """Create a CNApy map with only an image background (no JSON positions).

        User can then manually add reaction boxes to this map.

        Parameters:
        -----------
        image_filename : str
            Path to the PNG or SVG image file
        """
        # Generate map name from filename
        base_name = os.path.splitext(os.path.basename(image_filename))[0]
        name = base_name
        suffix = 1
        while name in self.appdata.project.maps:
            name = f"{base_name} {suffix}"
            suffix += 1

        # Create empty CNApy map with image background
        cmap = CnaMap(name)
        cmap["boxes"] = {}  # No boxes initially
        cmap["background"] = image_filename
        cmap["escher_map_data"] = ""  # No JSON data
        self.appdata.project.maps[name] = cmap

        mmap = MapView(self.appdata, self, name)
        self.connect_map_view_signals(mmap)
        idx = self.map_tabs.addTab(mmap, name)
        self.map_tabs.setCurrentIndex(idx)
        mmap.update()
        self.reaction_list.reaction_mask.update_state()
        self.parent.unsaved_changes()

        QMessageBox.information(
            self,
            "Map created",
            f"Map '{name}' created with image background.\n\n"
            "You can now add reaction boxes manually:\n"
            "• Drag reactions from the Reactions list onto the map\n"
            "• Or use the 'Add reaction' button in the map toolbar\n\n"
            "Click on a reaction box to edit its flux value.",
        )

    def maybe_create_cnapy_from_escher(self, escher_name: str):
        """If the Escher map has JSON data, create a CNApy map mirroring reaction positions."""
        escher_entry = self.appdata.project.maps.get(escher_name, {})
        escher_json = escher_entry.get("escher_map_data", "")
        data = self._parse_escher_json(escher_json)
        if data is None:
            return

        reactions = data.get("reactions", {})
        if not isinstance(reactions, dict) or len(reactions) == 0:
            return

        boxes = {}
        for _, r in reactions.items():
            rid = r.get("bigg_id")
            lx = r.get("label_x")
            ly = r.get("label_y")
            if rid and lx is not None and ly is not None:
                boxes[rid] = (lx, ly)

        if len(boxes) == 0:
            return

        base_name = escher_name + " (CNApy)"
        name = base_name
        suffix = 1
        while name in self.appdata.project.maps:
            name = f"{base_name} {suffix}"
            suffix += 1

        cmap = CnaMap(name)
        cmap["boxes"] = boxes
        self.appdata.project.maps[name] = cmap
        mmap = MapView(self.appdata, self, name)
        self.connect_map_view_signals(mmap)
        idx = self.map_tabs.addTab(mmap, name)
        # show CNApy map by default
        self.map_tabs.setCurrentIndex(idx)
        mmap.update()
        self.reaction_list.reaction_mask.update_state()
        self.parent.unsaved_changes()

    @Slot()
    def convert_current_escher_to_cnapy(self):
        """Convert the currently selected Escher map to a CNApy map by reading its JSON."""
        idx = self.map_tabs.currentIndex()
        if idx < 0:
            return
        map_widget = self.map_tabs.widget(idx)
        if not isinstance(map_widget, EscherMapView):
            QMessageBox.information(self, "Convert", "Select an Escher map tab first.")
            return

        # pull latest JSON from Escher
        sem = [0]
        map_widget.retrieve_map_data(sem)
        # wait for callback
        while sem[0] < 1:
            QApplication.processEvents()

        self.maybe_create_cnapy_from_escher(map_widget.name)

    def _parse_escher_json(self, escher_json):
        """Return dict escher data or None."""
        if not escher_json:
            return None
        data = escher_json
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None
        # some Escher exports return a list with one or more map dicts
        if isinstance(data, list):
            # merge reactions from all entries that have reactions
            merged_data = {}
            reactions_merged = {}
            for entry in data:
                if isinstance(entry, dict):
                    if "reactions" in entry and isinstance(entry["reactions"], dict):
                        # merge reactions dictionaries
                        reactions_merged.update(entry["reactions"])
                        # also merge other keys (like nodes, canvas, etc.)
                        for key, value in entry.items():
                            if key != "reactions":
                                merged_data[key] = value
            if reactions_merged:
                merged_data["reactions"] = reactions_merged
                data = merged_data
            else:
                # fallback: pick the first entry that looks like a map (has reactions)
                for entry in data:
                    if isinstance(entry, dict) and "reactions" in entry:
                        data = entry
                        break
                if isinstance(data, list):
                    return None
        # Escher can also store the map under the "map" key
        if isinstance(data, dict) and "map" in data and isinstance(data["map"], dict):
            if "reactions" in data["map"]:
                data = data["map"]
        if not isinstance(data, dict):
            return None
        return data

    def set_onoff(self):
        idx = self.tabs.currentIndex()
        if idx == ModelTabIndex.Reactions and self.appdata.project.comp_values_type == 0:
            self.__set_onoff_reaction_list()
        self.__set_onoff_map()

    def __set_onoff_reaction_list(self):
        # do coloring of LB/UB columns in this case?
        view = self.reaction_list
        # block itemChanged while recoloring
        view.reaction_list.blockSignals(True)
        root = view.reaction_list.invisibleRootItem()
        child_count = root.childCount()
        for i in range(child_count):
            item = root.child(i)
            key = item.text(0)
            if key in self.appdata.project.scen_values:
                value = self.appdata.project.scen_values[key]
                color = self.appdata.compute_color_onoff(value)
                item.setBackground(ReactionListColumn.Flux, color)
            elif key in self.appdata.project.comp_values:
                value = self.appdata.project.comp_values[key]
                color = self.appdata.compute_color_onoff(value)
                item.setBackground(ReactionListColumn.Flux, color)
        view.reaction_list.blockSignals(False)

    def __set_onoff_map(self):
        idx = self.map_tabs.currentIndex()
        if idx < 0:
            return
        name = self.map_tabs.tabText(idx)
        map_view = self.map_tabs.widget(idx)
        for key in self.appdata.project.maps[name]["boxes"]:
            if key in self.appdata.project.scen_values:
                value = self.appdata.project.scen_values[key]
                color = self.appdata.compute_color_onoff(value)
                map_view.reaction_boxes[key].set_color(color)
            elif key in self.appdata.project.comp_values:
                value = self.appdata.project.comp_values[key]
                color = self.appdata.compute_color_onoff(value)
                map_view.reaction_boxes[key].set_color(color)

    def set_heaton(self):
        (low, high) = self.appdata.low_and_high()
        idx = self.tabs.currentIndex()
        if idx == ModelTabIndex.Reactions and self.appdata.project.comp_values_type == 0:
            self.__set_heaton_reaction_list(low, high)
        self.__set_heaton_map(low, high)

    def __set_heaton_reaction_list(self, low, high):
        # TODO: coloring of LB/UB columns
        view = self.reaction_list
        # block itemChanged while recoloring
        view.reaction_list.blockSignals(True)
        root = view.reaction_list.invisibleRootItem()
        child_count = root.childCount()
        for i in range(child_count):
            item = root.child(i)
            key = item.text(0)
            if key in self.appdata.project.scen_values:
                value = self.appdata.project.scen_values[key]
                color = self.appdata.compute_color_heat(value, low, high)
                item.setBackground(ReactionListColumn.Flux, color)
            elif key in self.appdata.project.comp_values:
                value = self.appdata.project.comp_values[key]
                color = self.appdata.compute_color_heat(value, low, high)
                item.setBackground(ReactionListColumn.Flux, color)
        view.reaction_list.blockSignals(False)

    def set_heaton_map(self):
        (low, high) = self.appdata.low_and_high()
        self.__set_heaton_map(low, high)

    def __set_heaton_map(self, low, high):
        idx = self.map_tabs.currentIndex()
        if idx < 0:
            return
        name = self.map_tabs.tabText(idx)
        map_view = self.map_tabs.widget(idx)
        for key in self.appdata.project.maps[name]["boxes"]:
            if key in self.appdata.project.scen_values:
                value = self.appdata.project.scen_values[key]
                color = self.appdata.compute_color_heat(value, low, high)
                map_view.reaction_boxes[key].set_color(color)
            elif key in self.appdata.project.comp_values:
                value = self.appdata.project.comp_values[key]
                color = self.appdata.compute_color_heat(value, low, high)
                map_view.reaction_boxes[key].set_color(color)

    def __recolor_map(self):
        """recolor the map based on the activated coloring mode"""
        if self.parent.heaton_action.isChecked():
            self.set_heaton_map()
        elif self.parent.onoff_action.isChecked():
            self.__set_onoff_map()

    def jump_to_metabolite(self, metabolite: str):
        self.tabs.setCurrentIndex(ModelTabIndex.Metabolites)
        m = self.tabs.widget(ModelTabIndex.Metabolites)
        m.set_current_item(metabolite)

    def jump_to_reaction(self, reaction: str):
        self.tabs.setCurrentIndex(ModelTabIndex.Reactions)
        m = self.tabs.widget(ModelTabIndex.Reactions)
        m.set_current_item(reaction)

    def jump_to_gene(self, gene: str):
        self.tabs.setCurrentIndex(ModelTabIndex.Genes)
        m = self.tabs.widget(ModelTabIndex.Genes)
        m.set_current_item(gene)

    @Slot(int)
    def select_item_from_history(self, index: int):
        item_id, item_type = self.model_item_history.itemData(index)
        if item_type == ModelItemType.Reaction:
            self.jump_to_reaction(item_id)
        elif item_type == ModelItemType.Metabolite:
            self.jump_to_metabolite(item_id)
        elif item_type == ModelItemType.Gene:
            self.jump_to_gene(item_id)

    def add_model_item_to_history(self, item_id: str, item_name: str, item_type: ModelItemType):
        item_data = [item_id, item_type]
        index = self.model_item_history.findData(item_data)
        with QSignalBlocker(self.model_item_history):
            if index >= 0:
                index = self.model_item_history.removeItem(index)
            self.model_item_history.insertItem(0, item_id + " (" + ModelItemType(item_type).name + ")", item_data)
            self.model_item_history.setItemData(0, item_name, Qt.ToolTipRole)
            self.model_item_history.setCurrentIndex(0)

    def update_item_in_history(self, previous_id: str, new_id: str, new_name: str, item_type: ModelItemType):
        index = self.model_item_history.findData([previous_id, item_type])
        if index >= 0:
            self.model_item_history.setItemData(index, [new_id, item_type])
            self.model_item_history.setItemText(index, new_id + " (" + ModelItemType(item_type).name + ")")

    def remove_top_item_history_entry(self):
        # can be used when a reaction or metabolite is deleted because
        # in that case the item which is being deleted is at the top
        with QSignalBlocker(self.model_item_history):
            self.model_item_history.removeItem(0)
            self.model_item_history.setCurrentIndex(-1)

    @Slot()
    def clear_model_item_history(self):
        with QSignalBlocker(self.model_item_history):
            self.model_item_history.clear()

    broadcastReactionID = Signal(str)


class EscherMapFileDialog(QDialog):
    """Dialog to select JSON and/or image (PNG/SVG) files for Escher map.

    Supports two modes:
    1. JSON + PNG: Full map with reaction positions from JSON and PNG background
    2. PNG only: Map with PNG background, reactions can be added manually
    """

    def __init__(self, parent, work_directory: str):
        super().__init__(parent)
        self.setWindowTitle("Add CNApy Map from Files")
        self.json_file = None
        self.image_file = None

        layout = QVBoxLayout()

        # Explanation
        explanation = QLabel(
            "Create a CNApy map from files:\n"
            "• JSON file (optional): Contains reaction positions from Escher\n"
            "• PNG/SVG file (required): Background image for the map\n\n"
            "If JSON is provided, reaction boxes will be placed at the positions from JSON.\n"
            "If only image is provided, you can add reaction boxes manually."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        layout.addSpacing(10)

        # JSON file selection (optional)
        json_layout = QHBoxLayout()
        json_layout.addWidget(QLabel("JSON file (optional):"))
        self.json_edit = QLineEdit()
        self.json_edit.setReadOnly(True)
        self.json_edit.setPlaceholderText("Optional - for reaction positions")
        json_layout.addWidget(self.json_edit)
        json_btn = QPushButton("Browse...")
        json_btn.clicked.connect(self.select_json_file)
        json_layout.addWidget(json_btn)
        clear_json_btn = QPushButton("Clear")
        clear_json_btn.clicked.connect(self.clear_json_file)
        json_layout.addWidget(clear_json_btn)
        layout.addLayout(json_layout)

        # Image file selection (PNG/SVG)
        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("Image file (required):"))
        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        self.image_edit.setPlaceholderText("PNG or SVG file")
        image_layout.addWidget(self.image_edit)
        image_btn = QPushButton("Browse...")
        image_btn.clicked.connect(self.select_image_file)
        image_layout.addWidget(image_btn)
        layout.addLayout(image_layout)

        layout.addSpacing(10)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept_dialog)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.work_directory = work_directory

    def select_json_file(self):
        filename = QFileDialog.getOpenFileName(
            self, "Select Escher JSON file", self.work_directory, "JSON files (*.json)"
        )[0]
        if filename and os.path.exists(filename):
            self.json_file = filename
            self.json_edit.setText(os.path.basename(filename))
            # Update work directory
            self.work_directory = os.path.dirname(filename)

    def clear_json_file(self):
        self.json_file = None
        self.json_edit.clear()

    def select_image_file(self):
        filename = QFileDialog.getOpenFileName(
            self,
            "Select image file",
            self.work_directory,
            "Image files (*.png *.svg);;PNG files (*.png);;SVG files (*.svg)",
        )[0]
        if filename and os.path.exists(filename):
            if not (filename.lower().endswith(".png") or filename.lower().endswith(".svg")):
                QMessageBox.warning(self, "Invalid file", "Please select a PNG or SVG file.")
                return
            self.image_file = filename
            self.image_edit.setText(os.path.basename(filename))
            # Update work directory
            self.work_directory = os.path.dirname(filename)

    def accept_dialog(self):
        # Image file is required
        if not self.image_file or not os.path.exists(self.image_file):
            QMessageBox.warning(self, "Missing file", "Please select an image file (PNG or SVG).")
            return
        # JSON is optional
        if self.json_file and not os.path.exists(self.json_file):
            QMessageBox.warning(self, "Invalid JSON", "The selected JSON file does not exist.")
            return
        self.accept()


class ConfirmMapDeleteDialog(QDialog):
    def __init__(self, parent, idx: int, name: str):
        super(ConfirmMapDeleteDialog, self).__init__(parent)
        # Create widgets
        self.parent = parent
        self.idx = idx
        self.name = name
        self.lable = QLabel("Do you really want to delete this map?")
        self.button_yes = QPushButton("Yes delete")
        self.button_no = QPushButton("No!")
        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.lable)
        layout.addWidget(self.button_yes)
        layout.addWidget(self.button_no)
        # Set dialog layout
        self.setLayout(layout)
        # Add button signals to the slots
        self.button_yes.clicked.connect(self.delete)
        self.button_no.clicked.connect(self.reject)

    def delete(self):
        del self.parent.appdata.project.maps[self.name]
        self.parent.map_tabs.removeTab(self.idx)
        self.parent.reaction_list.reaction_mask.update_state()
        self.parent.parent.unsaved_changes()
        self.accept()


class MapReactionAddDialog(QDialog):
    """Dialog to pick a reaction ID (from model or custom) with auto-complete."""

    def __init__(self, parent, appdata: AppData):
        super().__init__(parent)
        self._reaction_id = None
        self.setWindowTitle("Add reaction box to map")
        self._reactions = appdata.project.cobra_py_model.reactions

        layout = QVBoxLayout()

        # Explanation
        explanation = QLabel(
            "Add a reaction box to the map.\n"
            "You can select a reaction from the model, or enter a custom ID\n"
            "for external reactions (the box will display flux values when computed)."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Reaction ID:"))

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Enter or select a reaction ID")
        # Build suggestions as "id | name"
        suggestions = [f"{r.id} | {r.name}" for r in self._reactions]
        completer = QCompleter(suggestions, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.activated[str].connect(self._apply_completion)
        self.edit.setCompleter(completer)
        self.edit.returnPressed.connect(self.accept_dialog)
        layout.addWidget(self.edit)

        # Custom ID checkbox
        self.custom_check = QCheckBox("Allow custom ID (not in model)")
        self.custom_check.setToolTip(
            "Check this to add a box for a reaction ID that is not in the model.\n"
            "This is useful for linking to external reactions or placeholders."
        )
        layout.addWidget(self.custom_check)

        btns = QHBoxLayout()
        ok_btn = QPushButton("Add")
        ok_btn.clicked.connect(self.accept_dialog)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        self.setLayout(layout)

    def accept_dialog(self):
        text = self.edit.text().strip()
        # Accept either pure ID or "id | name"
        if "|" in text:
            text = text.split("|", 1)[0].strip()

        if not text:
            QMessageBox.warning(self, "Empty ID", "Please enter a reaction ID.")
            return

        # Check if reaction is in model
        in_model = text in self._reactions

        if not in_model and not self.custom_check.isChecked():
            QMessageBox.warning(
                self,
                "Reaction not in model",
                f"Reaction '{text}' is not in the current model.\n\n"
                "If you want to add a custom reaction box (e.g., for external reactions),\n"
                "check the 'Allow custom ID' checkbox.",
            )
            return

        self._reaction_id = text
        self.accept()

    def _apply_completion(self, text: str):
        if "|" in text:
            text = text.split("|", 1)[0].strip()
        self.edit.setText(text)
        # keep focus so user can still cancel or press Add; do not auto-accept
        self.edit.setFocus()

    def reaction_id(self):
        return self._reaction_id

    def is_custom(self):
        """Return True if this is a custom ID not in the model."""
        return self.custom_check.isChecked()
