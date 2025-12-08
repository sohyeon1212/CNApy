from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit,
                            QMessageBox, QPushButton, QVBoxLayout, QSpinBox)
from cnapy.appdata import AppData

class FluxSamplingDialog(QDialog):
    def __init__(self, appdata: AppData):
        QDialog.__init__(self)
        self.setWindowTitle("Flux Sampling")
        self.appdata = appdata

        self.layout = QVBoxLayout()
        
        # Number of samples
        l1 = QHBoxLayout()
        l1.addWidget(QLabel("Number of samples:"))
        self.n_samples = QSpinBox()
        self.n_samples.setRange(1, 1000000)
        self.n_samples.setValue(5000)
        l1.addWidget(self.n_samples)
        self.layout.addItem(l1)

        # Thinning
        l2 = QHBoxLayout()
        l2.addWidget(QLabel("Thinning factor:"))
        self.thinning = QSpinBox()
        self.thinning.setRange(1, 10000)
        self.thinning.setValue(100)
        l2.addWidget(self.thinning)
        self.layout.addItem(l2)

        # Processes
        l3 = QHBoxLayout()
        l3.addWidget(QLabel("Processes (1 for single core):"))
        self.processes = QSpinBox()
        self.processes.setRange(1, 64)
        self.processes.setValue(4)
        l3.addWidget(self.processes)
        self.layout.addItem(l3)

        # Buttons
        l_btns = QHBoxLayout()
        self.button = QPushButton("Start Sampling")
        self.cancel = QPushButton("Cancel")
        l_btns.addWidget(self.button)
        l_btns.addWidget(self.cancel)
        self.layout.addItem(l_btns)
        
        self.setLayout(self.layout)

        self.cancel.clicked.connect(self.reject)
        self.button.clicked.connect(self.accept)
