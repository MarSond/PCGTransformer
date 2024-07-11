import sys
import json
import yaml
from pathlib import Path, WindowsPath
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, 
                             QTableWidgetItem, QFileDialog, QMessageBox,
                             QTabWidget, QLabel, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

# Import constants
from MLHelper.constants import *

def path_constructor(loader, node):
    return WindowsPath(loader.construct_scalar(node))

yaml.add_constructor('!WindowsPath', path_constructor)

class RunMetricsParser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Run Metrics Parser")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Runs")
        self.load_button.clicked.connect(self.load_runs)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.load_runs)
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected)
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.reload_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout.addLayout(self.button_layout)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.training_table = self.create_table()
        self.validation_table = self.create_table()
        self.inference_table = self.create_table()
        self.error_table = self.create_table()

        self.tab_widget.addTab(self.training_table, "Training")
        self.tab_widget.addTab(self.inference_table, "Inference")
        self.tab_widget.addTab(self.error_table, "Errors")

        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_scroll_area.setWidget(self.plot_widget)

        self.splitter = QHBoxLayout()
        self.splitter.addWidget(self.tab_widget, 2)
        self.splitter.addWidget(self.plot_scroll_area, 1)
        self.layout.addLayout(self.splitter)

        self.runs_dir = Path("./runs/")
        self.run_data = {"training": [], "inference": [], "error": []}

    def create_table(self):
        table = QTableWidget()
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSortingEnabled(True)
        table.itemClicked.connect(self.show_plots)
        return table

    def load_runs(self):
        if not self.runs_dir.exists():
            reply = QMessageBox.question(
                self, "Directory Not Found",
                f"The default directory '{self.runs_dir}' doesn't exist. Do you want to select a different directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.runs_dir = Path(QFileDialog.getExistingDirectory(self, "Select Runs Directory"))
            else:
                return

        if self.runs_dir.exists():
            self.parse_runs()
        else:
            QMessageBox.warning(self, "Error", "Selected directory does not exist.")

    def parse_runs(self):
        self.run_data = {"training": [], "validation": [], "inference": [], "error": []}
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                metrics_file = run_dir / FILENAME_METRICS_VALUE
                config_file = run_dir / FILENAME_RUN_CONFIG_VALUE
                if metrics_file.exists() and config_file.exists():
                    try:
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)
                        with open(config_file, "r") as f:
                            config = yaml.safe_load(f)
                        
                        run_type = config.get(TASK_TYPE, "unknown")
                        if run_type == TASK_TYPE_TRAINING:
                            self.run_data["training"].append({"name": run_dir.name, "metrics": metrics, "config": config})
                        elif run_type == TASK_TYPE_VALIDATION:
                            self.run_data["validation"].append({"name": run_dir.name, "metrics": metrics, "config": config})
                        elif run_type == TASK_TYPE_INFERENCE:
                            self.run_data["inference"].append({"name": run_dir.name, "metrics": metrics, "config": config})
                        else:
                            self.run_data["error"].append({"name": run_dir.name, "error": f"Unknown task type: {run_type}"})
                    except (json.JSONDecodeError, yaml.YAMLError) as e:
                        self.run_data["error"].append({"name": run_dir.name, "error": f"Error parsing files: {str(e)}"})
                else:
                    self.run_data["error"].append({"name": run_dir.name, "error": "Missing metrics or config file"})
        
        self.update_tables()

    def update_tables(self):
        for run_type, table in [("training", self.training_table),
                                ("inference", self.inference_table),
                                ("error", self.error_table)]:
            self.update_single_table(table, self.run_data[run_type], run_type)

    def update_single_table(self, table, data, run_type):
        table.setRowCount(len(data))
        if run_type == "error":
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels([RUN_NAME, "Error"])
            for row, run in enumerate(data):
                table.setItem(row, 0, QTableWidgetItem(run["name"]))
                table.setItem(row, 1, QTableWidgetItem(run["error"]))
        else:
            table.setColumnCount(6)
            table.setHorizontalHeaderLabels([RUN_NAME, METRICS_ACCURACY, METRICS_F1, METRICS_PRECISION, METRICS_RECALL, METRICS_MCC])
            for row, run in enumerate(data):
                table.setItem(row, 0, QTableWidgetItem(run["name"]))
                try:
                    validation_metrics = run["metrics"][AVERAGES][VALIDATION]
                    for col, metric in enumerate([METRICS_ACCURACY, METRICS_F1, METRICS_PRECISION, METRICS_RECALL, METRICS_MCC], start=1):
                        if metric in validation_metrics and validation_metrics[metric]:
                            value = validation_metrics[metric][-1][MEAN]
                            table.setItem(row, col, QTableWidgetItem(f"{value:.4f}"))
                        else:
                            table.setItem(row, col, QTableWidgetItem("N/A"))
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Error in run {run['name']}: {str(e)}")
                    for col in range(1, 6):
                        table.setItem(row, col, QTableWidgetItem("Error"))

        table.resizeColumnsToContents()

    def delete_selected(self):
        current_tab = self.tab_widget.currentWidget()
        selected_rows = set(index.row() for index in current_tab.selectedIndexes())
        if not selected_rows:
            return

        reply = QMessageBox.question(self, "Confirm Deletion", 
                                     "Are you sure you want to delete the selected runs?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            run_type = self.tab_widget.tabText(self.tab_widget.currentIndex()).lower()
            for row in sorted(selected_rows, reverse=True):
                run_path = self.runs_dir / self.run_data[run_type][row]["name"]
                try:
                    for file in run_path.glob("*"):
                        file.unlink()
                    run_path.rmdir()
                    del self.run_data[run_type][row]
                except Exception as e:
                    QMessageBox.warning(self, "Deletion Error", f"Error deleting {run_path}: {str(e)}")

            self.update_tables()

    def show_plots(self, item):
        # Clear previous plots
        for i in reversed(range(self.plot_layout.count())): 
            self.plot_layout.itemAt(i).widget().setParent(None)

        row = item.row()
        current_tab = self.tab_widget.currentWidget()
        run_type = self.tab_widget.tabText(self.tab_widget.currentIndex()).lower()

        if run_type in ["training", "inference"]:
            run_name = self.run_data[run_type][row]["name"]
            run_dir = self.runs_dir / run_name
            plot_file = run_dir / FILENAME_METRIC_PLOTS_VALUE

            if plot_file.exists():
                pixmap = QPixmap(str(plot_file))
                label = QLabel()
                label.setPixmap(pixmap)
                self.plot_layout.addWidget(label)
            else:
                self.plot_layout.addWidget(QLabel("No plots available for this run."))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RunMetricsParser()
    window.show()
    sys.exit(app.exec())
