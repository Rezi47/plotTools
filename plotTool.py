#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import itertools
from io import StringIO
import tkinter as tk
try:
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QWidget, QVBoxLayout, QPushButton, 
    QDialog, QLabel, QLineEdit, QDesktopWidget, QCheckBox, QHBoxLayout, 
    QFrame, QTextEdit, QMainWindow, QMessageBox
    )
    from PyQt5.QtGui import QGuiApplication, QDoubleValidator
    import sys
    import os
    pyqt_available = True
    from functools import partial
except ImportError:
    pyqt_available = False
    import tkinter as tk
    from tkinter import filedialog

app = QApplication(sys.argv)
          
def extract_general_values(file_path):
    """
    Reads data from the given file and general values.
    """
    with open(file_path, "r") as infile:
        cleaned_data = "".join(line.replace("(", "").replace(")", "") for line in infile)
    cleaned_file = StringIO(cleaned_data)
    data = np.loadtxt(cleaned_file, delimiter=None, skiprows=skip_row, usecols=(usecols))
    times = data[:, 0]
    variable_data = data[:, 1:].T
    return times, variable_data
                
def extract_motion_values(file_path):
    """
    Reads data from the given file and motion values.
    """
    times = []
    variable_data = []

    previous_line = ""  # Keep track of the previous line
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Time =") and previous_line.startswith("deltaT"):
                time_str = line.split('=')[1].strip()
                if time_str.endswith("s"):  # Check if the string ends with "s"
                    time_str = time_str[:-1]  # Remove the "s"
                time = float(time_str)  # Convert the cleaned string to a float
                times.append(time)            
            previous_line = line  # Update the previous line

            if line.strip().startswith('q ='):
                # Extract the content inside parentheses
                start = line.find('(') + 1
                end = line.find(')')
                if start > 0 and end > start:  # Ensure valid indices
                    values = list(map(float, line[start:end].split()))
                    variable_data.append(values)

    # Transpose the list of lists to separate variables
    variable_data = list(map(list, zip(*variable_data))) if variable_data else []
    return np.array(times), [np.array(var) for var in variable_data]

fig_config = {
    'general': {
        'label': 'General',
        'axisTitle': 'Amplitude',
        'dimension': '-',
        'function': extract_general_values
    },
    'motion': {
        'label': 'Motion',
        'axisTitle': 'Amplitude',
        'dimension': 'm',
        'function': extract_motion_values
    },
}

def extract_data(files):
    parsed_data = []

    # for file in files:
    for i, file in enumerate(files):
        if plot_type in fig_config:
            times, variable_data = fig_config[plot_type]['function'](file)
        else:
            sys.exit(f"Unknown plot_type: {plot_type}. Available plot_type: {', '.join(fig_config.keys())}")
            
        # Ensure arrays have the same length
        min_length = min(len(times), len(variable_data[0]))
        times, variable_data = times[:min_length], variable_data[:min_length]

        # # Apply shifts and scaling to each variable if needed
        for i, var in enumerate(variable_data):
            #if i == 0 or i == 2:
                var = var / scale  # Apply scale        
                var = var + shift  # Apply shift             
                variable_data[i] = var

        parsed_data.append((times, variable_data))
    
    return parsed_data

def normalize_to_origin(parsed_data):
    """
    Normalizes each variable's data in-place by subtracting its first value,
    making all variables start at 0 at the beginning of time.
    Modifies the input data directly instead of returning a new object.
    """
    for _, variable_data in parsed_data:  # Loop through each (times, variable_data) pair
        for var_array in variable_data:   # Loop through each variable array
            if len(var_array) > 0:
                first_value = np.mean(var_array)
                var_array -= first_value  # Subtract first value in-place (modifies original array)
def plot(parsed_data, labels, disable_plot):
    """
    Creates dynamic plots for an arbitrary number of variables extracted from multiple files.
    """

    colors = ['b', 'r', 'g', 'c', 'm', 'y']  # color cycle
    linestyles = ['-', '--', ':', '-.']  # linestyle cycle

    # Maximum length of variable_data
    max_num_variables = 0
    for item in parsed_data:
        current_length = len(item[1])
        max_num_variables = max(max_num_variables, current_length)

    # Create subplots dynamically
    fig, axs = plt.subplots(1, max_num_variables, figsize=(5 * max_num_variables, 5))

    if fig_title:
        fig.suptitle(fig_title, fontsize=14) 
    
    if max_num_variables == 1:  # Single subplot case
        axs = [axs]        
    # Plot data for each variable
    for i in range(max_num_variables):
        # Create cyclic iterators            
        color_cycle = itertools.cycle(colors)
        linestyle_cycle = itertools.cycle(linestyles)
        for (times, variables), label in zip(parsed_data, labels):
            color = next(color_cycle)
            linestyle = next(linestyle_cycle)                
            axs[i].plot(times, variables[i], label=label, linestyle=linestyle, linewidth=1, color=color)
        
        # Set subplot titles and labels
        #axs[i].set_title(f"{axis_title} {i+1}")       
        axs[i].set_xlabel(r"t ($s$)")
        axs[i].set_ylabel(fr"{axis_title} (${axis_dim}$)")
        axs[i].set_xlim(
            left=x_min if x_min is not None else axs[i].get_xlim()[0],
            right=x_max if x_max is not None else axs[i].get_xlim()[1]
        )
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    if disable_plot: plt.show()

    return fig

def save_plot_func(fig):
    """Saves the plot to a file"""
    output_dir = "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, f"{fig_title + '_' if fig_title else ''}{axis_title}.png"))
    print(f"Figure saved to {output_dir}/{fig_title + '_' if fig_title else ''}{axis_title}.png")

def save_data_func(parsed_data, labels, fig_title):
    base_dir = os.path.dirname(files[0]) if files else ""
    output_dir = os.path.join(base_dir, "extractedData")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First determine the maximum number of variables across all datasets
    max_vars = max(len(variables) for _, variables in parsed_data)

    # Create one file per variable index
    for var_idx in range(max_vars):
        # Create filename for this variable index
        file_path = os.path.join(output_dir, f"{fig_title + '_' if fig_title else ''}Variable_{var_idx+1}.csv")
        
        with open(file_path, 'w') as f:
            # Write header - Time columns for each dataset that has this variable
            headers = []
            for label, (times, variables) in zip(labels, parsed_data):
                if var_idx < len(variables):
                    headers.append(f"Time_{label}")
                    headers.append(f"Data_{label}")
            f.write(",".join(headers) + "\n")
            
            # Find maximum time length across datasets that have this variable
            relevant_datasets = [times for times, variables in parsed_data if var_idx < len(variables)]
            max_length = max(len(times) for times in relevant_datasets) if relevant_datasets else 0
            
            # Write data rows
            for row_idx in range(max_length):
                row_data = []
                for (times, variables), label in zip(parsed_data, labels):
                    if var_idx < len(variables):
                        # Add time value if exists, else empty
                        time_val = str(times[row_idx]) if row_idx < len(times) else ""
                        row_data.append(time_val)
                        # Add variable value if exists, else empty
                        var_val = str(variables[var_idx][row_idx]) if row_idx < len(variables[var_idx]) else ""
                        row_data.append(var_val)
                
                f.write(",".join(row_data) + "\n")
        
        print(f"Saved Variable {var_idx+1} data in: {file_path}")
            
def interactive_plot_type_selection_TK():
    root = tk.Tk()
    plot_type = []
    def set_plot_type(plot_value):
        nonlocal plot_type
        plot_type = plot_value
        root.quit()
        root.destroy()

    root.title("Select Plot type")
    button1 = tk.Button(root, text="motion", command=lambda: set_plot_type('motion'))
    button1.pack(padx=50, pady=10)

    button2 = tk.Button(root, text="flux", command=lambda: set_plot_type('flux'))
    button2.pack(padx=50, pady=10)

    button3 = tk.Button(root, text="force", command=lambda: set_plot_type('force'))
    button3.pack(padx=50, pady=10)

    button4 = tk.Button(root, text="interfaceHeight", command=lambda: set_plot_type('interfaceHeight'))
    button4.pack(padx=50, pady=10)
    
    root.mainloop()
   
    return plot_type

def interactive_plot_type_selection_QT():
    plot_type = None
    x_min = None
    x_max = None
    scale = 1
    shift = 0
    norm_origin = False
    files = []
    labels = []
    fig_title = None
    axis_title = None
    axis_dim = None
    skip_row = None
    usecols = None
    plotflag = False

    def update_values():
        nonlocal plot_type, x_min, x_max, scale, shift, norm_origin, fig_title, axis_title, axis_dim, skip_row, usecols
        # Update the current values from the UI elements
        axis_title = axis_title_input.text() if axis_title_input.text() else None
        fig_title = fig_title_input.text() if fig_title_input.text() else None
        axis_dim = axis_dim_input.text() if axis_dim_input.text() else None
        x_min = float(x_min_input.text()) if x_min_input.text() else None
        x_max = float(x_max_input.text()) if x_max_input.text() else None
        scale = float(scale_input.text()) if scale_input.text() else 1
        shift = float(shift_input.text()) if shift_input.text() else 0
        norm_origin = norm_origin_checkbox.isChecked()

        axis_title_input.setText(fig_config[plot_type]['axisTitle'])
        axis_dim_input.setText(fig_config[plot_type]['dimension'])
        scale_input.setText("1")
        shift_input.setText("0")

        # If they are valid integer, use it; otherwise default to 0.
        skip_row = int(skip_row_input.text()) if skip_row_input.text().isdigit() else 0
        usecols_text = usecols_input.text()
        usecols = [int(col) for col in usecols_text.split(",") if col.strip().isdigit()] if usecols_text else None

    def set_plot_type(plot_value):
        nonlocal plot_type
        plot_type = plot_value
        update_values()  # Capture the latest values when plot type is selected

        # Change the style of the selected button to indicate it's selected
        for button in plot_buttons.values():
            button.setStyleSheet('background-color: none')  # Reset all buttons
        plot_buttons[plot_value].setStyleSheet('background-color: lightblue')  # Highlight the selected one

        skip_row_label.setEnabled(True)
        skip_row_input.setEnabled(True)
        usecols_label.setEnabled(True)
        usecols_input.setEnabled(True)
        
        # Dynamically gray out fields for "Axis Title" and "Dimension" if "motion" is selected
        if plot_value == 'motion':
            skip_row_label.setEnabled(False)
            skip_row_input.setEnabled(False)
            usecols_label.setEnabled(False)
            usecols_input.setEnabled(False)
        
        window.adjustSize()

    def plot_button_clicked():
        nonlocal plotflag, x_min, x_max

        if not files:
            QMessageBox.critical(window, "Error", "Please select at least one file to plot.")
            return
        
        update_values()
        plotflag = True
        parsed_data = extract_data(files)
        if norm_origin: normalize_to_origin(parsed_data)
        plot(parsed_data, labels, disable_plot)
        # window.close()

    def write_plot_button_clicked():
        if not files:
            QMessageBox.critical(window, "Error", "Please select at least one file to plot.")
            return
        
        update_values()
        plotflag = True
        parsed_data = extract_data(files)
        if norm_origin: normalize_to_origin(parsed_data)
        fig = plot(parsed_data, labels, disable_plot)
        save_plot_func(fig)
    
    def write_data_button_clicked():
        if not files:
            QMessageBox.critical(window, "Error", "Please select at least one file to process.")
            return
        
        update_values()
        plotflag = True
        parsed_data = extract_data(files)
        if norm_origin: normalize_to_origin(parsed_data)
        save_data_func(parsed_data, labels, fig_title)

    def update_file_paths(file_paths, file_labels):
        nonlocal files, labels
        files = file_paths
        labels = file_labels
        
    def create_horizontal_line():
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line
    
    window = QWidget()
    window.setWindowTitle("Plot Setting")
    layout = QVBoxLayout()

    title_label = QLabel("Select the Plot Type")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size: 13px; font-weight: regular; margin-bottom: 10px;")
    layout.addWidget(title_label)

    plot_buttons = {key: QPushButton(config['label']) for key, config in fig_config.items()}

    for plot_value, button in plot_buttons.items():
        button.clicked.connect(lambda checked, pv=plot_value: set_plot_type(pv))
        layout.addWidget(button)

    # Auto-click the "general" button
    QTimer.singleShot(100, plot_buttons["general"].click)

    # Add Figure Title
    fig_title_layout = QHBoxLayout()
    fig_title_label = QLabel("Figure title:")
    fig_title_input = QLineEdit()

    fig_title_input.setFixedWidth(140)

    fig_title_layout.addWidget(fig_title_label)
    fig_title_layout.addWidget(fig_title_input)
    fig_title_layout.addStretch()

    layout.addLayout(fig_title_layout)

    ########### Add the 1st horizontal line separator ###########
    layout.addWidget(create_horizontal_line())

    # Add some fields right after the plot type buttons
    axis_layout = QHBoxLayout()
    axis_title_label = QLabel("Axis title:")
    axis_title_input = QLineEdit()
    axis_dim_label = QLabel("Dimension:")
    axis_dim_input = QLineEdit()

    axis_dim_input.setFixedWidth(70)

    # Add widgets to the horizontal layout
    axis_layout.addWidget(axis_title_label)
    axis_layout.addWidget(axis_title_input)
    axis_layout.addSpacing(10)  # Add some space between inputs for better visual separation
    axis_layout.addWidget(axis_dim_label)
    axis_layout.addWidget(axis_dim_input)
    axis_layout.addStretch()

    layout.addLayout(axis_layout)

    # Add "Skip Row" and "Usecols" fields in the same row
    data_layout = QHBoxLayout()
    skip_row_label = QLabel("Skiped rows:")
    skip_row_input = QLineEdit()
    usecols_label = QLabel("Used columns:")
    usecols_input = QLineEdit()
    usecols_input.setPlaceholderText("Example: 0,1,2,4")

    # Set fixed width for skip_row_input to allow 3-digit numbers
    skip_row_input.setFixedWidth(50)

    # Add widgets to the horizontal layout
    data_layout.addWidget(skip_row_label)
    data_layout.addWidget(skip_row_input)
    data_layout.addSpacing(10)  # Add space to align usecols label with the Dimension label
    data_layout.addWidget(usecols_label)
    data_layout.addWidget(usecols_input)
    data_layout.addStretch()

    layout.addLayout(data_layout)

    ########### Add the 2st horizontal line separator ###########
    layout.addWidget(create_horizontal_line())

    s_layout = QHBoxLayout()
    scale_label = QLabel("Scale:")
    scale_input = QLineEdit()
    shift_label = QLabel("Shift:")
    shift_input = QLineEdit()

    scale_input.setFixedWidth(60)
    shift_input.setFixedWidth(60)

    # Add spacing between the labels and inputs
    s_layout.addWidget(scale_label)
    s_layout.addWidget(scale_input)
    s_layout.addSpacing(10) # Reduced spacing between scale and shift
    s_layout.addWidget(shift_label)
    s_layout.addWidget(shift_input)
    s_layout.addStretch()

    # Align elements in the row
    s_layout.setAlignment(Qt.AlignLeft)

    norm_origin_layout = QHBoxLayout()
    norm_origin_checkbox = QCheckBox("Normalize to the Origin")
    norm_origin_layout.addWidget(norm_origin_checkbox)
    norm_origin_layout.setAlignment(Qt.AlignLeft)  # Align to the left   

    layout.addLayout(s_layout)
    layout.addLayout(norm_origin_layout)

    ########### Add the 3nd horizontal line separator ###########
    layout.addWidget(create_horizontal_line())

    # Add input fields for "x min" and "x max" in the same row
    x_layout = QHBoxLayout()
    x_min_label = QLabel("x min:")
    x_min_input = QLineEdit()
    x_max_label = QLabel("x max:")
    x_max_input = QLineEdit()

    x_min_input.setFixedWidth(60)
    x_max_input.setFixedWidth(60)

    # Add spacing between the labels and inputs
    x_layout.addWidget(x_min_label)
    x_layout.addWidget(x_min_input)
    x_layout.addSpacing(10) # Reduced spacing between scale and shift
    x_layout.addWidget(x_max_label)
    x_layout.addWidget(x_max_input)
    x_layout.addStretch()  # Push everything to the left

    # Align elements in the row
    x_layout.setAlignment(Qt.AlignLeft)
    layout.addLayout(x_layout)

    ########### Add the 5th horizontal line separator ###########
    layout.addWidget(create_horizontal_line())
  
    # Integrate FileSelectorApp (Browse button functionality)
    file_selector = FileSelectorApp()
    file_selector.files_updated.connect(update_file_paths)
    layout.addWidget(file_selector)

    button_layout = QHBoxLayout()
    
    plot_button = QPushButton("👁️ Plot")
    plot_button.clicked.connect(plot_button_clicked)
    plot_button.setFixedSize(120, 40)
    
    write_plot_button = QPushButton("💾 Save Plot")
    write_plot_button.clicked.connect(write_plot_button_clicked)
    write_plot_button.setFixedSize(120, 40)
    
    write_data_button = QPushButton("📊 Save Data")
    write_data_button.clicked.connect(write_data_button_clicked)
    write_data_button.setFixedSize(120, 40)
    
    # Style all buttons consistently
    for button in [plot_button, write_plot_button, write_data_button]:
        button.setStyleSheet("""
        font-size: 15px;
        font-weight: bold;
        padding: 3px;
        color: white;
        background-color: #007BFF;
        border-radius: 3px;
        """)
    
    button_layout.addWidget(plot_button)
    button_layout.addWidget(write_plot_button)
    button_layout.addWidget(write_data_button)
    layout.addLayout(button_layout)

    # Set the layout and show the window
    window.setLayout(layout)
    window.resize(400, 300)
    window.show()

    skip_row_input.setValidator(QDoubleValidator())
    x_min_input.setValidator(QDoubleValidator())
    x_min_input.setValidator(QDoubleValidator())
    x_max_input.setValidator(QDoubleValidator())
    scale_input.setValidator(QDoubleValidator())
    shift_input.setValidator(QDoubleValidator())

    # Center the window on the screen
    screen = QGuiApplication.primaryScreen()
    screen_geometry = screen.availableGeometry()
    window.move(screen_geometry.center() - window.rect().center())

    # Execute the application
    app.exec_()

    return (
        plot_type, fig_title, plotflag,
        axis_title, axis_dim, x_min, x_max,
        skip_row, usecols, scale, shift, norm_origin,
        files, labels
    )

class FileSelectorApp(QWidget):
    files_updated = pyqtSignal(list, list)  # Now emits two separate lists

    def __init__(self):
        super().__init__()
        self.files = []  # List to store file paths
        self.labels = []  # List to store labels
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("File Selector")
        self.layout = QVBoxLayout()
        self.add_file_row()
        self.setLayout(self.layout)

    def add_file_row(self):
        file_row_layout = QHBoxLayout()

        browse_button = QPushButton("Browse")
        file_path_box = QLineEdit()
        file_path_box.setReadOnly(True)
        label_input = QLineEdit()
        label_input.setPlaceholderText("Enter label")

        # Connect signals
        browse_button.clicked.connect(partial(self.select_file, file_path_box, label_input))
        label_input.textChanged.connect(lambda: self.update_values(file_path_box, label_input))

        file_row_layout.addWidget(browse_button)
        file_row_layout.addWidget(file_path_box)
        file_row_layout.addWidget(QLabel("Label:"))
        file_row_layout.addWidget(label_input)
        self.layout.addLayout(file_row_layout)

    def select_file(self, file_path_box, label_input):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        if file_path:
            file_name = os.path.basename(file_path)
            dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
            file_path_box.setText(file_name)
            file_path_box.setProperty("full_path", file_path)
            label_input.setPlaceholderText(dir_name)
            self.update_values(file_path_box, label_input)
            self.add_file_row()

    def update_values(self, file_path_box=None, label_input=None):
        self.files = []
        self.labels = []
        
        for file_row in self.layout.children():
            if isinstance(file_row, QHBoxLayout):
                current_file_box = file_row.itemAt(1).widget()
                current_label_input = file_row.itemAt(3).widget()
                
                file_path = current_file_box.property("full_path")
                label = current_label_input.text().strip() or current_label_input.placeholderText()
                
                if file_path:
                    self.files.append(file_path)
                    self.labels.append(label)
        
        self.files_updated.emit(self.files, self.labels)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process files with labels.")
    parser.add_argument('file_args', nargs=argparse.REMAINDER,
                       help="List of input files followed by their options: -label (add a label, default: dir_name)"
                       )
    parser.add_argument('-plot_type',   '-pt',  default="general",          help=f"Specify the Type of plot: {', '.join(fig_config.keys())}")
    parser.add_argument('-fig_title',   '-ft',  default=None,               help="Specify a Title for the figure")
    parser.add_argument('-axis_title',  '-at',  default="Amplitude",        help="Specify a Title for y Axis")
    parser.add_argument('-axis_dim',    '-ad',  default="-",                help="Specify a Dimension for y Axis")
    parser.add_argument('-skip_row',    '-sr',  default=0,      type=int,   help="Specify a Number to skip rows of files")
    parser.add_argument('-cols',        '-co',  default=None,               help="Specify columns numbers")
    parser.add_argument('-save_plot',   '-sp',  action='store_true',        help="Disable saving the plot (default: False)")
    parser.add_argument('-save_data',   '-sd',  action='store_true',        help="Disable saving the data (default: False)")
    parser.add_argument('-x_min',       '-xmi',                 type=float, help="Minimum x-axis value")
    parser.add_argument('-x_max',       '-xma',                 type=float, help="Maximum x-axis value")
    parser.add_argument('-scale',       '-sc',  default=1,      type=float, help="Scale value")
    parser.add_argument('-shift',       '-sh',  default=0,      type=float, help="Shift value")
    parser.add_argument('-disable_plot','-dp',  action='store_false',        help="Disable Plot")
    parser.add_argument('-norm_origin', '-no',  action='store_true',         help="Normalizes each variable's data at the beginning of time")

    args = parser.parse_args()

    usecols = [int(col) for col in args.cols.split(",") if col.strip().isdigit()] if args.cols else None

    # Use argparse for standard file input
    files = []
    labels = []
    i = 0
    while i < len(args.file_args):
        file_path = args.file_args[i]
        dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
        label = f"{dir_name}"
        i += 1

        while i < len(args.file_args) and args.file_args[i].startswith('-'):
            if args.file_args[i] == "-label": label = args.file_args[i + 1]; i += 2
            else: break

        files.append(file_path)
        labels.append(label)

    return (
        args.plot_type, args.fig_title, args.disable_plot,
        args.axis_title, args.axis_dim, args.x_min, args.x_max,
        args.skip_row, usecols, args.scale, args.shift, args.norm_origin,
        args.save_plot, args.save_data,
        files, labels
    )


if __name__ == "__main__":

    (
        plot_type, fig_title, disable_plot,
        axis_title, axis_dim, x_min, x_max,
        skip_row, usecols, scale, shift, norm_origin,
        save_plot, save_data,
        files, labels
    ) = parse_arguments()

    if not plot_type or not files:
        if pyqt_available:
            (
                plot_type, fig_title, plotflag,
                axis_title, axis_dim, x_min, x_max,
                skip_row, usecols, scale, shift, norm_origin,
                save_plot, save_data,
                files, labels
            ) = interactive_plot_type_selection_QT()

            if not plotflag:
                sys.exit(1)
        else: 
            plot_type = interactive_plot_type_selection_TK()   

    for file_path, label in zip(files, labels):
        print("File:", os.path.relpath(file_path))
        print("Label:", label)
        print()
    
    print(f"Plot type: {plot_type}")
    print(f"Figure title: {fig_title}") if fig_title else None
    print(f"Axis title: {axis_title}")
    print(f"Axis dimension: {axis_dim}")
    print(f"Skiped rows: {skip_row}") if skip_row else None
    print(f"Used columns: {usecols}") if usecols else None
    print(f"Scale value: {scale}") if scale != 1 else None
    print(f"Shift value: {shift}") if shift != 0 else None
    print(f"Normalized to the origin") if norm_origin != 0 else None
    if x_min or x_max:
        print(f"x Range: {'default' if x_min is None else x_min} - {'default' if x_max is None else x_max}")
    print()
    
