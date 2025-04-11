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

def extract_data():
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
                var = var / scale_value  # Apply scale        
                var = var + shift_value  # Apply shift             
                variable_data[i] = var

        parsed_data.append((times, variable_data))
    
    return parsed_data

def plot(disable_plot):
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

def save_func():
    """
    Saves plots and extracted data for a given fig_type.
    """
    # Save or show the plot
    if save_plot:
        output_dir = "."
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f"{fig_title + '_' if fig_title else ''}{axis_title}.png"))
        print(f"Figure saved to {output_dir}/{fig_title + '_' if fig_title else ''}{axis_title}.png")

    # Save extracted data if enabled
    if save_data:
        for file, (times, variables), label in zip(files, parsed_data, labels):
            base_dir = os.path.dirname(file)
            output_dir = os.path.join(base_dir, "extractedData")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for i, var in enumerate(variables):
                file_path = os.path.join(output_dir, f"{fig_title + '_' if fig_title else ''}{label}_{axis_title}_{i+1}.csv")
                with open(file_path, 'w') as f:
                    f.write(f"Time, {axis_title}\n")
                    for t, d in zip(times, var):
                        f.write(f"{t}, {d}\n")
                print(f"Data saved for {label}: {file_path}")
            
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
    save_plot = False
    save_data = False
    x_min = None
    x_max = None
    scale_value = 1
    shift_value = 0
    file_data = []
    fig_title = None
    axis_title = None
    axis_dim = None
    skip_row = None
    usecols = None
    plotflag = False

    def update_values():
        nonlocal plot_type, save_plot, save_data, x_min, x_max, scale_value, shift_value, fig_title, axis_title, axis_dim, skip_row, usecols
        # Update the current values from the UI elements
        axis_title = axis_title_input.text() if axis_title_input.text() else None
        fig_title = fig_title_input.text() if fig_title_input.text() else None
        axis_dim = axis_dim_input.text() if axis_dim_input.text() else None
        save_plot = save_plot_checkbox.isChecked()
        save_data = save_data_checkbox.isChecked()
        x_min = float(x_min_input.text()) if x_min_input.text() else None
        x_max = float(x_max_input.text()) if x_max_input.text() else None
        scale_value = float(scale_input.text()) if scale_input.text() else 1
        shift_value = float(shift_input.text()) if shift_input.text() else 0
        
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
        nonlocal plotflag

        if not file_data:
            QMessageBox.critical(window, "Error", "Please select at least one file to plot.")
            return
        
        # Update values before closing the window
        update_values()
        plotflag = True
        window.close()

    def update_file_paths(file_paths):
        nonlocal file_data  # Use nonlocal to modify the variable
        file_data = file_paths
        
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
    title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
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
    layout.addLayout(s_layout)

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

    ########### Add the 4rd horizontal line separator ###########
    layout.addWidget(create_horizontal_line())

    # Add input fields for "x min" and "x max" in the same row
    # Add checkboxes for "Save Plot" and "Save Data" in separate rows
    save_plot_layout = QHBoxLayout()
    save_plot_checkbox = QCheckBox("Save Plot")
    save_plot_text_label = QLabel(" Figure will be saved in: ./*_comparison.png")
    save_plot_text_label.setStyleSheet("color: gray;")  # Make text gray
    save_plot_text_label.setVisible(False)  # Hide initially
    save_plot_layout.addWidget(save_plot_checkbox)
    save_plot_layout.addWidget(save_plot_text_label)
    save_plot_layout.setAlignment(Qt.AlignLeft)  # Align to the left
    layout.addLayout(save_plot_layout)

    save_data_layout = QHBoxLayout()
    save_data_checkbox = QCheckBox("Save Data")
    save_data_text_label = QLabel(" Extracted data will be saved in: /extractedData/*.csv")
    save_data_text_label.setStyleSheet("color: gray;")  # Make text gray
    save_data_text_label.setVisible(False)  # Hide initially
    save_data_layout.addWidget(save_data_checkbox)
    save_data_layout.addWidget(save_data_text_label)
    save_data_layout.setAlignment(Qt.AlignLeft)  # Align to the left
    layout.addLayout(save_data_layout)

    # Connect the checkboxes to show/hide the gray text
    save_plot_checkbox.toggled.connect(lambda checked: save_plot_text_label.setVisible(checked))
    save_data_checkbox.toggled.connect(lambda checked: save_data_text_label.setVisible(checked))

    ########### Add the 5th horizontal line separator ###########
    layout.addWidget(create_horizontal_line())
  
    # Integrate FileSelectorApp (Browse button functionality)
    file_selector = FileSelectorApp()
    file_selector.files_data_updated.connect(update_file_paths)
    layout.addWidget(file_selector)

    # Add the final "Plot" button to execute selection
    plot_button = QPushButton("🖨️Plot")
    plot_button.clicked.connect(plot_button_clicked)
    plot_button.setFixedSize(120, 40)
    plot_button.setStyleSheet("""
    font-size: 15px;              /* Increase font size */
    font-weight: bold;           /* Make the text bold */
    padding: 3px;                /* Add internal padding for better spacing */
    color: white;                /* Set text color */
    background-color: #007BFF;   /* Set button background color (Bootstrap primary blue) */
    border-radius: 3px;         /* Add rounded corners */
    """)
    layout.addWidget(plot_button, alignment=Qt.AlignCenter)

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

    return plot_type, fig_title, axis_title, axis_dim, skip_row, usecols, save_plot, save_data, x_min, x_max, scale_value, shift_value, file_data, plotflag

class FileSelectorApp(QWidget):
    files_data_updated = pyqtSignal(list)  # Signal to emit when file paths are updated

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("File Selector")
        self.layout = QVBoxLayout()
        self.file_data = []  # List to store file data (path, label)

        # Add the initial file row
        self.add_file_row()

        self.setLayout(self.layout)

    def add_file_row(self):
        # Create a new row layout
        file_row_layout = QHBoxLayout()

        # Create "Browse" button and file path display
        browse_button = QPushButton("Browse")
        file_path_box = QLineEdit()
        file_path_box.setReadOnly(True)

        # Label input field
        label_input = QLineEdit()
        label_input.setPlaceholderText("Enter label")

        # Connect label input changes to update the file data
        label_input.textChanged.connect(self.update_values)

        # Use functools.partial to pass parameters to select_file
        browse_button.clicked.connect(partial(self.select_file, file_path_box, label_input))

        # Add widgets to the row layout
        file_row_layout.addWidget(browse_button)
        file_row_layout.addWidget(file_path_box)
        file_row_layout.addWidget(QLabel("Label:"))
        file_row_layout.addWidget(label_input)

        # Add the row layout to the main layout
        self.layout.addLayout(file_row_layout)

    def select_file(self, file_path_box, label_input):
        # Open file dialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        if file_path:
            # Extract and display only the last part of the file path
            file_name = os.path.basename(file_path)
            dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
            file_path_box.setText(file_name)

            # Store the full file path in the widget's object data
            file_path_box.setProperty("full_path", file_path)

            # Set placeholder text in the label input
            label_input.setPlaceholderText(dir_name)

            # Automatically update the file data
            self.update_values()

            # Automatically add a new row after the user selects a file
            self.add_file_row()

    def update_values(self):
        # Update file data with current file paths and labels
        self.file_data = []
        for file_row in self.layout.children():
            if isinstance(file_row, QHBoxLayout):
                file_path_box = file_row.itemAt(1).widget()  # File path box is the second widget
                label_input = file_row.itemAt(3).widget()    # Label input is the fourth widget

                file_path = file_path_box.property("full_path")
                label = label_input.text().strip() or label_input.placeholderText()

                if file_path:  # Only add data if file path is not empty
                    self.file_data.append((file_path, label))

        # Emit the updated file data
        self.files_data_updated.emit(self.file_data)
    
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

    args = parser.parse_args()

    usecols = [int(col) for col in args.cols.split(",") if col.strip().isdigit()] if args.cols else None

    # Use argparse for standard file input
    file_data = []
    i = 0
    while i < len(args.file_args):
        file_path = args.file_args[i]
        dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
        label = f"{dir_name}"
        i += 1

        while i < len(args.file_args) and args.file_args[i].startswith('-'):
            if args.file_args[i] == "-label": label = args.file_args[i + 1]; i += 2
            else: break

        file_data.append((file_path, label))

    return args.plot_type,args.axis_title,args.axis_dim, args.skip_row, usecols, args.save_plot, args.save_data, args.x_min, args.x_max, args.scale, args.shift, args.fig_title, args.disable_plot, file_data


if __name__ == "__main__":

    plot_type, axis_title, axis_dim, skip_row, usecols, save_plot, save_data, x_min, x_max, scale_value, shift_value, fig_title, disable_plot, file_data = parse_arguments()

    if not plot_type or not file_data:
        if pyqt_available:
            plot_type, fig_title, axis_title, axis_dim, skip_row, usecols, save_plot, save_data, x_min, x_max, scale_value, shift_value, file_data, plotflag = interactive_plot_type_selection_QT()
            if not plotflag:
                sys.exit(1)
        else: 
            plot_type = interactive_plot_type_selection_TK()   

    for file_info in file_data:
        print("File:", os.path.relpath(file_info[0]))
        print("Label:", file_info[1])
        print()
    
    print(f"Plot type: {plot_type}")
    print(f"Figure title: {fig_title}") if fig_title else None
    print(f"Axis title: {axis_title}")
    print(f"Axis dimension: {axis_dim}")
    print(f"Skiped rows: {skip_row}") if skip_row else None
    print(f"Used columns: {usecols}") if usecols else None
    print(f"Scale value: {scale_value}") if scale_value != 1 else None
    print(f"Shift value: {shift_value}") if shift_value != 0 else None
    if x_min or x_max:
        print(f"x Range: {'default' if x_min is None else x_min} - {'default' if x_max is None else x_max}")
    print()

    files = [entry[0] for entry in file_data]    
    labels = [entry[1] for entry in file_data]
    
    # Parse data from all files
    parsed_data = extract_data()

    # Generate the dynamic plot and get the figure object
    fig = plot(disable_plot)

    save_func()