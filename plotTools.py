#!/usr/bin/env python3

import numpy as np
import os
import argparse
import itertools
from io import StringIO
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QWidget, QVBoxLayout, QPushButton, 
    QDialog, QLabel, QLineEdit, QCheckBox, QHBoxLayout, 
    QFrame, QMessageBox, QSplitter, QMainWindow
)
from functools import partial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys

app = QApplication(sys.argv)

def extract_general_values(file_path, skip_row, usecols):
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
                
def extract_motion_values(file_path, skip_row, usecols):
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
        'def_axis_title': 'Amplitude',
        'def_dimension': '-',
        'function': extract_general_values
    },
    'motion': {
        'label': 'Motion',
        'def_axis_title': 'Heave',
        'def_dimension': 'm',
        'function': extract_motion_values
    },
}

def extract_data(files, plot_type):
    parsed_data = []

    for file_info in files:
        file_path = file_info["path"]
        scale = file_info["scale"]
        shift = file_info["shift"]
        norm_origin = file_info["norm_origin"]
        skip_row = file_info["skip_row"]
        usecols = file_info["usecols"]

        if plot_type in fig_config:
            times, variable_data = fig_config[plot_type]['function'](file_path, skip_row, usecols)
        else:
            sys.exit(f"Unknown plot_type: {plot_type}. Available plot_type: {', '.join(fig_config.keys())}")

        # Ensure arrays have the same length
        min_length = min(len(times), len(variable_data[0]))
        times, variable_data = times[:min_length], variable_data[:min_length]

        # Apply shifts and scaling
        for i, var in enumerate(variable_data):
            var = var / scale
            var = var + shift
            variable_data[i] = var

        # Normalize to origin if needed
        if norm_origin:
            for var_array in variable_data:
                if len(var_array) > 0:
                    first_value = np.mean(var_array)
                    var_array -= first_value

        parsed_data.append((times, variable_data))

    return parsed_data

def save_plot_func(fig, individual_figures, axis_title, fig_title):
    """Saves the plot and logs the saved paths."""
    output_dir = "."

    # Log file to store saved paths
    saved_paths = []

    # Save the main figure
    sp_file = os.path.join(output_dir, f"{fig_title + '_' if fig_title else ''}figure.png")
    fig.savefig(sp_file)
    saved_paths.append(sp_file)
    print(f"Figure saved to {sp_file}")

    if not len(individual_figures) == 1:
        # Save individual figures
        ind_output_dir = "./individual_figures"
        if not os.path.exists(ind_output_dir):
            os.makedirs(ind_output_dir)
        for variable_name, fig in individual_figures.items():
            ind_sp_file = os.path.join(ind_output_dir, f"{fig_title + '_' if fig_title else ''}{variable_name}.png")
            fig.savefig(ind_sp_file)
            saved_paths.append(ind_sp_file)
            print(f"Saved {variable_name} to {ind_sp_file}")

    return saved_paths

def save_data_func(parsed_data, labels, fig_title, x_axis_title, axis_title):

    output_dir = "./extracted_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []

    # First determine the maximum number of variables across all datasets
    max_vars = max(len(variables) for _, variables in parsed_data)
    
   
    # Create one file per variable index
    for var_idx in range(max_vars):
        # Create filename for this variable index
        file_path = os.path.join(output_dir, f"{fig_title + '_' if fig_title else ''}Variable_{var_idx+1}.csv")
        title = axis_title[var_idx] if var_idx < len(axis_title) else axis_title[0]

        with open(file_path, 'w') as f:
            # Write header - Time columns for each dataset that has this variable
            headers = []
            for label, (times, variables) in zip(labels, parsed_data):
                if var_idx < len(variables):
                    headers.append(f"{x_axis_title}")
                    headers.append(f"{title}_{label}")
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
        saved_paths.append(file_path)
        print(f"Saved Variable {var_idx+1} data in: {file_path}")
    return saved_paths
        
def interactive_plot_type_selection_QT(
    plot_type, fig_title, fig_width, fig_height, x_axis_title, x_axis_dim,
    axis_title, axis_dim, x_min, x_max,
    y_min, y_max, files
):
    """
    Creates a GUI for selecting plot types and configuring plot settings.
    """

    def update_values():
        nonlocal plot_type, x_min, x_max, y_min, y_max, fig_title, x_axis_title, x_axis_dim, axis_title, axis_dim, fig_width, fig_height

        axis_title = []
        axis_dim = []

        # Validate axis_title_inputs and axis_dim_inputs
        for row_index, (title_input, dim_input) in enumerate(zip(axis_title_inputs, axis_dim_inputs)):
            title = title_input.text().strip()
            dim = dim_input.text().strip()

            # Check if one is empty while the other is filled
            if (title and not dim) or (dim and not title):
                QMessageBox.critical(
                    window,
                    "Input Error",
                    f"Row {row_index + 1}: Both 'Axis Title' and 'Dimension' must be filled if one is provided."
                )
                return  # Stop execution until the user fixes the input

            # Add valid inputs to the lists
            if title or dim:  # Only add rows where at least one field is non-empty
                axis_title.append(title)
                axis_dim.append(dim)

        # Handle other values
        fig_title = fig_title_input.text() if fig_title_input.text() else None
        x_min = float(x_min_input.text()) if x_min_input.text() else None
        x_max = float(x_max_input.text()) if x_max_input.text() else None

        y_min = float(y_min_input.text()) if y_min_input.text() else None
        y_max = float(y_max_input.text()) if y_max_input.text() else None

        # Get user-defined figure dimensions
        fig_width = float(fig_width_input.text()) if fig_width_input.text() else 4
        fig_height = float(fig_height_input.text()) if fig_height_input.text() else 4

        # Get x-axis title and dimension
        x_axis_title = x_axis_title_input.text().strip()
        x_axis_dim = x_axis_dim_input.text().strip()

    def set_plot_type(plot_value):
        nonlocal plot_type
        plot_type = plot_value

        update_values()  # Capture the latest values when plot type is selected

        # Change the style of the selected button to indicate it's selected
        for button in plot_buttons.values():
            button.setStyleSheet('background-color: none')  # Reset all buttons
        plot_buttons[plot_value].setStyleSheet('background-color: lightblue')  # Highlight the selected one
 
    def plot_button_clicked():
        if not files:
            QMessageBox.critical(window, "Error", "Please select at least one file to plot.")
            return

        update_values()
        parsed_data = extract_data(files, plot_type)

        canvas.plot(
            parsed_data,
            [file["label"] for file in files],
            x_min,
            x_max,
            y_min,
            y_max,
            fig_title,
            fig_width,
            fig_height,
            axis_title,
            axis_dim,
            x_axis_title,
            x_axis_dim
        )

        # Enable the "Save Plot" button after plotting
        write_plot_button.setEnabled(True)
        write_plot_button.setStyleSheet("""
            font-size: 15px;
            font-weight: bold;
            padding: 3px;
            color: white;
            background-color: #007BFF;
            border-radius: 3px;
        """)

        # Show the plot panel and adjust sizes
        if not plot_widget.isVisible():
            splitter.addWidget(plot_widget)  # Add the plot widget to the splitter
            splitter.setStretchFactor(0, 0)  # Keep settings_panel fixed
            splitter.setStretchFactor(1, 1)  # Allow plot_widget to stretch
            plot_widget.show()  # Show the plot widget

        # Resize the main window dynamically based on max_num_variables
        max_num_variables = max(len(item[1]) for item in parsed_data)
        rows = (max_num_variables + 1) // 2  # Calculate the number of rows (2 figures per row)
        cols = 2 if max_num_variables > 1 else 1  # Use 2 columns if more than 1 figure

        # Calculate the new window size
        new_width = (fig_width * cols * 100) + 450
        new_height = fig_height * rows * 100
        
        window.adjustSize()
        # Resize the window (convert to integers)
        window.resize(int(new_width), int(new_height))

    def write_plot_button_clicked():
        # Save the plot using the canvas's figure
        update_values()
        saved_paths = save_plot_func(canvas.fig, canvas.individual_figures, axis_title, fig_title)

        # Format the saved paths as a list for the QMessageBox
        saved_paths_list = "\n".join(saved_paths)
        QMessageBox.information(window, "Success", f"Figures saved to:\n\n{saved_paths_list}")

    def write_data_button_clicked():
        if not files:
            QMessageBox.critical(window, "Error", "Please select at least one file to process.")
            return
        
        update_values()
        parsed_data = extract_data(files, plot_type)
        saved_paths = save_data_func(parsed_data, [file["label"] for file in files], fig_title, x_axis_title, axis_title)
        
        # Format the saved paths as a list for the QMessageBox
        saved_paths_list = "\n".join(saved_paths)
        QMessageBox.information(window, "Success", f"Extraced Data saved to:\n\n{saved_paths_list}")

    def update_file_paths(updated_files):
        nonlocal files
        files = updated_files

    def add_axis_row():
        axis_row_layout = QHBoxLayout()

        # Create a counter label for the variable number
        variable_number = len(axis_title_inputs) + 1  # Determine the current row number
        variable_label = QLabel(f"Variable {variable_number}:")
        variable_label.setFixedWidth(80)  # Set a fixed width for alignment

        # Create the title and dimension labels and inputs
        axis_title_label = QLabel("Title:")
        axis_title_input = QLineEdit()
        axis_dim_label = QLabel("Dimension:")
        axis_dim_input = QLineEdit()

        # Set fixed width for input fields
        axis_title_input.setFixedWidth(140)
        axis_dim_input.setFixedWidth(50)

        # Add widgets to the horizontal layout
        axis_row_layout.addWidget(variable_label)  # Add the variable label
        axis_row_layout.addWidget(axis_title_label)
        axis_row_layout.addWidget(axis_title_input)
        axis_row_layout.addSpacing(10)
        axis_row_layout.addWidget(axis_dim_label)
        axis_row_layout.addWidget(axis_dim_input)
        axis_row_layout.addStretch()

        # Add the new row to the container layout
        axis_container_layout.addLayout(axis_row_layout)

        axis_title_input.setToolTip("Enter the title for the y-axis.")
        axis_dim_input.setToolTip("Enter the dimension for the y-axis.")

        # Keep track of the row
        axis_title_inputs.append(axis_title_input)
        axis_dim_inputs.append(axis_dim_input)

        # Connect signals to dynamically add new rows
        axis_title_input.textChanged.connect(lambda: check_and_add_row(axis_title_input, axis_dim_input))
        axis_dim_input.textChanged.connect(lambda: check_and_add_row(axis_title_input, axis_dim_input))

    def check_and_add_row(title_input, dim_input):
        if title_input.text() or dim_input.text():
            # Ensure a new row is added only once
            title_input.textChanged.disconnect()
            dim_input.textChanged.disconnect()
            add_axis_row()

    def create_section_title_with_line(title):
        """Creates a horizontal line with a title aligned to the left and a line extending to the right."""
        layout = QHBoxLayout()

        # Left line
        left_line = QFrame()
        left_line.setFrameShape(QFrame.HLine)
        left_line.setFrameShadow(QFrame.Sunken)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; padding-right: 10px;")

        # Right line
        right_line = QFrame()
        right_line.setFrameShape(QFrame.HLine)
        right_line.setFrameShadow(QFrame.Sunken)

        layout.addWidget(left_line, 1) 
        layout.addWidget(title_label, 0)  
        layout.addWidget(right_line, 1) 

        # Remove extra spacing around the layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)  # Adjust spacing between the title and lines

        return layout
     
    # Main window with QSplitter
    window = QWidget()
    window.setWindowTitle("Plot Tools (v0.1)")
    splitter = QSplitter(Qt.Horizontal)  # Horizontal splitter

    ############### Plot Setting Panel ###############
    settings_panel = QWidget()
    settings_layout = QVBoxLayout(settings_panel)
    settings_layout.setAlignment(Qt.AlignTop)  # Align all widgets to the top

    # Set a fixed width for the settings panel
    # settings_panel.setFixedWidth(420)  # Adjust this value based on your widgets
    settings_panel.setLayout(settings_layout)

    # Add only the settings panel to the splitter initially
    splitter.addWidget(settings_panel)

    title_label = QLabel("Plot Setting panel")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
    settings_layout.addWidget(title_label)

    ########### Plot Type Selection ###########
    settings_layout.addLayout(create_section_title_with_line("Plot Type"))

    plot_buttons = {key: QPushButton(config['label']) for key, config in fig_config.items()}

    for plot_value, button in plot_buttons.items():
        button.clicked.connect(lambda checked, pv=plot_value: set_plot_type(pv))
        settings_layout.addWidget(button)

    # Auto-click the plot button
    QTimer.singleShot(100, plot_buttons[plot_type].click)

    ########### Figure Properties ###########
    settings_layout.addLayout(create_section_title_with_line("Figure Properties"))

    fig_title_layout = QHBoxLayout()
    fig_title_label = QLabel("Figure title:")
    fig_title_input = QLineEdit(fig_title)
    fig_title_input.setToolTip("Enter a title for the figure.")
    fig_title_input.setFixedWidth(140)

    fig_title_layout.addWidget(fig_title_label)
    fig_title_layout.addWidget(fig_title_input)
    fig_title_layout.addStretch()

    settings_layout.addLayout(fig_title_layout)
    
    # Add input fields for "Figure Width" and "Figure Height"
    fig_size_layout = QHBoxLayout()
    fig_width_label = QLabel("Figure Width:")
    fig_width_input = QLineEdit(str(fig_width))  # Default width
    fig_width_input.setToolTip("Specify the width of the figure in inches.")
    fig_height_label = QLabel("Figure Height:")
    fig_height_input = QLineEdit(str(fig_height))  # Default height
    fig_height_input.setToolTip("Specify the height of the figure in inches.")

    fig_width_input.setFixedWidth(60)
    fig_height_input.setFixedWidth(60)

    fig_size_layout.addWidget(fig_width_label)
    fig_size_layout.addWidget(fig_width_input)
    fig_size_layout.addSpacing(10)
    fig_size_layout.addWidget(fig_height_label)
    fig_size_layout.addWidget(fig_height_input)
    fig_size_layout.addStretch()

    settings_layout.addLayout(fig_size_layout)

    ########### X-Axis Settings ###########
    settings_layout.addLayout(create_section_title_with_line("X-Axis"))

   # Add input fields for "x min" and "x max" in the same row
    x_layout = QHBoxLayout()
    range_label = QLabel("x Range:")
    x_min_input = QLineEdit(str(x_min) if x_min is not None else "")
    x_min_input.setToolTip("Enter the minimum value for the x-axis range.")
    seperator = QLabel("-   ")
    x_max_input = QLineEdit(str(x_max) if x_max is not None else "")
    x_max_input.setToolTip("Enter the maximum value for the x-axis range.")

    x_min_input.setFixedWidth(50)
    x_max_input.setFixedWidth(50)

    # Add spacing between the labels and inputs
    x_layout.addWidget(range_label)
    x_layout.addWidget(x_min_input)
    x_layout.addSpacing(10)
    x_layout.addWidget(seperator)
    x_layout.addWidget(x_max_input)
    x_layout.addStretch()  # Push everything to the left

    # Align elements in the row
    x_layout.setAlignment(Qt.AlignLeft)
    settings_layout.addLayout(x_layout)

    # Add input fields for "X-Axis Title" and "X-Axis Dimension"
    x_axis_layout = QHBoxLayout()
    x_axis_title_label = QLabel("Title:")
    x_axis_title_input = QLineEdit(x_axis_title)
    x_axis_title_input.setToolTip("Enter the title for the x-axis.")
    x_axis_dim_label = QLabel("Dimension:")
    x_axis_dim_input = QLineEdit(x_axis_dim)
    x_axis_dim_input.setToolTip("Enter the dimension for the x-axis.")

    # Set fixed width for input fields
    x_axis_title_input.setFixedWidth(140)
    x_axis_dim_input.setFixedWidth(50)

    # Add widgets to the horizontal layout
    x_axis_layout.addWidget(x_axis_title_label)
    x_axis_layout.addWidget(x_axis_title_input)
    x_axis_layout.addSpacing(10)
    x_axis_layout.addWidget(x_axis_dim_label)
    x_axis_layout.addWidget(x_axis_dim_input)
    x_axis_layout.addStretch()

    # Add the layout to the settings panel
    settings_layout.addLayout(x_axis_layout)

    ########### Y-Axis Settings ###########
    settings_layout.addLayout(create_section_title_with_line("Y-Axis"))

    # Add input fields for "y min" and "y max" in the same row
    y_layout = QHBoxLayout()
    y_range_label = QLabel("y Range:")
    y_min_input = QLineEdit(str(y_min) if y_min is not None else "")
    y_min_input.setToolTip("Enter the minimum value for the y-axis range.")
    y_separator = QLabel("-   ")
    y_max_input = QLineEdit(str(y_max) if y_max is not None else "")
    y_max_input.setToolTip("Enter the maximum value for the y-axis range.")
    y_min_input.setFixedWidth(50)
    y_max_input.setFixedWidth(50)

    # Add spacing between the labels and inputs
    y_layout.addWidget(y_range_label)
    y_layout.addWidget(y_min_input)
    y_layout.addSpacing(10)
    y_layout.addWidget(y_separator)
    y_layout.addWidget(y_max_input)
    y_layout.addStretch()  # Push everything to the left

    # Align elements in the row
    y_layout.setAlignment(Qt.AlignLeft)
    settings_layout.addLayout(y_layout)

    # Keep track of all axis rows
    axis_title_inputs = []
    axis_dim_inputs = []

    # Create a dedicated container layout for axis rows
    axis_container_layout = QVBoxLayout()
    settings_layout.addLayout(axis_container_layout)  # Add the container to the main settings layout

    # Determine the maximum length between axis_title and axis_dim
    max_length = max(len(axis_title), len(axis_dim))

    add_axis_row()

    # Set the text for each row based on the values in axis_title and axis_dim
    for i in range(max_length):
        
        title = axis_title[i] if i < len(axis_title) else "Amplitude"
        dim = axis_dim[i] if i < len(axis_dim) else "-"

        axis_title_inputs[i].setText(title)
        axis_dim_inputs[i].setText(dim)

    ########### File Selection ###########
    settings_layout.addLayout(create_section_title_with_line("File Selection"))

    # Integrate FileSelectorApp (Browse button functionality)
    file_selector = FileSelectorApp(files)
    file_selector.files_updated.connect(update_file_paths)
    settings_layout.addWidget(file_selector)

    ############### Buttons ###############
    button_layout = QHBoxLayout()
    
    plot_button = QPushButton("👁️ Plot")
    plot_button.setToolTip("Click to generate the plot.")
    plot_button.clicked.connect(plot_button_clicked)
    plot_button.setFixedSize(120, 40)
    
    write_plot_button = QPushButton("💾 Save Plot")
    write_plot_button.setToolTip("Click to save the generated plot as an image.")
    write_plot_button.setEnabled(False)
    write_plot_button.setStyleSheet("""
        font-size: 15px;
        font-weight: bold;
        padding: 3px;
        color: gray;
        background-color: #d3d3d3;
        border-radius: 3px;
    """)
    write_plot_button.clicked.connect(write_plot_button_clicked)
    write_plot_button.setFixedSize(120, 40)
    
    write_data_button = QPushButton("📊 Save Data")
    write_data_button.setToolTip("Click to save the extracted data to a file.")
    write_data_button.clicked.connect(write_data_button_clicked)
    write_data_button.setFixedSize(120, 40)
    
    # Style all buttons consistently
    for button in [plot_button, write_data_button]:
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
    settings_layout.addLayout(button_layout)

    ############### Plot Panel ###############
    plot_widget = QWidget()
    plot_layout = QVBoxLayout(plot_widget)
    canvas = PlotCanvas(plot_widget, width=5, height=4, dpi=100)
    plot_layout.addWidget(canvas)
    plot_widget.setLayout(plot_layout)
    plot_widget.hide()  # Initially hide the plot widget

    # Add the splitter to the main layout
    main_layout = QVBoxLayout(window)
    main_layout.addWidget(splitter)
    window.setLayout(main_layout)

    # Show the main window
    window.adjustSize()
    window.show()
    if files:
        plot_button.click()
    app.exec_()

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axs = []  # Store axes for dynamic plotting
        self.individual_figures = {}  # Dictionary to store individual figures

        # Variables for dragging the title and legends
        self.title = None
        self.dragging_title = False
        self.dragging_legend = None  # Track which legend is being dragged
        self.last_mouse_pos = None

        # Connect mouse events
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("button_release_event", self.on_release)

    def plot(self, parsed_data, labels, x_min, x_max, y_min, y_max, fig_title, fig_width, fig_height, axis_title, axis_dim, x_axis_title, x_axis_dim):
        self.fig.clear()  # Clear the main figure for new plots
        colors = ['b', 'r', 'g', 'c', 'm', 'y']
        linestyles = ['-', '--', ':', '-.']

        max_num_variables = max(len(item[1]) for item in parsed_data)

        # Dynamically determine the layout based on max_num_variables
        rows = (max_num_variables + 1) // 2  # Calculate rows (2 figures per row)
        cols = 2 if max_num_variables > 1 else 1  # Use 2 columns if more than 1 figure

        # Calculate the overall figure size
        figsize = (fig_width * cols, fig_height * rows)

        # Resize the figure dynamically
        self.fig.set_size_inches(figsize)

        # Create subplots dynamically
        self.axs = self.fig.subplots(rows, cols, squeeze=False).flatten()

        for i in range(max_num_variables):
            # Create color and linestyle cycles once
            color_cycle = itertools.cycle(colors)
            linestyle_cycle = itertools.cycle(linestyles)
            fig_individual = Figure(figsize=(fig_width, fig_height), dpi=100)
            ax_individual = fig_individual.add_subplot(111)

            for (times, variables), label in zip(parsed_data, labels):
                color = next(color_cycle)
                linestyle = next(linestyle_cycle)
                self.axs[i].plot(times, variables[i], label=label, linestyle=linestyle, linewidth=1, color=color)
                ax_individual.plot(times, variables[i], label=label, linestyle=linestyle, linewidth=1, color=color)
                
            title = axis_title[i] if i < len(axis_title) else axis_title[0]
            dim = axis_dim[i] if i < len(axis_dim) else axis_dim[0]

            self.axs[i].set_xlabel(fr"{x_axis_title} (${x_axis_dim}$)")
            self.axs[i].set_ylabel(fr"{title} (${dim}$)")
            self.axs[i].set_xlim(
                left=x_min if x_min is not None else self.axs[i].get_xlim()[0],
                right=x_max if x_max is not None else self.axs[i].get_xlim()[1]
            )
            self.axs[i].set_ylim(
            bottom=y_min if y_min is not None else self.axs[i].get_ylim()[0],
            top=y_max if y_max is not None else self.axs[i].get_ylim()[1]
)
            legend = self.axs[i].legend()
            legend.set_draggable(True)  # Enable draggable legends

            ax_individual.set_xlabel(self.axs[i].get_xlabel())
            ax_individual.set_ylabel(self.axs[i].get_ylabel())
            ax_individual.set_xlim(self.axs[i].get_xlim())
            legend = self.axs[i].get_legend()
            if legend is not None:
                handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]
                ax_individual.legend(handles, labels)
            fig_individual.tight_layout()
            # Store the individual figure in the dictionary
            self.individual_figures[f"Variable_{i+1}_{title}"] = fig_individual

        # Hide unused subplots
        for j in range(max_num_variables, len(self.axs)):
            self.fig.delaxes(self.axs[j])

        # Add the figure title
        self.title = self.fig.suptitle(fig_title, fontsize=14)

        # Reduce margins around the figure
        self.fig.tight_layout()

        self.draw()  # Render the updated plot

    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes is None:
            # Check if the title is clicked
            if self.title is not None:
                bbox = self.title.get_window_extent(renderer=self.get_renderer())
                if bbox.contains(event.x, event.y):
                    self.dragging_title = True
                    self.last_mouse_pos = (event.x, event.y)
                    return

            # Check if any legend is clicked
            for ax in self.axs:
                legend = ax.get_legend()
                if legend is not None:
                    bbox = legend.get_window_extent(renderer=self.get_renderer())
                    if bbox.contains(event.x, event.y):
                        self.dragging_legend = legend
                        self.last_mouse_pos = (event.x, event.y)
                        return

    def on_motion(self, event):
        """Handle mouse motion events."""
        if self.last_mouse_pos is None:
            return

        dx = event.x - self.last_mouse_pos[0]
        dy = event.y - self.last_mouse_pos[1]
        self.last_mouse_pos = (event.x, event.y)

        if self.dragging_title and self.title is not None:
            # Update the title's position
            current_x, current_y = self.title.get_position()
            new_x = current_x + (dx / self.fig.bbox.width)
            new_y = current_y + (dy / self.fig.bbox.height)
            self.title.set_position((new_x, new_y))
            self.draw()  # Redraw the figure

        elif self.dragging_legend is not None:
            # Update the legend's position
            legend = self.dragging_legend
            current_x, current_y = legend.get_bbox_to_anchor()._bbox.x0, legend.get_bbox_to_anchor()._bbox.y0
            new_x = current_x + (dx / self.fig.bbox.width)
            new_y = current_y + (dy / self.fig.bbox.height)
            legend.set_bbox_to_anchor((new_x, new_y))
            self.draw()  # Redraw the figure

    def on_release(self, event):
        """Handle mouse release events."""
        self.dragging_title = False
        self.dragging_legend = None
        self.last_mouse_pos = None

class FileSelectorApp(QWidget):
    files_updated = pyqtSignal(list)  # Emits the entire files list

    def __init__(self, files=None):
        super().__init__()
        self.files = files if files else []  # Initialize with provided files or an empty list
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("File Selector")
        self.layout = QVBoxLayout()

        # Populate rows with the provided files
        for file_info in self.files:
            self.add_file_row(file_info)

        # Add an empty row for user input
        self.add_file_row()
        self.setLayout(self.layout)

    def add_file_row(self, file_info=None):
        # Check if the last row is empty
        if not file_info and self.layout.count() > 0:
            last_row = self.layout.itemAt(self.layout.count() - 1)
            if isinstance(last_row, QHBoxLayout):
                file_path_box = last_row.itemAt(1).widget()
                label_input = last_row.itemAt(3).widget()
                if file_path_box.text().strip() == "" and label_input.text().strip() == "":
                    # If the last row is empty, do not add a new row
                    return

        # Create a new file row
        file_row_layout = QHBoxLayout()

        # Add a fixed-width spacer for the close button
        close_button_spacer = QLabel()  # Empty QLabel as a spacer
        close_button_spacer.setFixedSize(20, 20)  # Same size as the close button

        browse_button = QPushButton("📂")
        browse_button.setFixedSize(30, 30)
        browse_button.setToolTip("Click to browse and select a file.")

        file_path_box = QLineEdit()
        file_path_box.setReadOnly(True)
        file_path_box.setToolTip("Displays the selected file path.")
        file_path_box.setProperty("settings", {
            "scale": 1,
            "shift": 0,
            "norm_origin": False,
            "skip_row": 0,
            "usecols": None
        })  # Initialize default settings

        label_title = QLabel("Label:")

        label_input = QLineEdit()
        label_input.setPlaceholderText("Enter label")
        label_input.setToolTip("Enter a label for the selected file.")

        settings_button = QPushButton("⚙️ Settings")
        settings_button.setFixedSize(80, 30)
        settings_button.setToolTip("Click to configure file-specific settings.")

        close_button = QPushButton("❌")  # Add a close button
        close_button.setFixedSize(20, 20)
        close_button.setToolTip("Click to remove this file row.")
        close_button.setStyleSheet("""
        QPushButton {
            background-color: transparent;  /* No background */
            border: none;                   /* No border */
            font-size: 14px;                /* Adjust font size for the cross symbol */
            color: black;                   /* Cross color */
        }
        QPushButton:hover {
            color: red;                     /* Change color on hover */
        }
        """)
        close_button.hide()  # Initially hide the close button

        # Populate the row with file info if provided
        if file_info:
            file_path_box.setText(os.path.basename(file_info["path"]))
            file_path_box.setProperty("full_path", file_info["path"])
            label_input.setText(file_info["label"])
            file_path_box.setProperty("settings", {
                "scale": file_info.get("scale", 1),
                "shift": file_info.get("shift", 0),
                "norm_origin": file_info.get("norm_origin", False),
                "skip_row": file_info.get("skip_row", 0),
                "usecols": file_info.get("usecols", None),
            })
            close_button.show()  # Show the close button for pre-filled rows
            close_button_spacer.hide()

        # Connect the settings button to open the settings dialog
        settings_button.clicked.connect(lambda: self.open_settings_dialog(file_path_box))
        label_input.textChanged.connect(lambda: self.update_values(file_path_box, label_input))

        browse_button.clicked.connect(partial(self.select_file, file_path_box, label_input, close_button, close_button_spacer))

        # Connect the close button to remove the row
        def remove_row():
            for i in reversed(range(file_row_layout.count())):
                widget = file_row_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            self.layout.removeItem(file_row_layout)
            self.update_values()  # Update the file list after removing a row

        close_button.clicked.connect(remove_row)

        # Add widgets to the horizontal layout
        file_row_layout.addWidget(browse_button)
        file_row_layout.addWidget(file_path_box)
        file_row_layout.addWidget(label_title)
        file_row_layout.addWidget(label_input)
        file_row_layout.addWidget(settings_button)
        file_row_layout.addWidget(close_button_spacer)  # Add the spacer for the close button
        file_row_layout.addWidget(close_button)
        self.layout.addLayout(file_row_layout)

    def select_file(self, file_path_box, label_input, close_button, close_button_spacer, scale_input=None, shift_input=None, norm_origin_checkbox=None, skip_row_input=None, usecols_input=None):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        if file_path:
            file_name = os.path.basename(file_path)
            dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
            file_path_box.setText(file_name)
            file_path_box.setProperty("full_path", file_path)
            label_input.setPlaceholderText(dir_name)
            close_button.show()  # Show the close button when a file is selected
            close_button_spacer.hide()
            self.update_values()
            self.add_file_row()

    def update_values(self, file_path_box=None, label_input=None):
        self.files = []

        for file_row in self.layout.children():
            if isinstance(file_row, QHBoxLayout):
                current_file_box = file_row.itemAt(1).widget()
                current_label_input = file_row.itemAt(3).widget()

                file_path = current_file_box.property("full_path")
                label = current_label_input.text().strip() or current_label_input.placeholderText()
                settings = current_file_box.property("settings") or {}

                # Ensure all required keys are present with default values
                settings = {
                    "scale": settings.get("scale", 1),
                    "shift": settings.get("shift", 0),
                    "norm_origin": settings.get("norm_origin", False),
                    "skip_row": settings.get("skip_row", 0),
                    "usecols": settings.get("usecols", None),
                }

                if file_path:
                    self.files.append({
                        "path": file_path,
                        "label": label,
                        **settings  # Include the settings directly
                    })
        self.files_updated.emit(self.files)

    def open_settings_dialog(self, file_path_box):
        file_path = file_path_box.property("full_path")
        if not file_path:
            QMessageBox.warning(self, "Warning", "File path is not set for this row.")
            return

        # Get the current settings for this file row
        current_settings = file_path_box.property("settings")

        dialog = SettingsDialog(
            self,
            scale=current_settings["scale"],
            shift=current_settings["shift"],
            skip_row=current_settings["skip_row"],
            usecols=",".join(map(str, current_settings["usecols"])) if current_settings["usecols"] else "",
            normalize=current_settings["norm_origin"]
        )

        if dialog.exec_() == QDialog.Accepted:
            # Update the settings for this file row
            new_settings = dialog.get_settings()
            file_path_box.setProperty("settings", new_settings)

            # Emit the updated files list
            self.update_values()

class SettingsDialog(QDialog):
    def __init__(self, parent=None, scale=1, shift=0, skip_row=0, usecols="", normalize=False):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(300, 250)

        self.scale = scale
        self.shift = shift
        self.skip_row = skip_row
        self.usecols = usecols
        self.normalize = normalize

        layout = QVBoxLayout()

        # Scale
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Scale:")
        self.scale_input = QLineEdit(str(self.scale))
        self.scale_input.setPlaceholderText("e.g., 1.0")  # Add placeholder text
        self.scale_input.setToolTip("Specify the scale factor to divide the data (e.g., 1.0 for no scaling).")
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_input)
        layout.addLayout(scale_layout)

        # Shift
        shift_layout = QHBoxLayout()
        shift_label = QLabel("Shift:")
        self.shift_input = QLineEdit(str(self.shift))
        self.shift_input.setPlaceholderText("e.g., 0.0")  # Add placeholder text
        self.shift_input.setToolTip("Specify the shift value to add to the data (e.g., 0.0 for no shift).")
        shift_layout.addWidget(shift_label)
        shift_layout.addWidget(self.shift_input)
        layout.addLayout(shift_layout)

        # Skip Rows
        skip_row_layout = QHBoxLayout()
        skip_row_label = QLabel("Skip Rows:")
        self.skip_row_input = QLineEdit(str(self.skip_row))
        self.skip_row_input.setPlaceholderText("e.g., 0")  # Add placeholder text
        self.skip_row_input.setToolTip("Specify the number of rows to skip at the start of the file (e.g., 0 for no skipping).")
        skip_row_layout.addWidget(skip_row_label)
        skip_row_layout.addWidget(self.skip_row_input)
        layout.addLayout(skip_row_layout)

        # Used Columns
        usecols_layout = QHBoxLayout()
        usecols_label = QLabel("Used Columns:")
        self.usecols_input = QLineEdit(self.usecols)
        self.usecols_input.setPlaceholderText("e.g., 0,1,3")
        self.usecols_input.setToolTip("Specify the column indices to use, separated by commas (e.g., 0,1,3).")
        usecols_layout.addWidget(usecols_label)
        usecols_layout.addWidget(self.usecols_input)
        layout.addLayout(usecols_layout)

        # Normalize
        normalize_layout = QHBoxLayout()
        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setChecked(self.normalize)
        self.normalize_checkbox.setToolTip("Enable this option to normalize the data to its origin (subtract the mean of the data).")
        normalize_layout.addWidget(self.normalize_checkbox)
        layout.addLayout(normalize_layout)

        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_settings(self):
        return {
            "scale": float(self.scale_input.text()) if self.scale_input.text() else 1,
            "shift": float(self.shift_input.text()) if self.shift_input.text() else 0,
            "skip_row": int(self.skip_row_input.text()) if self.skip_row_input.text() else 0,
            "usecols": [int(col) for col in self.usecols_input.text().split(",")] if self.usecols_input.text() else None,
            "norm_origin": self.normalize_checkbox.isChecked()
        }
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process files.")
    parser.add_argument('file_args', nargs=argparse.REMAINDER,
                        help="List of input files followed by their options: -label, -scale, -shift, -norm_origin, -skip_row, -usecols")
    parser.add_argument('-plot_type', '-pt', default="general", help=f"Specify the Type of plot: {', '.join(fig_config.keys())}")
    parser.add_argument('-fig_title', '-ft', default=None, help="Specify a Title for the figure")
    parser.add_argument('-fig_width', '-fw', type=float, default=5, help="Specify the width of the figure in inches")
    parser.add_argument('-fig_height', '-fh', type=float, default=4, help="Specify the height of the figure in inches")
    parser.add_argument('-x_axis_title', '-xat', default="t", help="Specify a Title for the x Axis")
    parser.add_argument('-x_axis_dim', '-xad', default="s", help="Specify a Dimension for the x Axis")
    parser.add_argument('-y_axis_title', '-yat', default="Amplitude", help="Specify Titles for the y Axes (comma-separated for multiple)")
    parser.add_argument('-y_axis_dim', '-yad', default="-", help="Specify Dimensions for the y Axes (comma-separated for multiple)")
    parser.add_argument('-x_min', '-xmi', type=float, help="Minimum x-axis value")
    parser.add_argument('-x_max', '-xma', type=float, help="Maximum x-axis value")
    parser.add_argument('-y_min', '-ymi', type=float, help="Minimum y-axis value")
    parser.add_argument('-y_max', '-yma', type=float, help="Maximum y-axis value")
    parser.add_argument('-save_plot', '-sp', action='store_true', help="Enable saving the plot as a PNG image")
    parser.add_argument('-save_data', '-sd', action='store_true', help="Enable saving the extracted data to CSV files")
    args = parser.parse_args()

    y_axis_title_list = args.y_axis_title.split(',')
    y_axis_dim_list = args.y_axis_dim.split(',')   

    files = []
    i = 0
    while i < len(args.file_args):
        file_path = args.file_args[i]
        dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
        label = f"{dir_name}"
        scale = 1
        shift = 0
        norm_origin = False
        skip_row = 0
        usecols = None
        i += 1

        while i < len(args.file_args) and args.file_args[i].startswith('-'):
            if args.file_args[i] == "-label": label = args.file_args[i + 1]; i += 2
            elif args.file_args[i] == "-scale": scale = float(args.file_args[i + 1]); i += 2
            elif args.file_args[i] == "-shift": shift = float(args.file_args[i + 1]); i += 2
            elif args.file_args[i] == "-norm_origin": norm_origin = True; i += 1
            elif args.file_args[i] == "-skip_row": skip_row = int(args.file_args[i + 1]); i += 2
            elif args.file_args[i] == "-usecols": usecols = [int(col) for col in args.file_args[i + 1].split(",")]; i += 2
            else: break

        files.append({
            "path": file_path,
            "label": label,
            "scale": scale,
            "shift": shift,
            "norm_origin": norm_origin,
            "skip_row": skip_row,
            "usecols": usecols
        })
    
    return (
        args.plot_type, args.fig_title, args.fig_width, args.fig_height, args.x_axis_title, args.x_axis_dim,
        y_axis_title_list,y_axis_dim_list , args.x_min, args.x_max,
        args.y_min, args.y_max, args.save_plot, args.save_data, files
    )

if __name__ == "__main__":
    # Ensure QApplication is created only once
    app = QApplication.instance() or QApplication(sys.argv)

    (
        plot_type, fig_title,fig_width, fig_height,
        x_axis_title, x_axis_dim,
        y_axis_title, y_axis_dim,
        x_min, x_max, y_min, y_max, save_plot, save_data, files
    ) = parse_arguments()

    if files and (save_data or save_plot) :
        for file in files:
            print("File:", os.path.relpath(file["path"]))
            print("Label:", file["label"])
            print(f"Scale value: {file['scale']}") if file["scale"] != 1 else None
            print(f"Shift value: {file['shift']}") if file["shift"] != 0 else None
            print(f"Normalized to the origin") if file["norm_origin"] else None
            print(f"Skipped rows: {file['skip_row']}") if file["skip_row"] else None
            print(f"Used columns: {file['usecols']}") if file["usecols"] else None
            print()
        
        print(f"Plot type: {plot_type}")
        print(f"Figure title: {fig_title}") if fig_title else None
        print(f"X Axis title: {x_axis_title}")
        print(f"X Axis dimension: {x_axis_dim}")
        print(f"Y Axis titles: {y_axis_title}")
        print(f"Y Axis dimensions: {y_axis_dim}")
        if x_min or x_max:
            print(f"x Range: {'default' if x_min is None else x_min} - {'default' if x_max is None else x_max}")
            print(f"y Range: {'default' if y_min is None else y_min} - {'default' if y_max is None else y_max}")
        print()

        parsed_data = extract_data(files, plot_type)

        if save_plot:
            canvas = PlotCanvas(None, width=5, height=4, dpi=100)
            canvas.plot(
            parsed_data,
            [file["label"] for file in files],
            x_min, x_max, y_min, y_max, fig_title, fig_width, fig_height,
            y_axis_title, y_axis_dim,
            x_axis_title=x_axis_title,
            x_axis_dim=x_axis_dim
        )
            save_plot_func(canvas.fig, canvas.individual_figures, y_axis_title, fig_title)
        if save_data:
            save_data_func(parsed_data, [file["label"] for file in files], fig_title, x_axis_title, y_axis_title)

    else:
        interactive_plot_type_selection_QT (
        plot_type, fig_title,fig_width, fig_height,
        x_axis_title, x_axis_dim,
        y_axis_title, y_axis_dim,
        x_min, x_max, y_min, y_max, files
        )