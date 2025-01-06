#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import itertools
from io import StringIO
import tkinter as tk
try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QVBoxLayout, QPushButton, QDialog, QLabel, QLineEdit, QDesktopWidget, QCheckBox, QHBoxLayout, QFrame, QTextEdit
    import sys
    import os
    pyqt_available = True
    from functools import partial
except ImportError:
    pyqt_available = False
    import tkinter as tk
    from tkinter import filedialog

app = QApplication(sys.argv)

def extract_interfaceHeight_values(file_path):
    """
    Reads data from the given file, skipping the first 5 comment lines and
    extracting the Time (column 1) and the interfaceHeight values.
    """
    # Returns the directory part of a path, handling both file and directory inputs
    #file_path = file_path if os.path.isdir(file_path) else os.path.dirname(file_path)    
    #file_path=f"{file_path}/postProcessing/interfaceHeight/0/height.dat"
    data = np.loadtxt(file_path, delimiter=None, skiprows=0, usecols=(0 ,1 ,3 ,5 ,7))
    data=data.T
    times = data[0]
    variable_data = [data[1],data[2],data[3],data[4]]
    return times, variable_data
    
def extract_force_values(file_path):
    """
    Reads data from the given file, skipping the first 4 comment lines and
    extracting the Time (column 1) and the force values.
    """
    # Returns the directory part of a path, handling both file and directory inputs
    #file_path = file_path if os.path.isdir(file_path) else os.path.dirname(file_path)    
    #file_path=f"{file_path}/postProcessing/forcesHull/0/force.dat"
    with open(file_path, "r") as infile:
        cleaned_data = "".join(line.replace("(", "").replace(")", "") for line in infile)
    cleaned_file = StringIO(cleaned_data)
    data = np.loadtxt(cleaned_file, delimiter=None, skiprows=4, usecols=(0, 1, 2, 3))
    data=data.T
    times = data[0]
    variable_data = [data[1],data[2],data[3]]
    return times, variable_data

def extract_flux_values(file_path):
    """
    Reads data from the given file, skipping the first 5 comment lines and
    extracting the Time (column 1) and the flux values.
    """
    # Returns the directory part of a path, handling both file and directory inputs
    #file_path = file_path if os.path.isdir(file_path) else os.path.dirname(file_path)    
    #file_path=f"{file_path}/postProcessing/fluxSummary/0/fl.dat"
    data = np.loadtxt(file_path, delimiter=None, skiprows=5, usecols=(0, 3, 4))
    data=data.T
    times = data[0]
    variable_data = [data[1],data[2]]
    return times, variable_data
                
def extract_motion_values(log_file_path):
    """
    Extracts Time and variable values from the log file.
    """
    times = []
    variable_data = []

    previous_line = ""  # Keep track of the previous line
    with open(log_file_path, 'r') as file:
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

def select_fig(file_path, fig_type):
    """
    Parses a file to extract Time and variable data.
    """
    if fig_type == 'motion':
        times, variable_data = extract_motion_values(file_path)
        figures = [
    	    # figure Name, Dimension
    	    ["Heave", r"$cm$"  ],
    	    ["Roll", r"$deg$"  ],
    	    ["Surge" , r"$cm$" ],
    	    ["Pitch", r"$deg$" ],
    	    ["Yaw"  , r"$deg$" ],
    	    ["Sway" , r"$cm$"  ]
        ]	        
    elif fig_type == 'flux':       
        times, variable_data = extract_flux_values(file_path)
        figures = [
    	    # figure Name, Dimension
    	    ["Flux Net", r'$m^3/t$' ],
    	    ["Flux Abs", r'$m^3/t$' ]
        ]
    elif fig_type == 'force':   
        times, variable_data = extract_force_values(file_path)
        figures = [
    	    # figure Name, Dimension
    	    ["x-Force", r'$N$' ],
    	    ["y-Force", r'$N$' ],
    	    ["z-Force", r'$N$' ]
        ]
    elif fig_type == 'interfaceHeight':   
        times, variable_data = extract_interfaceHeight_values(file_path)
        figures = [
    	    # figure Name, Dimension
    	    ["Amplitude (Gauge 1)", r'$m$' ],
    	    ["Amplitude (Gauge 2)", r'$m$' ],
    	    ["Amplitude (Gauge 3)", r'$m$' ],
    	    ["Amplitude (Gauge 4)", r'$m$' ],
    	    ["Amplitude (Gauge 5)", r'$m$' ],
    	    ["Amplitude (Gauge 6)", r'$m$' ],
    	    ["Amplitude (Gauge 7)", r'$m$' ]
        ]         
    else:
        raise ValueError(f"Unknown fig_type: {fig_type} Available fig_type: motion, flux, force, interfaceHeight")
        
    figure_names = [entry[0] for entry in figures]
    dims = [entry[1] for entry in figures]
        
    return times, variable_data, figure_names, dims

def save_extracted_data(directory, label, figure_name, times, data):
    """
    Saves extracted data to a file in the specified directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f"{label}_{figure_name}.csv")
    with open(file_path, 'w') as f:
        f.write(f"Time, {figure_name}\n")
        for t, d in zip(times, data):
            f.write(f"{t}, {d}\n")

def dynamic_plot(files, labels, fig_type, x_min, x_max, save_plot=False, save_data=False):
    """
    Creates dynamic plots for an arbitrary number of variables extracted from multiple files.
    """
    # Parse data from all files
    parsed_data = []
    max_num_variables = 0

    colors = ['b', 'r', 'g', 'c', 'm', 'y']  # color cycle
    linestyles = ['-', '--', ':', '-.']  # linestyle cycle
    
    for i, file in enumerate(files):
        times, variables, figure_names, dims = select_fig(file, fig_type)
        max_num_variables = max(max_num_variables, len(variables))
       
        # Ensure arrays have the same length
        min_length = min(len(times), len(variables[0]))
        times, variables = times[:min_length], variables[:min_length]

        # # Apply shifts and scaling to each variable if needed
        # if i == 0 or i== 1: //select file(s) to apply shift and scale
        #     shift_value=0
        #     scale_value=1       
        #     for i, var in enumerate(variables):
        #         if i == 0 or i == 2:  //select plot(s) to apply shift and scale
        #             var = var / scale_value  # Apply scale        
        #         if i == 0 or i == 2: 
        #             var = var + shift_value  # Apply shift             
        #         variables[i] = var

        parsed_data.append((times, variables))
    
    # Create subplots dynamically
    fig, axs = plt.subplots(1, max_num_variables, figsize=(5 * max_num_variables, 5))
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
        axs[i].set_title(figure_names[i])       
        axs[i].set_xlabel(r"t ($s$)")
        axs[i].set_ylabel(f"{figure_names[i]} ({dims[i]})")
        axs[i].set_xlim(
            left=x_min if x_min is not None else axs[i].get_xlim()[0],
            right=x_max if x_max is not None else axs[i].get_xlim()[1]
        )
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_plot:
        output_dir = "."
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{fig_type}_comparison.png"))
        print(f"Figure saved to {output_dir}/{fig_type}_comparison.png")
    plt.show()

    # Save extracted data if enabled
    if save_data:
        for file, (times, variables), label in zip(files, parsed_data, labels):
            base_dir = os.path.dirname(file)
            output_dir = os.path.join(base_dir, "extractedData")
            for i, var in enumerate(variables):
                save_extracted_data(output_dir, label, figure_names[i], times, var)
            print(f"Data saved for {label} in {output_dir}")

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
    file_data = []

    def update_values():
        nonlocal plot_type, save_plot, save_data, x_min, x_max
        # Update the current values from the UI elements
        save_plot = save_plot_checkbox.isChecked()
        save_data = save_data_checkbox.isChecked()
        x_min = float(x_min_input.text()) if x_min_input.text() else None
        x_max = float(x_max_input.text()) if x_max_input.text() else None

    def set_plot_type(plot_value):
        nonlocal plot_type
        plot_type = plot_value
        update_values()  # Capture the latest values when plot type is selected

        # Change the style of the selected button to indicate it's selected
        for button in plot_buttons.values():
            button.setStyleSheet('background-color: none')  # Reset all buttons
        plot_buttons[plot_value].setStyleSheet('background-color: lightblue')  # Highlight the selected one

    def plot_button_clicked():
        # Update values before closing the window
        update_values()
        window.close()

    def update_file_paths(file_paths):
        nonlocal file_data  # Use nonlocal to modify the variable
        file_data = file_paths
        
    window = QWidget()
    window.setWindowTitle("Plot Setting")
    layout = QVBoxLayout()

    # Add a title before the buttons
    title_label = QLabel("Select the Plot Type")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
    layout.addWidget(title_label)

    # Buttons for each plot type with toggling functionality
    plot_buttons = {
        "motion": QPushButton("motion"),
        "flux": QPushButton("flux"),
        "force": QPushButton("force"),
        "interfaceHeight": QPushButton("interfaceHeight")
    }

    for plot_value, button in plot_buttons.items():
        button.clicked.connect(lambda checked, pv=plot_value: set_plot_type(pv))
        layout.addWidget(button)

    # Add the first horizontal line separator
    line1 = QFrame()
    line1.setFrameShape(QFrame.HLine)
    line1.setFrameShadow(QFrame.Sunken)
    layout.addWidget(line1)

    # Add checkboxes for "Save Plot" and "Save Data" in the same row
    save_layout = QHBoxLayout()
    save_plot_checkbox = QCheckBox("Save Plot")
    save_data_checkbox = QCheckBox("Save Data")
    save_layout.addWidget(save_plot_checkbox)
    save_layout.addWidget(save_data_checkbox)
    save_layout.setAlignment(Qt.AlignLeft)  # Align checkboxes to the left
    layout.addLayout(save_layout)

    # Add the second horizontal line separator
    line2 = QFrame()
    line2.setFrameShape(QFrame.HLine)
    line2.setFrameShadow(QFrame.Sunken)
    layout.addWidget(line2)

    # Add input fields for "x min" and "x max" in the same row
    x_layout = QHBoxLayout()
    x_min_label = QLabel("x min:")
    x_min_input = QLineEdit()
    x_max_label = QLabel("x max:")
    x_max_input = QLineEdit()

    # Set fixed width for input boxes
    x_min_input.setFixedWidth(50)
    x_max_input.setFixedWidth(50)

    x_layout.addWidget(x_min_label)
    x_layout.addWidget(x_min_input)
    x_layout.addWidget(x_max_label)
    x_layout.addWidget(x_max_input)
    layout.addLayout(x_layout)

    # Add the third horizontal line separator
    line3 = QFrame()
    line3.setFrameShape(QFrame.HLine)
    line3.setFrameShadow(QFrame.Sunken)
    layout.addWidget(line3)

    # Integrate FileSelectorApp (Browse button functionality)
    file_selector = FileSelectorApp()
    file_selector.files_data_updated.connect(update_file_paths)
    layout.addWidget(file_selector)

    # Add the final "Plot" button to execute selection
    plot_button = QPushButton("Plot")
    plot_button.clicked.connect(plot_button_clicked)
    layout.addWidget(plot_button)

    # Set the layout and show the window
    window.setLayout(layout)
    window.resize(400, 300)
    window.show()

    # Center the window on the screen
    screen_center = QDesktopWidget().availableGeometry().center()
    window_rect = window.frameGeometry()
    window_rect.moveCenter(screen_center)
    window.move(window_rect.center())

    # Execute the application
    app.exec_()

    return plot_type, save_plot, save_data, x_min, x_max, file_data

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

                file_path = file_path_box.text().strip()
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
    parser.add_argument('-plot_type', '-pt', type=str,
                       help='Specify the type of plot (motion, flux, force, interfaceHeight)'
                       )
    parser.add_argument('-save_plot', '-sp', action='store_true', help="Disable saving the plot (default: False)")
    parser.add_argument('-save_data', '-sd', action='store_true', help="Disable saving the data (default: False)")
    parser.add_argument('-x_min', type=float, help="Minimum x-axis value")
    parser.add_argument('-x_max', type=float, help="Maximum x-axis value")

    # Parse known arguments first
    args = parser.parse_args()

    # If Plot Type and no file are not provided, ask in interactive input
    if  args.plot_type and args.file_args:
        plot_type = args.plot_type
        save_plot = args.save_plot
        save_data = args.save_data
        x_min = args.x_min
        x_max = args.x_max   

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
    else:
        # plot_type = input("Enter the plot type (motion, flux, force, interfaceHeight): ")
        if pyqt_available:

            plot_type, save_plot, save_data, x_min, x_max, file_data = interactive_plot_type_selection_QT()

        else:
            save_plot = args.save_plot
            save_data = args.save_data
            x_min = args.x_min
            x_max = args.x_max   
            plot_type = interactive_plot_type_selection_TK()

    return plot_type, save_plot, save_data, x_min, x_max, file_data


if __name__ == "__main__":
    
#    file_data = [
#          # File, label
#         ["./OverSet/log.interFoam","OverSet"],
#         ["./BodyFitted/log.interFoam","BodyFitted"]      
#    ]
      
    plot_type, save_plot, save_data, x_min, x_max, file_data = parse_arguments()

    for file_info in file_data:
        print("File:", os.path.relpath(file_info[0]))
        print("Label:", file_info[1])
        print()
    print(f"Plot type: {plot_type}")
    print(f"Save Plot: {save_plot}")
    print(f"Save Data: {save_data}")
    print(f"x Range: {'default' if x_min is None else x_min} - {'default' if x_max is None else x_max}")
    print()

    files = [entry[0] for entry in file_data]    
    labels = [entry[1] for entry in file_data]
            
    # Plot and save data
    dynamic_plot(files, labels, plot_type, x_min, x_max, save_plot, save_data)
