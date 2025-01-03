#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import itertools
from io import StringIO
import tkinter as tk
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QVBoxLayout, QPushButton, QDialog, QLabel, QLineEdit, QDesktopWidget, QCheckBox, QHBoxLayout, QFrame
    import sys
    pyqt_available = True
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
    	    # figure Name, Dimension, shift_flags, scale_flags
    	    ["Heave", r"$cm$" , 1, 1],
    	    ["Roll", r"$deg$" , 0, 1],
    	    ["Surge" , r"$cm$", 0 ,0],
    	    ["Pitch", r"$deg$", 0 ,0],
    	    ["Yaw"  , r"$deg$", 0 ,0],
    	    ["Sway" , r"$cm$" , 0 ,0]
        ]	        
    elif fig_type == 'flux':       
        times, variable_data = extract_flux_values(file_path)
        figures = [
    	    # figure Name, Dimension, shift_flags, scale_flags
    	    ["Flux Net", r'$m^3/t$' , 1, 1],
    	    ["Flux Abs", r'$m^3/t$' , 1, 1]
        ]
    elif fig_type == 'force':   
        times, variable_data = extract_force_values(file_path)
        figures = [
    	    # figure Name, Dimension, shift_flags, scale_flags
    	    ["x-Force", r'$N$' , 1, 1],
    	    ["y-Force", r'$N$' , 1, 1],
    	    ["z-Force", r'$N$' , 1, 1]
        ]
    elif fig_type == 'interfaceHeight':   
        times, variable_data = extract_interfaceHeight_values(file_path)
        figures = [
    	    # figure Name, Dimension, shift_flags, scale_flags
    	    ["Amplitude (Gauge 1)", r'$m$' , 1, 1],
    	    ["Amplitude (Gauge 2)", r'$m$' , 1, 1],
    	    ["Amplitude (Gauge 3)", r'$m$' , 1, 1],
    	    ["Amplitude (Gauge 4)", r'$m$' , 1, 1],
    	    ["Amplitude (Gauge 5)", r'$m$' , 1, 1],
    	    ["Amplitude (Gauge 6)", r'$m$' , 1, 1],
    	    ["Amplitude (Gauge 7)", r'$m$' , 1, 1]
        ]         
    else:
        raise ValueError(f"Unknown fig_type: {fig_type} Available fig_type: motion, flux, force, interfaceHeight")
        
    figure_names = [entry[0] for entry in figures]
    dims = [entry[1] for entry in figures]    
    shift_flags = [entry[2] for entry in figures]
    scale_flags = [entry[3] for entry in figures]
        
    return times, variable_data, figure_names, dims, shift_flags, scale_flags

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

def dynamic_plot(files, labels, fig_type, x_min, x_max, save_plot=False, save_data=False, shift_values=None, scale_values=None):
    """
    Creates dynamic plots for an arbitrary number of variables extracted from multiple files.
    """
    # Parse data from all files
    parsed_data = []
    max_num_variables = 0

    colors = ['b', 'r', 'g', 'c', 'm', 'y']  # color cycle
    linestyles = ['-', '--', ':', '-.']  # linestyle cycle
    
    for i, file in enumerate(files):
        times, variables, figure_names, dims, shift_flags, scale_flags = select_fig(file, fig_type)
        max_num_variables = max(max_num_variables, len(variables))
       
        # Ensure arrays have the same length
        min_length = min(len(times), len(variables[0]))
        times, variables = times[:min_length], variables[:min_length]

        # Apply shifts and scaling to each variable if needed
        shift_value=shift_values[i]
        scale_value=scale_values[i]        
        for i, var in enumerate(variables):
            if scale_flags[i]:  # Check if scale is applied
                var = var / scale_value  # Apply scale        
            if shift_flags[i]:  # Check if shift is applied
                var = var + shift_value  # Apply shift
                
            variables[i] = var

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


def interactive_file_selection_TK():
    """
    Open a file dialog to select files interactively and return the selected files with their corresponding shift and scale values.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask for corresponding shift and scale values
    file_paths = []

    # Open file dialog
    file_paths = filedialog.askopenfilenames(title="Select Files", filetypes=[("All Files", "*.*")])

    return file_paths

def interactive_file_selection_QT():
    """
    Open a file dialog to select files interactively and return the selected files with their corresponding shift and scale values.
    """
    file_dialog = QFileDialog()  # Create a file dialog instance

    # Set options for the file dialog
    file_dialog.setFileMode(QFileDialog.ExistingFiles)  # Allow selecting multiple files
    file_dialog.setNameFilter("All Files (*.*)")  # Filter for all files
    file_dialog.setWindowTitle("Select Files")  # Set the dialog title

    # Ask for corresponding shift and scale values
    file_paths = []

    # Open file dialog
    if file_dialog.exec_():
        file_paths = file_dialog.selectedFiles()  # Return the selected file paths
    
    return file_paths

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

    def set_plot_type(plot_value):
        nonlocal plot_type, save_plot, save_data, x_min, x_max
        plot_type = plot_value
        save_plot = save_plot_checkbox.isChecked()
        save_data = save_data_checkbox.isChecked()
        x_min = float(x_min_input.text()) if x_min_input.text() else None
        x_max = float(x_max_input.text()) if x_max_input.text() else None
        window.close()

    window = QWidget()
    window.setWindowTitle("Plot Setting")
    layout = QVBoxLayout()

    # Add a title before the buttons
    title_label = QLabel("Select the Plot Type")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
    layout.addWidget(title_label)

    # Buttons for each plot type
    buttons = {
        "motion": QPushButton("motion"),
        "flux": QPushButton("flux"),
        "force": QPushButton("force"),
        "interfaceHeight": QPushButton("interfaceHeight")
    }

    for plot_value, button in buttons.items():
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


    # Center the window on the screen
    screen_center = QDesktopWidget().availableGeometry().center()
    window_rect = window.frameGeometry()
    window_rect.moveCenter(screen_center)
    window.move(window_rect.center())

    window.setLayout(layout)
    window.show()

    # Execute the application
    app.exec_()

    return plot_type, save_plot, save_data, x_min, x_max

class InputDialog(QDialog):
    def __init__(self, dir_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Values")
        self.dir_name = dir_name
        self.result = None  # To store the input values
        self.more_cases = False  # To indicate whether to add more cases

        # Create layout
        layout = QVBoxLayout()

        # Label input
        self.label_input = QLineEdit(self)
        self.label_input.setText(dir_name)  # Default value
        layout.addWidget(QLabel(f"Enter label:"))
        layout.addWidget(self.label_input)

        # Shift value input
        self.shift_input = QLineEdit(self)
        self.shift_input.setText("0")  # Default value
        layout.addWidget(QLabel("Enter shift value:"))
        layout.addWidget(self.shift_input)

        # Scale value input
        self.scale_input = QLineEdit(self)
        self.scale_input.setText("1")  # Default value
        layout.addWidget(QLabel("Enter scale value:"))
        layout.addWidget(self.scale_input)

        # Submit and Add button
        submit_add_button = QPushButton("Submit and Add", self)
        submit_add_button.clicked.connect(self.submit_and_add)
        layout.addWidget(submit_add_button)

        # Submit and Finish button
        submit_finish_button = QPushButton("Submit and Finish", self)
        submit_finish_button.clicked.connect(self.submit_and_finish)
        layout.addWidget(submit_finish_button)

        self.setLayout(layout)

    def submit_and_finish(self):
        self.more_cases = False
        self.accept_inputs()

    def submit_and_add(self):
        self.more_cases = True
        self.accept_inputs()

    def accept_inputs(self):
        try:
            label = self.label_input.text() or self.dir_name
            shift_value = float(self.shift_input.text() or 0)
            scale_value = float(self.scale_input.text() or 1)
            self.result = {"label": label, "shift_value": shift_value, "scale_value": scale_value}
            self.accept()  # Close the dialog with success
        except ValueError:
            # Handle invalid input cases (optional)
            pass
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process files with labels, shift values, and scale values.")
    parser.add_argument('file_args', nargs=argparse.REMAINDER,
                       help="List of input files followed by their options: -label (add a label, default: dir_name), -sh (shift, default: 0), -sc (scale, default: 1)"
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

    save_plot = args.save_plot
    save_data = args.save_data
    x_min = args.x_min
    x_max = args.x_max   

    # If Plot Type is not provided, ask in interactive input
    if  args.plot_type:
        plot_type = args.plot_type
    else:
        # plot_type = input("Enter the plot type (motion, flux, force, interfaceHeight): ")
        if pyqt_available:
            plot_type, save_plot, save_data, x_min, x_max = interactive_plot_type_selection_QT()
        else:
            plot_type = interactive_plot_type_selection_TK()

    # If no files are provided, launch interactive file selection
    if  args.file_args:
        # Use argparse for standard file input
        file_data = []
        i = 0
        while i < len(args.file_args):
            file_path = args.file_args[i]
            dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
            label, shift_value, scale_value = f"{dir_name}", 0, 1.0
            i += 1

            while i < len(args.file_args) and args.file_args[i].startswith('-'):
                if args.file_args[i] == "-label": label = args.file_args[i + 1]; i += 2
                elif args.file_args[i] == "-sh": shift_value = float(args.file_args[i + 1]); i += 2
                elif args.file_args[i] == "-sc": scale_value = float(args.file_args[i + 1]); i += 2
                else: break

            file_data.append((file_path, label, shift_value, scale_value))
        
    else:
        file_data = []
        while True:
            # Open file dialog
            if pyqt_available:
                file_paths = interactive_file_selection_QT()
            else:
                file_paths = interactive_file_selection_TK()
            
            for file_path in file_paths:
                dir_name = os.path.basename(os.path.abspath(os.path.dirname(file_path)))
 #               label = input(f"Enter label (default: {dir_name}): ") or dir_name
 #               shift_value = float(input(f"Enter shift value for {label} (default: 0): ") or 0)
 #               scale_value = float(input(f"Enter scale value for {label} (default: 1): ") or 1)
                dialog = InputDialog(dir_name)
                if dialog.exec_():  # If the user submits the dialog
                    label = dialog.result["label"]
                    shift_value = dialog.result["shift_value"]
                    scale_value = dialog.result["scale_value"]
                file_data.append((file_path, label, shift_value, scale_value))

            more_cases = dialog.more_cases
#           more_cases = input("Do you want to add more cases? (yes/no, default: no): ").strip().lower() in ['yes', 'y']
            if not more_cases:
                break

    return file_data, save_plot, save_data, plot_type, x_min, x_max


if __name__ == "__main__":
    
#    file_data = [
#          # File, label, shift_values, scale_values
#         ["./OverSet/log.interFoam","OverSet", 0, 0],
#         ["./BodyFitted/log.interFoam","BodyFitted", 0, 1]      
#    ]
      
    file_data, save_plot, save_data, plot_type, x_min, x_max = parse_arguments()

    for file_info in file_data:
        print("File:", os.path.relpath(file_info[0]))
        print("Label:", file_info[1])
        print("Shift Value:", file_info[2])
        print("Scale Value:", file_info[3])
        print()
    print(f"Plot type: {plot_type}")
    print(f"Save Plot: {save_plot}")
    print(f"Save Data: {save_data}")
    print(f"x Range: {'default' if x_min is None else x_min} - {'default' if x_max is None else x_max}")
    print()

    files = [entry[0] for entry in file_data]    
    labels = [entry[1] for entry in file_data]    
    shift_values = [entry[2] for entry in file_data]
    scale_values = [entry[3] for entry in file_data]
            
    # Plot and save data
    dynamic_plot(files, labels, plot_type, x_min, x_max, save_plot, save_data, shift_values, scale_values)
