#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import itertools
from io import StringIO

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
                time = float(line.split('=')[1].strip())
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
    	    ["Amplitude Gauge 1", r'$m$' , 1, 1],
    	    ["Amplitude Gauge 2", r'$m$' , 1, 1],
    	    ["Amplitude Gauge 3", r'$m$' , 1, 1],
    	    ["Amplitude Gauge 4", r'$m$' , 1, 1],
    	    ["Amplitude Gauge 5", r'$m$' , 1, 1],
    	    ["Amplitude Gauge 6", r'$m$' , 1, 1],
    	    ["Amplitude Gauge 7", r'$m$' , 1, 1]
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process files with labels, shift values, and scale values.")
    parser.add_argument('file_args', nargs=argparse.REMAINDER,
                       help="List of input files followed by their options: -label (add a label, default: dir_name), -sh (shift, default: 0), -sc (scale, default: 1)"
                       )
    parser.add_argument('-plot_type', '-pt', type=str,
                       help='Specify the type of plot (motion, flux, force, interfaceHeight) (default: motion)',
                       default='motion',
                       choices=['motion', 'flux', 'force', 'interfaceHeight']
                       )                       
    parser.add_argument('-save_plot', '-sp',action='store_true', help="Disable saving the plot (default: False)")
    parser.add_argument('-save_data', '-sd',action='store_true', help="Disable saving the data (default: False)")
    parser.add_argument('-x_min', type=float, help="Minimum x-axis value")
    parser.add_argument('-x_max', type=float, help="Maximum x-axis value")
    
    args = parser.parse_args()

    file_data = []
    i = 0
    while i < len(args.file_args):
        file_path = args.file_args[i]
        # Get the parent directory and the last part
        parent_dir = os.path.abspath(os.path.dirname(file_path))
        dir_name = os.path.basename(parent_dir)
        label, shift_value, scale_value = f"{dir_name}", 0, 1.0
        #label, shift_value, scale_value = f"Case_{len(file_data) + 1}", 0, 1.0
        i += 1

        while i < len(args.file_args) and args.file_args[i].startswith('-'):
            if args.file_args[i] == "-label": label = args.file_args[i + 1]; i += 2
            elif args.file_args[i] == "-sh": shift_value = float(args.file_args[i + 1]); i += 2
            elif args.file_args[i] == "-sc": scale_value = float(args.file_args[i + 1]); i += 2
            else: break

        file_data.append((file_path, label, shift_value, scale_value))
    return file_data, args.save_plot, args.save_data, args.plot_type, args.x_min, args.x_max

if __name__ == "__main__":
    
#    file_data = [
#          # File, label, shift_values, scale_values
#         ["./OverSet/log.interFoam","OverSet", 0, 0],
#         ["./BodyFitted/log.interFoam","BodyFitted", 0, 1]      
#    ]
      
    file_data, save_plot, save_data, plot_type, x_min, x_max = parse_arguments()
    
    for file_info in file_data:
        print("File:", file_info[0])
        print("Label:", file_info[1])
        print("Shift Value:", file_info[2])
        print("Scale Value:", file_info[3])
        print()
    print(f"Save Plot: {save_plot}")
    print(f"Save Data: {save_data}")
    print()

    files = [entry[0] for entry in file_data]    
    labels = [entry[1] for entry in file_data]    
    shift_values = [entry[2] for entry in file_data]
    scale_values = [entry[3] for entry in file_data]
            
    # Plot and save data
    dynamic_plot(files, labels, plot_type, x_min, x_max, save_plot, save_data, shift_values, scale_values)
