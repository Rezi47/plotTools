# Data Plotting and Extraction Tool
This Python script is designed to read, process, and visualize various types of simulation data from files. It supports extracting time-series data for multiple variables from a range of file formats and provides functionality for plotting the data and saving it as CSV files.

## Features
- Data Extraction: The script can extract data such as motion, force, flux, and interface height from specific file formats (e.g., OpenFOAM simulation results).
- Data Visualization: It can generate dynamic plots with multiple variables and labels, offering flexibility in data presentation.
- Data Saving: Extracted data can be saved to CSV files for further analysis.
- Flexible Plotting: Allows for shifting and scaling of data before plotting.
- Multi-file Support: The script can handle multiple files and generate comparative plots.
## Supported Data Types
- Motion: Extracts motion data such as heave, roll, surge, pitch, yaw, and sway.
- Force: Extracts force data in x, y, and z directions.
- Flux: Extracts flux-related data.
- Interface Height: Extracts interface height data from gauges.
## Dependencies
### This script requires the following Python packages:

- `matplotlib`: For plotting graphs.
- `numpy`: For handling numerical operations and data manipulation.
- `argparse`: For parsing command-line arguments.
You can install the dependencies using pip:

```bash
pip install matplotlib numpy
```

### Modifying the controlDict for rigidBodyMotion Solver
If you are using the rigidBodyMotion solver and want to enable motion features in the code, you need to add the following code to your controlDict file:

```bash
DebugSwitches
{
    rigidBodyModel 1;
}
```
This enables the rigid body motion model to save motion data in the log file.

## Usage
### Command-line Arguments
You can run the script with various arguments. The basic syntax is:

```bash
python script_name.py <file_path1> <file_path2> ... [options]
```

### Arguments
- **file_args:** List of input file paths followed by options for each file. Each file can be optionally followed by:

    - `-label <label>`: Specify a label for the file (default: directory name).
    - `-sh <shift_value>`: Shift the data by a specified value (default: 0).
    - `-sc <scale_value>`: Scale the data by a specified factor (default: 1).
- **-plot_type, -pt:** Type of plot to generate. Options are:

    - `motion`: Data related to motion (default).
    - `flux`: Data related to flux.
    - `force`: Data related to forces.
    - `interfaceHeight`: Data related to interface height.

- **-save_plot, -sp:** Enable saving the plot as a PNG image (default: disabled).

- **-save_data, -sd:** Enable saving the extracted data to CSV files (default: disabled).

- **-x_min:** Minimum value for the x-axis.

- **-x_max:** Maximum value for the x-axis.

### Example Usage
```bash
python script_name.py -pt force -sp -sd -x_max 100 ./data1/log.solverName -label "Case 1" -sc 0.2 ./data2/log.solverName -label "Case 2" -sh 1 -sc 0.2 
```

This command will:

- Process the files `log.solverName` located in the `./data1/` and `./data2/` directories.
- Assign the label "Case 1" and "Case 2" to the files.
- Apply a shift-up of 1 to the Case 2 and a scale of 0.2 to the both cases.
- Save the plot as a PNG file.
- Save the extracted data to CSV files.
- Generate force-related plots with the x-axis up to 100.

## How It Works
1. Data Extraction:

    - The script reads specific columns from files based on the `fig_type` (motion, flux, force, or interfaceHeight).
    - Each file is processed to extract time-series data and other relevant variables.
2. Plotting:

    - A dynamic plot is created, with one subplot per variable.
    - Each subplot can be labeled, and the x-axis can be customized (min/max).
    - Shifting and scaling are applied to the data if specified.
3. Saving Data:

    - The script saves extracted data as CSV files, one for each variable in each file.
4. Saving Plots:

    - The plot can be saved as a PNG file, with all the data visualized for comparison.
## Example Output
### Plot:
A dynamic plot showing the data for each variable across all files. The plot will include labels and a legend, and the x-axis and y-axis labels will be set based on the type of data.

### CSV Data Files:
The script saves the extracted data in CSV format with columns like `Time` and the variable name (e.g., `x-Force`, `Heave`), depending on the type of data extracted.

## Customization
### Plot Customization:
You can control the appearance of the plot by modifying the following variables:

- `colors`: Set the colors for the plot lines.
- `linestyles`: Set the line styles (solid, dashed, etc.).
- `x_min` and `x_max`: Adjust the x-axis range for the plot.
### Data Processing:
The script allows for shift and scale adjustments, which can be useful if the data needs to be adjusted before plotting.

### Adding More Data Types:
If you want to add more data extraction functions or customize the figures, you can modify the select_fig function to include new data types and define corresponding figures.

## Contributing
Feel free to fork the repository, create a branch, and submit a pull request with new features, bug fixes, or improvements. Please ensure that your changes are well-documented and tested.
