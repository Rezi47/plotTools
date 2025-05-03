# Plot Tools

**Plot Tools** 
This Python script is designed to read, process, and visualize simulation data from various file formats. It supports extracting time-series data for multiple variables, visualizing the data with flexible plotting options, and saving the extracted data as CSV files for further analysis. It works with OpenFOAM simulation results and other structured data files.

## Features

- **Plot Types**: Supports multiple plot types such as `general` and `motion`.
- **File Selection**: Easily select files and configure settings like scale, shift, normalization, skipped rows, and used columns.
- **Dynamic Plotting**: Automatically adjusts the layout based on the number of variables.
- **Save Options**:
  - Save plots as PNG images.
  - Save extracted data as CSV files.
- **Interactive GUI**: Configure plot settings, axis titles, dimensions, and ranges through an intuitive GUI.
- **Command-Line Support**: Run the tool with command-line arguments for automated workflows.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rezi47/plotTools/tree/main
   cd plot-tools
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy==1.23.5 matplotlib==3.7.1 PyQt5==5.15.9
   ```

3. Run the tool:
   ```bash
   python plotTools.py
   ```

---

## Usage

### GUI Mode

Run the script without arguments to launch the GUI:
```bash
python plotTools.py
```

### Command-Line Mode

Run the script with arguments to process files and save plots or data without launching the GUI:
```bash
python plotTools.py -pt general -ft "My Plot" -fw 6 -ft 4 -xmi 0 -xma 10 -ymi -5 -yma 5 -sp -sd \ 
file1.txt -label "File 1" -scale 2 -shift 1 -norm_origin -skip_row 3 -usecols 0,1,2 \
file2.txt -label "File 2" -scale 1 -shift 0 -usecols 0,2

```

---

## Command-Line Arguments

| Argument         | Description                                                     | Default Value |
|------------------|-----------------------------------------------------------------|---------------|
| file_args        | List of input files followed by their options                   | None          |
| `-plot_type`     | Specify the type of plot (`general`, `motion`)                  | `general`     |
| `-fig_title`     | Specify a title for the figure                                  | None          |
| `-fig_width`     | Specify the width of the figure in inches                       | 5             |
| `-fig_height`    | Specify the height of the figure in inches                      | 4             |
| `-x_axis_title`  | Specify a title for the x-axis                                  | `t`           |
| `-x_axis_dim`    | Specify a dimension for the x-axis                              | `s`           |
| `-y_axis_title`  | Specify titles for the y-axes (comma-separated for multiple)    | `Amplitude`   |
| `-y_axis_dim`    | Specify dimensions for the y-axes (comma-separated)             | `-`           |
| `-x_min`         | Minimum x-axis value                                            | None          |
| `-x_max`         | Maximum x-axis value                                            | None          |
| `-y_min`         | Minimum y-axis value                                            | None          |
| `-y_max`         | Maximum y-axis value                                            | None          |
| `-save_plot`     | Save the plot as a PNG image                                    | Disabled      |
| `-save_data`     | Save the extracted data to CSV files                            | Disabled      |

---

## GUI Features

- **Plot Type Selection**: Choose the type of plot (`general`, `motion`).
- **Figure Properties**: Set the figure title, width, and height.
- **Axis Settings**: Configure x-axis and y-axis titles, dimensions, and ranges.
- **File Selection**: Add files, configure settings (scale, shift, normalization, etc.).

### Buttons

- 👁️ **Plot**: Generate the plot.
- 💾 **Save Plot**: Save the generated plot as an image.
- 📊 **Save Data**: Save the extracted data to a file.

---

## File Settings

For each file, you can configure the following:

- **Label**: A custom label for the file.
- **Scale**: A scaling factor for the data.
- **Shift**: A shift value to add to the data.
- **Normalize**: Normalize the data to its origin.
- **Skip Rows**: Number of rows to skip at the start of the file.
- **Used Columns**: Specify the column indices to use (comma-separated).

---

## Output

### Plots

- Saved as PNG images in the current directory or in `./individual_figures`.

### Extracted Data

- Saved as CSV files in the `./extracted_data` directory.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---
