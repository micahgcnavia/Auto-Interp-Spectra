# Auto-Interp-Spectra

This is a code to automate the process of interpolating stellar and exoplanet atmospheric spectra based on PHOENIX and ATMO libraries. The code is structured as follows:

## Project Structure 📦

### 1. **`filtering.py`** (under construction 🛠️)
This file is the first step to get the interpolation process done. It:
- Checks if you have all spectra needed to interpolate your targets
- Returns a `.csv` file detailing each model needed (its temperature, gravity and metallicity)
- Returns a `.csv` file with flags to indicate which parameters to interpolate

### 2. **`interpolate.py`**
This file contains the complete interpolation process:
- Checks which parameters to interpolate
- Performs the interpolation
- Returns the interpolated spectra

### 3. **`plotting.py`** (under construction 🛠️)
This file enables the visualization of the interpolation process.
- Interative plots (change wavelength range, select which spectra to plot)
- Saves the plots [Optional]

## How to Use 💻

1. Download or clone the repository into your machine:
   ```bash
   git clone https://github.com/micahgcnavia/Auto-Interp-Spectra.git
   ```
2. Change `database_path` in `main.py` to match the folder path in your machine where you store the models
3. Change `targets` in `main.py` to match the file where you have the parameters for your objects. For example:

    | Object   | teff | logg | feh   |
    |----------|------|------|-------|
    | CoRoT-1  | 6000 | 4.25 | -0.30 |
    | CoRoT-2  | 5600 | 4.53 | 0.04  |

    Where 'teff' is the effective temperature, 'logg' is the logarithm of the surface gravity and 'feh' is the metallicity. See `stars.csv` for more examples. It is important to keep the same column names and order!

4. Run `filter.py` to get the clean table ready for analysis (remember to change `database_path` to match the path in your machine!)
5. Run `main.py` to get the interpolated spectra for your list of objects

### Citation 📰

If you feel like using this code in your own projects, please consider citing this repository! Thank you! :)
