# Auto-Interp-Spectra

This is a code to automate the process of interpolating stellar and exoplanet atmospheric spectra based on PHOENIX and ATMO libraries. The code is structured as follows:

## Project Structure 📦

### 1. **`filtering.py`** (under construction 🛠️)
This file is the first step to get the interpolation process done. It:
- Checks if you have all spectra needed to interpolate your targets
- Returns a `.csv` file detailing each model needed (its temperature, gravity and metallicity)
- Returns a `.csv` file with flags to indicate which parameters to interpolate

### 2. **`interpolate.py`** (under construction 🛠️)
This file contains the complete interpolation process:
- Checks which parameters to interpolate
- Performs the interpolation
- Plots the result [optional]
- Returns the interpolated spectra

## How to Use 💻

1. Download or clone the repository into your machine:
   ```bash
   git clone https://github.com/micahgcnavia/Auto-Interp-Spectra.git
   ```
2. Change `database_path` in `filtering.py` to match the folder path in your machine where you store the models
3. Change `targets` in `filtering.py` and `interpolate.py` to match the file where you have the parameters for your objects. For example:

    | Object   | teff | logg | feh   |
    |----------|------|------|-------|
    | CoRoT-1  | 6000 | 4.25 | -0.30 |
    | CoRoT-2  | 5600 | 4.53 | 0.04  |

    See `teff_logg_FeH.csv` for more examples. It is important to keep the same column names and order!

4. Run `filtering.py` to get the clean table ready for analysis (remember to change `database_path` to match the path in your machine!)
5. Run `interpolate.py` to get the interpolated spectra for your list of objects

### Citation 📰

If you feel like using this code in your own projects, please consider citing this repository! Thank you! :)
