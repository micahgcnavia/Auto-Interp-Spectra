import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from glob import glob
from natsort import os_sorted
import os
from tqdm import tqdm
from matplotlib import rcParams
from itertools import product

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

from lib.plotting import graph

class SpectrumInterpolator:
    def __init__(self, wav_ref, delta_params=None):
        """
        Initialize the SpectrumInterpolator with reference wavelength and optional delta parameters.
        
        Args:
            wav_ref (array): Reference wavelength array
            delta_params (dict): Dictionary of parameter deltas (default: {'teff': 100, 'logg': 0.5, 'feh': 0.5})
        """
        print('Initializing class...\n')
        self.wav_ref = wav_ref
        self.delta_params = delta_params or {
            'teff': 100,  # [K]
            'logg': 0.5,  # [dex]
            'feh': 0.5    # [dex]
        }
        self.cwd = os.getcwd()
        
    @staticmethod
    def interp_partial(spectrum1, spectrum2, factor, delta_param):
        """
        Linear interpolation between two spectra.
        """
        return spectrum1 + (((spectrum2 - spectrum1) * factor) / delta_param)
    
    def get_extreme_spectra(self, spectra, interpolate_flags, param_values):
        """
        Get spectra with extreme parameter values based on interpolation flags.
        """
        conditions = []
        
        for param, value in param_values.items():
            if interpolate_flags[param][0]:
                col = param  # 'teff', 'logg', or 'feh'
                val = min(spectra[col]) if value == 'min' else max(spectra[col])
                conditions.append(spectra[col] == val)
        
        if not conditions:
            return pd.DataFrame()
            
        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition &= cond
            
        return spectra.loc[combined_condition]
    
    def load_and_interpolate_spectrum(self, spectrum_row):
        """
        Load spectrum from file and interpolate to reference wavelength.
        """
        wav, spec = np.loadtxt(self.cwd + spectrum_row['path'].item(), unpack=True)
        return np.interp(self.wav_ref, wav, spec)
    
    def perform_interpolation(self, spectra_pairs, target_params, interpolate_flags, spectra):
        """
        Perform interpolation steps for given spectra pairs and parameters.
        """
        print('Performing interpolation...\n')

        current_spectra = [self.load_and_interpolate_spectrum(pair) for pair in spectra_pairs]
        
        # Perform interpolation for each parameter that needs it
        interp_steps = {'raw_spec': current_spectra}

        # Initialize parameters tracking with the raw spectra parameters
        current_params = [
            (row['teff'], row['logg'], row['feh']) 
            for row in spectra_pairs
        ]
        
        for i, interp_flag in enumerate(interpolate_flags.items()):
            if not interp_flag[1].item():
                continue
                
            if len(current_spectra) % 2 != 0:
                raise ValueError("Odd number of spectra for interpolation")
            
            new_spectra = []
            new_params = []
            
            for j in range(0, len(current_spectra), 2):
                spec1 = current_spectra[j]
                spec2 = current_spectra[j+1]
                
                # Perform the interpolation
                interpolated = self.interp_partial(
                    spec1, spec2, 
                    target_params[interp_flag[0]] - min(spectra[interp_flag[0]]), 
                    self.delta_params[interp_flag[0]]
                )
                new_spectra.append(interpolated)
                
                # Create new parameter set for the interpolated spectrum
                param_values = {}
                for p in interpolate_flags:
                    if p == interp_flag[0]:
                        param_values[p] = target_params[p]  # Interpolated value
                    else:
                        # Take value from either spectrum (should be same for non-interpolated params)
                        param_values[p] = current_params[j][list(interpolate_flags.keys()).index(p)]
                
                new_params.append(tuple(param_values[p] for p in interpolate_flags))
            
            current_spectra = new_spectra
            current_params = new_params
            interp_steps[f'interp{i+1}_spec'] = current_spectra
            interp_steps[f'interp{i+1}_params'] = current_params
    
        return current_spectra[0], interp_steps
    
    def interpolate(self, target, spectra, interpolate_flags, cwd, show_graphs=False, save_file=False, save_fig=False):
        """
        Main interpolation method.
        
        Args:
            target: DataFrame with target parameters
            spectra: DataFrame of spectra to use for interpolation
            interpolate_flags: Dictionary with flags for which parameters to interpolate
            show_graphs: Whether to show interpolation graphs
            save_file: Whether to save the interpolated spectrum
            save_fig: Whether to save the graphs
            path: Path to save files (required if save_file or save_fig is True)
            
        Returns:
            Dictionary with interpolation results and steps
        """
        name = target['star'].replace(' ', '').lower()
        params = {
            'teff': target['teff'].item(),
            'logg': target['logg'].item(),
            'feh': target['feh'].item()
        }
        
        # Determine which parameters to interpolate
        interp_params = [p for p in interpolate_flags if interpolate_flags[p][0]]
        
        # Get all combinations of min/max for parameters we're interpolating
        extremes = list(product(*[['min', 'max'] if interpolate_flags[p][0] else ['fixed'] for p in interpolate_flags]))
        
        # Get the extreme spectra
        extreme_spectra = []
        for combo in extremes:
            param_values = {p: combo[i] for i, p in enumerate(interpolate_flags)}
            extreme_spectra.append(self.get_extreme_spectra(spectra, interpolate_flags, param_values))
        
        # Check if we have all required spectra
        if any(len(s) == 0 for s in extreme_spectra):
            print(f'Missing spectra for {name}')
            return None
        
        # Perform the interpolation

        try:
            final_spectrum, interp_steps = self.perform_interpolation(extreme_spectra, params, interpolate_flags, spectra)
        except ValueError as e:
            print(f'Interpolation error for {name}: {str(e)}')
            return None
        
        # Prepare the return dictionary
        interp_steps['final_params'] = [params[p] for p in interpolate_flags]
        interp_steps['final_spec'] = final_spectrum
        interp_steps['raw_params'] = [(row['teff'], row['logg'], row['feh']) for _, row in spectra.iterrows()]
        
        cwd = os.getcwd()

        # Handle output options
        if show_graphs:
            graph(name, interp_steps, self.wav_ref, cwd, save_fig=save_fig)
        
        if save_file:

            try:
                path = cwd+'/output/interp_spectra/'
                os.mkdir(path)
            
            except:
                pass

            df = pd.DataFrame({'wavelength': self.wav_ref, 'flux': final_spectrum})
            df.to_csv(f"{path}{name}_interp.csv", index=False)
        
        return interp_steps