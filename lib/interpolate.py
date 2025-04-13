import pandas as pd
import numpy as np
from glob import glob
from natsort import os_sorted
import os
from tqdm import tqdm
from more_itertools import pairwise
from collections import defaultdict
from lib.plotting import graph

class SpectrumInterpolator:
    def __init__(self, wav_ref, target, delta_params=None):
        """
        Initialize the SpectrumInterpolator with reference wavelength and optional delta parameters.
        
        Args:
            wav_ref (array): Reference wavelength array
            delta_params (dict): Dictionary of parameter deltas (default: {'teff': 100, 'logg': 0.5, 'feh': 0.5})
        """

        print('='*20+' Initializing SpectrumInterpolator '+'='*20+'\n')
        self.wav_ref = wav_ref
        self.delta_params = delta_params or {
            'teff': 100,  # [K]
            'logg': 0.5,  # [dex]
            'feh': 0.5    # [dex]
        }
        self.cwd = os.getcwd()
        self.target = target
        self.params = {
            'teff': self.target['teff'].item(),
            'logg': self.target['logg'].item(),
            'feh': self.target['feh'].item()
        }
        self.parameters = list(self.params.keys())

        
    @staticmethod
    def interp_partial(spectrum1, spectrum2, factor, delta_param):
        """
        Linear interpolation between two spectra.
        """

        return spectrum1 + (((spectrum2 - spectrum1) * factor) / delta_param)
    
    def sort_df(self, df, param):

        results = []
        
        grouped = df.groupby([p for p in self.parameters if p != param])
        
        for _, group in grouped:
            min_row = group.loc[group[param].idxmin()]
            max_row = group.loc[group[param].idxmax()]
            results.extend([min_row, max_row])
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def load_and_interpolate_spectrum(self, spectrum_row):
        """
        Load spectrum from file and interpolate to wavelength resolution of reference.
        """

        wav, spec = np.loadtxt(self.cwd + spectrum_row['path'], unpack=True)
        return np.interp(self.wav_ref, wav, spec)

    def combine_dicts(self, dict_list):

        combined = defaultdict(list)
        for d in dict_list:
            for key, value in d.items():
                combined[key].append(value)

        return pd.DataFrame(dict(combined))

    def check_spectra_availability(self, interpolate_flags, spectra):
        """
        Checks whether we have all spectra needed for interpolation based on the number of parameters to interpolate.
        """
        print('Checking spectra availability...')
        n = interpolate_flags.sum().sum() # number of parameters to interpolate

        N = len(spectra) # number of spectra available

        if N != 2**n:
            return False

        else:
            return True
    
    def interp_param(self, df, param):

        interp_steps = [] # This will storage each dictionary with updated parameters and the corresponding interpolated flux

        for i in tqdm(range(0, len(df), 2), desc='Interpolating '+str(param)):
            row1 = df.iloc[i, :].to_dict()  # First row of the pair
            row2 = df.iloc[i + 1, :].to_dict() if i + 1 < len(df) else None  # Second row (if exists)
            
            if row2 is None:
                pass

            try:
                spec1 = self.load_and_interpolate_spectrum(row1) # This will work only for the first interpolated parameter
                spec2 = self.load_and_interpolate_spectrum(row2)

            except:
                spec1 = row1['flux'] # Then for the rest of the parameters this should work
                spec2 = row2['flux']

            factor = self.params[param] - row1[param] # Desidered value - minimum parameter value

            interp_flux = self.interp_partial(
                spec1,
                spec2,
                factor,
                self.delta_params[param]
            )

            dic = {key: self.params[param] if key == param else row1[key] for key in self.parameters} # It can be row1 or row2 because the parameters' values are the same
            dic['flux'] = interp_flux

            interp_steps.append(dic)

        return interp_steps

    
    def interpolate_spectra(self, spectra, interpolate_flags, show_graphs=False, save_file=False, save_fig=False):
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
        name = self.target['star'].item().strip().lower()

        # Check if we have all required spectra
        if self.check_spectra_availability(interpolate_flags, spectra):
            print('All spectra available!')

        else:
            print(f'Missing spectra for {name}')
            return None
        
        # Perform the interpolation

        steps = []

        for param, condition in interpolate_flags.iteritems():

            if condition[0]: # Checks whether the current parameter needs to be interpolated

                if len(steps) == 0:
                    source = spectra # Original dataframe retrieved from filtering the database
                else:
                    source = steps[-1] # Gets latest dataframe after first interpolation loop

                df = self.sort_df(source, param) # Sorts dataframe 
                df = self.interp_param(df, param) # Interpolates current parameter
                df = self.combine_dicts(df) # Combine dictionaries to create a new dataframe with updated values after interpolation
                steps.append(df)
            else:
                pass

        print('='*40+' Finish! '+'='*40)        
        print('Result:')
        print(steps[-1])
        print('='*89)

        # Handle output options
        if show_graphs:
            graph(name, steps, self.wav_ref, self.cwd, save_fig=save_fig)
        
        if save_file:

            try:
                path = self.cwd+'/output/interp_spectra/'
                os.mkdir(path)
            
            except:
                pass

            df = pd.DataFrame({'wavelength': self.wav_ref, 'flux': steps[-1]['flux']})
            df.to_csv(f"{path}{name}_interp.csv", index=False)
        
        return steps