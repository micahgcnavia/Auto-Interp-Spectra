import pandas as pd
import numpy as np
from glob import glob
from natsort import os_sorted
import os
from tqdm import tqdm
from more_itertools import pairwise
from collections import defaultdict

class SpectrumInterpolator:

    """
        Encapsulates the entire interpolation process.
    
    """

    def __init__(self, wav_ref, target, delta_params=None):

        """
            Initializes the SpectrumInterpolator class.
            
            :param wav_ref: Wavelength array of reference.
            :type wav_ref: numpy.ndarray
            :param target: DataFrame containing the name of the object and its parameters.
            :type target: pandas.DataFrame
            :param delta_params: Optional DataFrame describing the parameters steps for the model.
            :type delta_params: pandas.DataFrame

        """

        print('='*27+' Initializing SpectrumInterpolator '+'='*27+'\n')
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

            :param spectrum1: Flux with lowest parameter value.
            :type spectrum1: numpy.ndarray
            :param spectrum2: Flux with highest parameter value.
            :type spectrum2: numpy.ndarray
            :param factor: Amount to increase.
            :type factor: float or int
            :param delta_param: Model's parameter step
            :type delta_param: float or int
            :return: The interpolated flux.
            :rtype: numpy.ndarray

        """

        return spectrum1 + (((spectrum2 - spectrum1) * factor) / delta_param)
    
    def sort_df(self, df, param):

        """
            Sorts the DataFrame to facilitate the interpolation order.

            :param df: DataFrame containing all spectra needed for interpolation and their parameters' values.
            :type df: pandas.DataFrame
            :param param: Name of the parameter to sort the DataFrame accordingly.
            :type param: str
            :return: The sorted DataFrame.
            :rtype: pandas.DataFrame

        """

        results = []
        
        grouped = df.groupby([p for p in self.parameters if p != param])
        
        for _, group in grouped:
            min_row = group.loc[group[param].idxmin()]
            max_row = group.loc[group[param].idxmax()]
            results.extend([min_row, max_row])
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def load_and_interpolate_spectrum(self, spectrum_row):

        """
            Loads spectrum from file and interpolates its flux into the wavelength array of reference.

            :param spectrum_row: DataFrame row containing the path to the spectrum of interest.
            :type spectrum_row: pandas.DataFrame
            :return: The interpolated flux.
            :rtype: numpy.ndarray

        """

        wav, spec = np.loadtxt(self.cwd + spectrum_row['path'], unpack=True)
        return np.interp(self.wav_ref, wav, spec)

    def combine_dicts(self, dict_list):

        """
            Combines multiple dictionaries and merged them into a DataFrame.

            :param dict_list: List of dictionaries to combine.
            :type dict_list: list[dict]
            :return: The combined DataFrame.
            :rtype: pandas.DataFrame

        """

        combined = defaultdict(list)
        for d in dict_list:
            for key, value in d.items():
                combined[key].append(value)

        return pd.DataFrame(dict(combined))

    def check_spectra_availability(self, interpolate_flags, spectra):

        """
            Checks whether we have all spectra needed for interpolation based on the number of parameters to interpolate.

            :param interpolate_flags: DataFrame specifing which parameters to interpolate based on bool values.
            :type interpolate_flags: pandas.DataFrame
            :param spectra: DataFrame containing all spectra needed for interpolation and their parameters' values.
            :type spectra: pandas.DataFrame
            :return: True of False
            :rtype: bool

        """

        print('Checking spectra availability...')
        n = interpolate_flags.sum().sum() # number of parameters to interpolate

        N = len(spectra) # number of spectra available

        if N != 2**n:
            return False

        else:
            return True
    
    def interp_param(self, df, param):

        """
            Interpolates the spectra based on a specific parameter.

            :param df: Sorted DataFrame containing all spectra needed for interpolation and their parameters' values.
            :type df: pandas.DataFrame
            :param param: Name of the parameter to interpolate.
            :type param: str
            :return: List of dictionaries stating the updated parameters at each interpolation and the interpolated flux.
            :rtype: list[dict]

        """

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

    
    def interpolate_spectra(self, spectra, interpolate_flags, save_file=False):
        
        """
            Main interpolation function.

            :param spectra: DataFrame containing all spectra needed for interpolation and their parameters' values.
            :type spectra: pandas.DataFrame
            :param interpolate_flags: DataFrame specifing which parameters to interpolate based on bool values.
            :type interpolate_flags: pandas.DataFrame 
            :param save_file: Flag to indicate whether to save the interpolated spectrum in a CSV file.
            :type save_file: bool
            :return: List of DataFrames containing updated parameters and fluxes at each interpolation step.
            :rtype: list[pandas.DataFrame]
        
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
        
        if save_file:

            try:
                path = self.cwd+'/output/interp_spectra/'
                os.mkdir(path)
            
            except:
                pass

            df = pd.DataFrame({'wavelength': self.wav_ref, 'flux': steps[-1]['flux'].item()})
            df.to_csv(f"{path}{name}_interp.csv", index=False)
        
        return steps