import pandas as pd
import numpy as np
from glob import glob
from natsort import os_sorted
import os
import ast
from tqdm import tqdm
from more_itertools import pairwise
from collections import defaultdict
from lib.scraping import *

class SpectrumInterpolator(Input):

    """
        Encapsulates the entire interpolation process.
    
    """

    def __init__(self):

        """
            Initializes the SpectrumInterpolator class.

        """

        super().__init__()

        print('='*20+' Initializing SpectrumInterpolator '+'='*20+'\n')

        scraper = Scraper()

        self.delta_params = scraper.get_delta_params()

        # Getting filtered data
        self.spectra = pd.read_csv(self.cwd+f'/output/filtered/{self.name.lower()}_data.csv')
        self.interpolate_flags = pd.read_csv(self.cwd+f'/output/filtered/{self.name.lower()}_interpolate.csv')
        
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

        wav, spec = np.loadtxt(spectrum_row['path'], unpack=True)
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

            data = {key: self.params[param] if key == param else row1[key] for key in self.parameters} # It can be row1 or row2 because the parameters' values are the same
            data['flux'] = interp_flux

            interp_steps.append(data)

        return interp_steps

    
    def interpolate_spectra(self, save_file):
        
        """
            Main interpolation function.

            :param save_file: Flag to indicate whether to save the interpolated spectrum in a CSV file.
            :type save_file: bool
            :return: List of DataFrames containing updated parameters and fluxes at each interpolation step.
            :rtype: list[pandas.DataFrame]
        
        """
        
        # Perform the interpolation

        print(f'Interpolating spectra for {self.name}\n')

        steps = []

        for param, condition in self.interpolate_flags.iteritems():

            if condition[0]: # Checks whether the current parameter needs to be interpolated

                if len(steps) == 0:
                    source = self.spectra # Original dataframe retrieved from filtering the database
                else:
                    source = steps[-1] # Gets latest dataframe after first interpolation loop

                df = self.sort_df(source, param) # Sorts dataframe 
                df = self.interp_param(df, param) # Interpolates current parameter
                df = self.combine_dicts(df) # Combine dictionaries to create a new dataframe with updated values after interpolation
                steps.append(df)
            else:
                pass
    
        print('\nFinal parameters:')
        print(steps[-1][self.parameters])
        print('='*33+' Finish! '+'='*33)
        
        if save_file:

            path = self.cwd+'/output/interp_spectra/'


            df = pd.DataFrame({'wavelength': self.wav_ref, 'flux': steps[-1]['flux'].item()})
            df.to_csv(f"{path}{self.name.lower()}_interp.csv", index=False)
        
        return steps

def main(save_file=False):

    # Initializes interpolator
    interpolator = SpectrumInterpolator()

    # Performs interpolation
    result = interpolator.interpolate_spectra(save_file=save_file)

    return result


if __name__ == "__main__":
    main()
