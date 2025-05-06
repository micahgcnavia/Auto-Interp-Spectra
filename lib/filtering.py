import pandas as pd
import numpy as np
from glob import glob
from natsort import os_sorted
import os
import ast
from tqdm import tqdm
import itertools
from more_itertools import pairwise
from functools import reduce
from collections import defaultdict
from lib.scraping import *

class Filter(Input):

    """
        Filters the database and gets the missing spectra from the Spanish Virtual Observatory (SVO), if any.

    """

    def __init__(self):

        """
            Initializes the Filter class.

        """

        super().__init__()

        print('='*27+' Initializing Filter '+'='*27+'\n')

        self.scraper = Scraper()

        self.delta_params = self.scraper.get_delta_params()

        self.params_range = {param: self.scraper.get_param_range(param) for param in self.parameters}
        
        # Initializing interpolate dictionary
        self.interpolate = {
            'teff': None,
            'logg': None,
            'meta': None}

        # The conditions to filter the database will be storaged in this list
        self.conditions = []


    def get_header_data(self, file_path, show_model=False):

        """
            Reads the models' header and encapsulates it into a dictionary. 

            :param file_path: Spectrum's path.
            :type file_path: str
            :param show_model: Flag to indicate whether to show the model name or not.
            :type show_model: bool
            :return: Dictionary containg the models' metadata.
            :rtype: dict
        
        """
    
        file = open(file_path, mode='r', newline='')
        lines = file.readlines()
        header = lines[0:5]
        
        # Getting the parameters' names
        param_name = [header[i].split(' ')[1] for i in np.arange(1,len(header))]
        
        header_dict = {}

        for i, param in zip(np.arange(1,len(header)), param_name):
        
            # Retrieving the values
            header_dict[param] = float(header[i].split(' ')[3])
            
        model_names = header[0].split(' ')

        model = model_names[1].strip()

        if len(model_names) > 2:

            for name in model_names[2:]:
                model += '_'+name.strip()

        header_dict['model'] = model
        header_dict['path'] = file_path

        if header_dict['alpha'] != 0:

            print('='*80)
            print('Found alpha != 0 in path:', file_path)

        else:
            pass

        return header_dict

    def get_all_models(self, save=None):

        """
            Storages all models in a pandas DataFrame.
            
            :param save: Flag to indicate whether to save the DataFrame in a CSV file.
            :type save: bool
            :return: DataFrame with the metadata of all models in the database.
            :rtype: pandas.DataFrame

        """
        
        models = [self.get_header_data(file) for file in tqdm(self.models_list, 'reading files...')]

        df = pd.DataFrame(models)

        if save:

            df.to_csv(self.cwd+'/all_models_'+self.model.lower()+'.csv', index=False)

        return df

    def generate_combinations(self, teff, logg, meta):
    
        # Generate all possible combinations (Cartesian product)
        combinations = itertools.product(teff, logg, meta)
        
        # Convert each combination into a dictionary
        result = [
            {"teff": t, "logg": lg, "meta": m}
            for t, lg, m in combinations
        ]
        return result

    def filter_params(self):

        # This will storage the combination of parameters to look for when retrieving the models to interpolate.
        values = {}

        for param in self.parameters:

            # Checking if the input value is not in the model pre-computed grid
            if self.params[param] not in self.params_range[param]:

                # If not, gets the model values around the input value using the corresponding parameter step
                values[param] = self.params_range[param][abs(self.params_range[param] - self.params[param]) <= self.delta_params[param]]
                self.interpolate[param] = True

            else:

                # If true, creates an array with the value
                values[param] = np.array([self.params[param]])
                self.interpolate[param] = False

        combinations = self.generate_combinations(values['teff'], values['logg'], values['meta'])
        
        return pd.DataFrame(combinations)

    def filter_database(self, all_models):

        filtered_df = self.filter_params()

        # Merge with the database dataframe to find matching rows
        matched = pd.merge(all_models, filtered_df, how='inner', on=['teff', 'logg', 'meta'])
        
        # Find conditions NOT matched
        missing_conditions = filtered_df.merge(matched, on=['teff', 'logg', 'meta'], how='left', indicator=True)
        # Isolating only the unmatched rows
        missing_conditions = missing_conditions[missing_conditions['_merge'] == 'left_only'].drop('_merge', axis=1)

        return matched, missing_conditions[self.parameters]


    def filter_all(self, save_all_models=False, save_result=False):

        """
            Applies the filtering for the target and its parameters. Creates auxiliary files for the interpolation script.

            :param save_all_models: Flag to indicate whether to save the table with all models' information.
            :type save_all_models: bool
            :param save_result: Flag to indicate whether to save the filtered table and the instructions table for interpolation.
            :type save_result: bool
            :return: A table containing all spectra needed for interpolation and a boolean table indicating which parameters to interpolate.
            :rtype: pandas.DataFrame

        """

        try:

            # Reads the file if it already exists
            all_models = pd.read_csv(self.cwd+f'/all_models_{self.model.lower()}.csv')

        except:

            # If there is no all_models file, creates one
            all_models = self.get_all_models(save=save_all_models)

        print(f'\nFiltering the database to get spectra for {self.name}\n')

        print(f'{self.name} parameters:\n')

        for key, item in self.params.items():

            print(key+f' = {item}')

        filtered_df, missing_spectra = self.filter_database(all_models=all_models)

        if len(missing_spectra) == 0:

            print('All spectra available!')

            interpolate = pd.DataFrame(self.interpolate, index=[0]) # This will be a simple DataFrame stating which parameters to interpolate

            if save_result:

                name = self.name.lower()

                filtered_df.to_csv(self.cwd+f'/output/filtered/{name}_data.csv', index=False)
                interpolate.to_csv(self.cwd+f'/output/filtered/{name}_interpolate.csv', index=False)
            
            return filtered_df, interpolate

        else:

            pass

    def check_spectra_availability(self, interpolate, filtered_df):

        """
            Checks whether we have all spectra needed for interpolation based on the number of parameters to interpolate.

            :param filtered_df: DataFrame containing all spectra needed for interpolation and their parameters' values.
            :type filtered_df: pandas.DataFrame
            :return: True of False
            :rtype: bool

        """

        print('\nChecking spectra availability...')
        n = interpolate.sum().sum() # number of parameters to interpolate

        N = len(filtered_df) # number of spectra available

        if N != 2**n:
            print(f'Missing {2**n - N} spectra for {self.name}')
            return False

        else:
            print('All spectra available!')
            return True

def filter_models():

    # Starts Filter class
    filter = Filter()

    filtered_df, interpolate = filter.filter_all(save_all_models=False, save_result=False)

    print('\nParameters to interpolate:\n')

    for param in list(interpolate.columns):

        if interpolate[param].item():
            print('->', param)

    print('\nModel step for each parameter:\n')

    print(filter.delta_params)

    print('\nSpectra needed for interpolation:\n')
    print(filtered_df)

    if filter.check_spectra_availability(interpolate, filtered_df):

        print('='*34+' Finish '+'='*34+'\n')

    else:
        pass

if __name__ == "__main__":
    filter_models()