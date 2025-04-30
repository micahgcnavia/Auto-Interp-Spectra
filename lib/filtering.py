import pandas as pd
import numpy as np
from glob import glob
from natsort import os_sorted
import os
import ast
from tqdm import tqdm
from more_itertools import pairwise
from functools import reduce
from collections import defaultdict
from lib.get_config import *

class Filter():

    """
        Filters the database and gets the missing spectra from the Spanish Virtual Observatory (SVO), if any.

    """

    def __init__(self):

        """
            Initializes the Filter class.

        """

        print('='*27+' Initializing Filter '+'='*27+'\n')

        # Current working directory
        self.cwd = os.getcwd()

        # Getting user data
        config = get_config()

        # Getting targets
        targets_path = config['USER_DATA']['targets_path']
        targets = pd.read_csv(targets_path)
        target = targets.loc[targets['star'] == 'CoRoT-1'] # This will be replaced by a loop in the future
        objct = list(target.columns)[0]
        self.name = target[objct].item().strip() # Target name

        # Getting parameters and model step
        self.delta_params = ast.literal_eval(config['USER_DATA']['delta_params']) # Getting the dictionary
        self.params = {
            'teff': target['teff'].item(),
            'logg': target['logg'].item(),
            'feh': target['feh'].item()
        }
        self.parameters = list(self.params.keys())

        # Getting database and model info
        database_path = config['USER_DATA']['database_path']
        self.library = config['USER_DATA']['library_name'].lower()
        self.models_list = sorted(glob(database_path+self.library+'/*'))
        
        #self.save_final_spectra = config.getboolean('SETTINGS', 'save_final_spectra')
        
        # Initializing interpolate dictionary
        self.interpolate = {
            'teff': None,
            'logg': None,
            'feh': None}

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

        df = df.rename(columns= {'meta': 'feh'}) # just to assure consistency with the interpolate.py script

        if save:

            df.to_csv(self.cwd+'/all_models_'+self.library+'.csv', index=False)

        return df

    def filter_param(self, df, param):

        """
            Filters the DataFrame with all models and looks for rows where there is a match for the input parameter
            or retrieves the rows necessary for interpolation based on the model's parameter step.

            :param df: DataFrame with the metadata of all models in the database.
            :type df: pandas.DataFrame
            :param param: Name of the parameter to filter by.
            :type param: str
        
        """

        update_database = {}

        # Checks if the input parameter value matches one of the models' default values
        if self.params[param] in list(df[param]):

            self.interpolate[param] = False
            self.conditions.append(df[param] == self.params[param])

        else:

            self.interpolate[param] = True
            # Gets values 1 step around the input parameter value
            self.conditions.append(abs(df[param] - self.params[param]) <= self.delta_params[param])

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
            all_models = pd.read_csv(self.cwd+f'/all_models_{self.library}.csv')

        except:

            # If there is no all_models file, creates one
            all_models = self.get_all_models(save=save_all_models)

        print(f'\nFiltering the database to get spectra for {self.name}\n')

        print(f'{self.name} parameters:\n')

        for key, item in self.params.items():

            print(key+f' = {item}')

        for param in (self.parameters):

            self.filter_param(all_models, param)

        filtered_df = all_models[reduce(lambda x, y: x & y, self.conditions)] # Makes sure all conditions are satisfied

        interpolate = pd.DataFrame(self.interpolate, index=[0]) # This will be a simple DataFrame stating which parameters to interpolate

        if save_result:

            name = self.name.lower()

            filtered_df.to_csv(self.cwd+f'/output/filtered/{name}_data.csv', index=False)
            interpolate.to_csv(self.cwd+f'/output/filtered/{name}_interpolate.csv', index=False)
        
        return filtered_df, interpolate

    def check_spectra_availability(self, filtered_df):

        """
            Checks whether we have all spectra needed for interpolation based on the number of parameters to interpolate.

            :param filtered_df: DataFrame containing all spectra needed for interpolation and their parameters' values.
            :type filtered_df: pandas.DataFrame
            :return: True of False
            :rtype: bool

        """

        print('\nChecking spectra availability...')
        n = self.interpolate_flags.sum().sum() # number of parameters to interpolate

        N = len(filtered_df) # number of spectra available

        if N != 2**n:
            print(f'Missing {2**n - N} spectra for {self.name}')
            return False

        else:
            print('All spectra available!')
            return True

def main():

    # Starts Filter class
    filter = Filter()

    filtered_df, interpolate = filter.filter_all(save_all_models=True, save_result=True)

    print('\nParameters to interpolate:\n')

    for param in list(interpolate.columns):

        if interpolate[param].item():
            print('->', param)

    if filter.check_spectra_availability(interpolate, filtered_df):

        print('\nSpectra needed for interpolation:\n')
        print(filtered_df)

        print('='*34+' Finish '+'='*34+'\n')

    else:
        pass

if __name__ == "__main__":
    main()