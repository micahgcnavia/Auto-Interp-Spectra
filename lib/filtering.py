import pandas as pd
import numpy as np
from glob import glob
from natsort import os_sorted
import os
from tqdm import tqdm
from more_itertools import pairwise
from functools import reduce
from collections import defaultdict

class Filter():

    """
        Filters the database and gets the missing spectra from the Spanish Virtual Observatory (SVO), if any.

    """

    def __init__(self, database_path, target, delta_params=None):

        """
            Initializes the Filter class.
            
            :param database_path: Path of the folder that contains all models.
            :type database_path: str
            :param target: DataFrame containing the name of the object and its parameters.
            :type target: pandas.DataFrame
            :param delta_params: Optional DataFrame describing the parameters steps for the model.
            :type delta_params: pandas.DataFrame

        """

        print('='*27+' Initializing Filter '+'='*27+'\n')
        self.cwd = os.getcwd()
        self.database_path = database_path
        self.models_list = sorted(glob(self.database_path+'*'))
        self.target = target
        self.params = {
            'teff': self.target['teff'].item(),
            'logg': self.target['logg'].item(),
            'feh': self.target['feh'].item()
        }
        self.parameters = list(self.params.keys())
        self.delta_params = delta_params or {
            'teff': 100,  # [K]
            'logg': 0.5,  # [dex]
            'feh': 0.5    # [dex]
        }
        self.interpolate = {
            'teff': None,
            'logg': None,
            'feh': None}
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
        
        param_name = [header[i].split(' ')[1] for i in np.arange(1,len(header))]
        
        header_dict = {}

        for i, param in zip(np.arange(1,len(header)), param_name):
        
            header_dict[param] = float(header[i].split(' ')[3])
            
        if show_model:
            
            if header[0].split(' ')[1] == 'BT-NextGen':
                
                print(header[0].split(' ')[1]+' '+header[0].split(' ')[2].split('\n')[0])
                
            else:
                
                print(header[0].split(' ')[1])

        header_dict['path'] = file_path

        if header_dict['alpha'] != 0:

            print('='*80)
            print('Found alpha != 0 in path:', file_path)

        else:
            pass

        return header_dict

    def get_all_models(self, save=False):

        """
            Stores all models in a pandas DataFrame.
            
            :param save: Flag to indicate whether to save the DataFrame in a CSV file.
            :type save: bool
            :return: DataFrame with the metadata of all models in the database.
            :rtype: pandas.DataFrame

        """

        models = [self.get_header_data(file) for file in tqdm(self.models_list, 'reading files...')]

        df = pd.DataFrame(models)

        df = df.rename(columns= {'meta': 'feh'}) # just to assure consistency with the interpolate.py script

        if save:

            df.to_csv('all_models.csv', index=False)

        return df

    def filter_param(self, df, param):

        """
            Filters the DataFrame with all models and looks for rows where there is a match for the input parameter
            or retrieves the rows necessary for interpolation based on the model's parameter step.

            :param df: DataFrame with the metadata of all models in the database.
            :type df: pandas.DataFrame
            :param param: Name of the parameter to filter with.
            :type param: str
        
        """

        if self.params[param] in list(df[param]):

            self.interpolate[param] = False
            self.conditions.append(df[param] == self.params[param])

        else:

            self.interpolate[param] = True
            self.conditions.append(abs(df[param] - self.params[param]) <= self.delta_params[param])

    def filter_all(self):

        """
        Applies the filtering for the target and its parameters. Creates auxiliary files for the interpolation script.

        """

        all_models = self.get_all_models()

        for param in self.parameters:

            self.filter_param(all_models, param)

        filtered_df = all_models[reduce(lambda x, y: x & y, self.conditions)]

        interpolate = pd.DataFrame(self.interpolate, index=[0])
        
        return filtered_df, interpolate

def main():

    cwd = os.getcwd()
    database_path = cwd+'/database/'

    targets = pd.read_csv('stars.csv') # List of objects to interpolate

    target = targets.loc[targets['star'] == 'CoRoT-1 ']

    filter = Filter(database_path=database_path, target=target)


    filtered_df, interpolate = filter.filter_all()
    print(filtered_df)
    print(interpolate)


if __name__ == "__main__":
    main()