import pandas as pd
import numpy as np
from lib.get_config import *
from glob import glob
import ast

class Input:

    """
        Retrieves data from the config.ini file.
    
    """

    def __init__(self):

        # Current working directory
        self.cwd = os.getcwd()

        # Getting user data
        config = get_config()

        # Save spectra flag
        self.save_final_spectra = config.getboolean('SETTINGS', 'save_final_spectra')

        # Getting targets
        targets_path = config['USER_DATA']['targets_path']
        targets = pd.read_csv(targets_path)
        target = targets.loc[targets['star'] == 'CoRoT-1'] # This will be replaced by a loop in the future
        objct = list(target.columns)[0]
        self.name = target[objct].item().strip() # Target name

        # Getting parameters
        self.params = {
            'teff': target['teff'].item(),
            'logg': target['logg'].item(),
            'meta': target['meta'].item()
        }
        self.parameters = list(self.params.keys())

        # Getting database and model info
        database_path = config['USER_DATA']['database_path']
        self.model = config['USER_DATA']['library_name']
        self.models_list = sorted(glob(database_path+self.model.lower()+'/*'))

        # Importing the reference spectrum
        wav_ref_path = config['USER_DATA']['reference_spectrum']
        self.wav_ref, _ = np.loadtxt(wav_ref_path, unpack=True)