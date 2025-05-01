import pandas as pd
import numpy as np
import time
import ast
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from lib.get_input_data import Input

class Scraper():

    """
        Gets missing spectra from the SVO theoretical models database.

    """

    def __init__(self):

        """
            Initializes the Scraper class.

        """
        Input.__init__(self)

        print('='*27+' Initializing Scraper '+'='*26+'\n')

        # Setting up driver
        self.service = Service()
        self.options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        url = f'https://svo2.cab.inta-csic.es/theory/newov2/index.php?models={self.model.strip().lower()}'
        self.driver.get(url)

    def get_intervals(self,  teff_min, teff_max,
                        logg_min, logg_max,
                        feh_min, feh_max):

        intervals = {
        'teff': {'min': teff_min, 'max': teff_max},
        'logg': {'min': logg_min, 'max': logg_max},
        'meta': {'min': feh_min,  'max': feh_max}
        }

        return intervals

    def get_param_range(self, param):

        model = self.model.strip().lower()

        range_list = self.driver.find_element(By.NAME, f"params[{model}][{param}][min]").text.split('\n')

        return np.array([float(value) for value in range_list])

    def get_param_step(self, param_range):

        return {i: abs(i-j) for i, j in zip(param_range, param_range[1:])}

    def get_closest_value(self, value, array):

        if value > max(array) or value < min(array):

            print('Input value not in the parameter range.')
            return None

        else:

            idx = (np.abs(array - value)).argmin()
            return array[idx]

    def get_delta_params(self):

        delta_params = {}

        for param in self.parameters:

            param_range = self.get_param_range(param)
            param_step = self.get_param_step(param_range)

            delta_params[param] = param_step.get(self.get_closest_value(self.params[param], param_range))

        return delta_params


    def select_value(self, key, intervals, limit='min', delay=None):

        model = self.model.strip().lower()

        selection = Select(self.driver.find_element(By.NAME, f'params[{model}][{key}][{limit}]'))
        selection.select_by_visible_text(str(intervals[key][limit]))

        time.sleep(delay)

    def search(self, delay=None):

        search_button = self.driver.find_element(By.NAME, 'nres')
        select_search = Select(search_button)

        # Selecting all spectra available
        select_search.select_by_value('all')

        time.sleep(delay)

        XPATH = '/html/body/div[5]/table/tbody/tr/td/div/form/table/tbody/tr[1]/td[1]/table/tbody/tr[5]/td/input'

        self.driver.find_element(By.XPATH, XPATH).click()

        time.sleep(delay)

    def retrieve(self, delay=None):

        mark_all_ASCII = '/html/body/div[5]/table/tbody/tr/td/div/form/table/tbody/tr[1]/td[2]/table[1]/tbody/tr/td[1]/input'
        retrieve_button = '/html/body/div[5]/table/tbody/tr/td/div/form/table/tbody/tr[1]/td[2]/table[1]/tbody/tr/td[4]/input'

        self.driver.find_element(By.XPATH, mark_all_ASCII).click()
        time.sleep(delay)

        self.driver.find_element(By.XPATH, retrieve_button).click()

def scrap():

    scraper = Scraper()

    intervals = scraper.get_intervals(
                        teff_min = 5100,
                        teff_max = 5200,
                        logg_min = 4,
                        logg_max = 4.5,
                        feh_min = 0,
                        feh_max = 0.5
    )

    print(f'Parameter range for {scraper.model}:')

    labels = {param: name for param, name in zip(list(intervals.keys()), ['Effective temperature (K)',
                                                                                  'Surface gravity (dex)',
                                                                                  'Metallicity (dex)'
                                                                                 ])}

    for param in list(intervals.keys()):

        print(f'\n{labels[param]}:\n')
        print(scraper.get_param_range(param))

        scraper.select_value(param, intervals, limit='min', delay=0)
        scraper.select_value(param, intervals, limit='max', delay=0)

    scraper.search(delay=0)
    scraper.retrieve(delay=2)

    # time.sleep(3)

    
if __name__ == "__main__":
    scrap()

