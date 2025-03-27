from importlib import reload 
import lib.core as functions
reload(functions) 
from lib.core import *
import os
cwd = os.getcwd()


def main():

    targets = pd.read_csv('teff_logg_feH.csv')

    wav_ref, flux_ref = np.loadtxt(cwd+'/database/lte050-4.5-0.0a+0.0.BT-NextGen.7.dat.txt', unpack=True)

    path = cwd+'/example/'

    corot1_data = pd.read_csv(path+'corot-1_data.csv')
    corot1_interp = pd.read_csv(path+'corot-1_interpolate.csv')
            

    flux_interp = interp(targets.iloc[0,:], wav_ref, corot1_data, corot1_interp, path, show_graphs=True, save_fig=True)

if __name__ == "__main__":
    main()