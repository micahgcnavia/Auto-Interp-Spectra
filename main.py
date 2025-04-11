from importlib import reload 
import lib.interpolate as functions
reload(functions) 
from lib.interpolate import *
import os

def main():

    cwd = os.getcwd()
    path = cwd+'/example/'

    # Initialize interpolator

    wav_ref, _ = np.loadtxt(cwd+'/database/lte050-4.5-0.0a+0.0.BT-NextGen.7.dat.txt', unpack=True)
    interpolator = SpectrumInterpolator(wav_ref=wav_ref)

    # Prepare inputs
    targets = pd.read_csv('stars.csv') # List of objects to interpolate
    target = targets.iloc[0,:]
    spectra = pd.read_csv(path+'corot-1_data.csv')  # Your spectra data
    interpolate_flags = pd.read_csv(path+'corot-1_interpolate.csv')

    # Perform interpolation
    result = interpolator.interpolate(
        target=target,
        spectra=spectra,
        interpolate_flags=interpolate_flags,
        cwd=cwd,
        show_graphs=False,
        save_fig=False,
        save_file=False
    )

    #print(result)

if __name__ == "__main__":
    main()
