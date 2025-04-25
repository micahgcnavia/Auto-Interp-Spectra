from importlib import reload 
import lib.interpolate as functions
reload(functions) 
from lib.interpolate import *
import os

def main():
    """
        Runs the interpolation pipeline.

        :return: A list of DataFrames containing updated parameters and fluxes at each interpolation step.
        :rtype: list[pandas.DataFrame]
    
    """

    cwd = os.getcwd()
    path = cwd+'/output/filtered/'

    # Prepare inputs
    wav_ref, _ = np.loadtxt(cwd+'/database/bt-settl/lte050-4.5-0.0a+0.0.BT-NextGen.7.dat.txt', unpack=True)
    targets = pd.read_csv('stars.csv') # List of objects to interpolate
    target = targets.loc[targets['star'] == 'CoRoT-1']
    spectra = pd.read_csv(path+'corot-1_data.csv')  # Your spectra data
    interpolate_flags = pd.read_csv(path+'corot-1_interpolate.csv')

    # Initialize interpolator
    interpolator = SpectrumInterpolator(wav_ref=wav_ref, target=target)

    # Perform interpolation
    result = interpolator.interpolate_spectra(
        spectra=spectra,
        interpolate_flags=interpolate_flags,
        save_file=True
    )

    return result

if __name__ == "__main__":
    main()
