import configparser
import os

cwd = os.getcwd()
example_targets_path = cwd+'/stars.csv'
example_database_path = cwd+'/database/'
example_ref_spectrum_path = cwd+'/example/lte050-4.5-0.0a+0.0.BT-NextGen.7.dat.txt'


def create_config(config_path='config.ini'):
    
    config = configparser.ConfigParser()
    
    # Get user input data
    targets_path = input("Enter targets' file path [/.../your_file.csv]: ") or example_targets_path
    db_path = input("Enter database path [/your_db_folder/]: ") or example_database_path
    lib_name = input("Enter library name [BT-Settl]: ") or "BT-Settl"
    ref_spectrum = input("Enter reference spectrum path [/.../your_spectrum.txt]:") or example_ref_spectrum_path

    # Get user settings data
    save_result = input("Save final spectra [True/False]?") or 'True'
    
    # Set configuration values
    config['USER_DATA'] = {
        'targets_path': targets_path,
        'database_path': db_path,
        'library_name': lib_name,
        'reference_spectrum': ref_spectrum
    }
    
    config['SETTINGS'] = {
        'save_final_spectra': save_result
    }
    
    # Write to file
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    print(f"Configuration saved to {config_path}")

if __name__ == "__main__":
    create_config()