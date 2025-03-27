import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from glob import glob
from natsort import os_sorted
import os
from tqdm import tqdm
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

cwd = os.getcwd()

def interp_partial(spectrum1, spectrum2, factor, delta_param):

    """
    Essa função tem a forma de uma interpolação para parâmetros quaisquer.
    Retorna o espectro interpolado.

    spectrum1 tem sempre o valor mínimo do parâmetro que será interpolado.
    delta_param: variação do parâmetro (pode variar com diferentes grades de modelos). 
    Exemplo: para BT-Settl, Δlogg = 0.5.
    factor: valor do gabarito - valor mínimo do parâmetro do modelo. Exemplo:
    Gabarito: logg = 4.3. 
    factor = 4.3 - 4.0 = 0.3
    """
    
    return spectrum1 + (((spectrum2 - spectrum1)*factor)/delta_param)

def graph(name, interp, wav_ref, save_fig=False):

    """
    Essa função mostra os passos da interpolação para o alvo escolhido.
    name: string com o nome do alvo
    interp: dicionários com todas as informações dos espectrus crus e interpolados
    """
    x_lower, x_higher = [6552.5/1e4, 6657/1e4]
    y_lower, y_higher = [0.7e7/1e4, 0.9e7/1e4]

    lw=1

    if 'interp_spec' not in interp.keys():

        x_lower, x_higher = [6552.5/1e4, 6657/1e4]
        y_lower, y_higher = [0.6e7/1e4, 0.73e7/1e4]

        fig, ax = plt.subplots(1,figsize=(14,6))

        fig.set(facecolor='white')

        plt.title('Interpolação para '+name, fontsize='xx-large', y=1)
        
        #ax.yaxis.tick_right()

        for raw_spec, raw_param in zip(interp['raw_spec'], interp['raw_params']):

            ax.plot(wav_ref/1e4,raw_spec/1e4, lw=lw, label = '$T_{ef}$ = '+str(int(raw_param[0]))+' K, log g = '+
                     str(raw_param[1]).replace('.', ',')+', [Fe/H] = '+str(raw_param[2]).replace(',', '.'))

        ax.plot(wav_ref/1e4,interp['final_spec']/1e4, lw=lw, label = '$T_{ef}$ = '+str(int(interp['final_params'][0]))+' K, log g = '+
                 str(interp['final_params'][1]).replace('.', ',')+', [Fe/H] = '+str(interp['final_params'][2]).replace(',', '.'))
        
        ax.legend(fontsize = 'x-large', loc='upper center', bbox_to_anchor=(0.5, 1.0),
               fancybox=False, ncol=3, frameon=False)
    
        ax.set_xlim(x_lower, x_higher)
        ax.set_ylim(y_lower, y_higher)
        ax.locator_params(axis='y', nbins=5)
        plt.tick_params(labelcolor='k', which='both', top=False,
                    bottom=True, left=False, right=True, labelsize='large')
    
        plt.xlabel("Comprimento de onda (μm)",fontsize = 'x-large')
        plt.ylabel("Fluxo (erg/cm^2/s/μm)",fontsize = 'x-large')
    
        plt.tight_layout()
    
        if save_fig:
        
            plt.savefig(cwd+'/interpolacao_'+name+'.png', bbox_inches = 'tight',dpi = 300)
        
        plt.show()

    else:
    
        fig, (ax1, ax2) = plt.subplots(2,figsize=(18,12), sharex=True)
    
        fig.set(facecolor='white')
    
        fig.suptitle('Interpolação para '+name, fontsize='xx-large',x=0.5,y=0.97)
        
        #ax1.yaxis.tick_right()
    
        for raw_spec, raw_param in zip(interp['raw_spec'], interp['raw_params']):
    
            ax1.plot(wav_ref/1e4,raw_spec/1e4, lw=lw, label = '$T_{ef}$ = '+str(int(raw_param[0]))+' K, log g = '+
                     str(raw_param[1]).replace(',', '.')+', [Fe/H] = '+str(raw_param[2]).replace(',', '.'))

        line_styles = ['--', '-', '-.', ':']
        
        for interp_spec, interp_param, ls in zip(interp['interp_spec'], interp['interp_params'], line_styles):
    
            ax1.plot(wav_ref/1e4,interp_spec/1e4, c='k', ls=ls, lw=lw, label = '$T_{ef}$ = '+str(int(interp_param[0]))+' K, log g = '+
                     str(interp_param[1]).replace(',', '.')+', [Fe/H] = '+str(interp_param[2]).replace(',', '.'))
        
        ax1.legend(fontsize = 17, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   fancybox=False, ncol=3, frameon=False)
        
        ax1.set_xlim(x_lower, x_higher)
        ax1.set_ylim(y_lower, y_higher)
        ax1.locator_params(axis='y', nbins=5)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        plt.setp(ax1.get_yticklabels()[0], visible=False)
            
        #ax2.yaxis.tick_right()
    
        if 'interp2_spec' not in interp.keys():
    
            for interp_spec, interp_param in zip(interp['interp_spec'], interp['interp_params']):
        
                ax2.plot(wav_ref/1e4,interp_spec/1e4, lw=lw, label = '$T_{ef}$ = '+str(int(interp_param[0]))+' K, log g = '+
                         str(interp_param[1]).replace(',', '.')+', [Fe/H] = '+str(interp_param[2]).replace(',', '.'))
    
        else:
    
            for interp_spec, interp_param in zip(interp['interp_spec'], interp['interp_params']):
    
                ax2.plot(wav_ref/1e4,interp_spec/1e4, lw=lw, label = '$T_{ef}$ = '+str(int(interp_param[0]))+' K, log g = '+
                         str(interp_param[1]).replace(',', '.')+', [Fe/H] = '+str(interp_param[2]).replace(',', '.')) 
    
            for interp2_spec, interp2_param in zip(interp['interp2_spec'], interp['interp2_params']):
    
                ax2.plot(wav_ref/1e4,interp2_spec/1e4, lw=lw, label = '$T_{ef}$ = '+str(int(interp2_param[0]))+' K, log g = '+
                         str(interp2_param[1]).replace(',', '.')+', [Fe/H] = '+str(interp2_param[2]).replace(',', '.'))
    
        ax2.plot(wav_ref/1e4,interp['final_spec']/1e4, c='k', lw=lw, label = '$T_{ef}$ = '+str(int(interp['final_params'][0]))+' K, log g = '+
                     str(interp['final_params'][1]).replace(',', '.')+', [Fe/H] = '+str(interp['final_params'][2]).replace(',', '.'))
        
        ax2.legend(fontsize = 17, loc='upper center',bbox_to_anchor=(0.5, 1.0),
                  fancybox=False, ncol=3, frameon=False)
    
        ax2.set_xlim(x_lower, x_higher)
        ax2.set_ylim(y_lower, y_higher)
        ax2.locator_params(axis='y', nbins=5)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        
        fig.add_subplot(111, frameon=False)
        
        plt.tick_params(labelcolor='none', which='both', top=False,
                        bottom=False, left=False, right=False)

        plt.xlabel("Wavelength (μm)",fontsize = 26, labelpad=12)
        plt.ylabel("Flux (erg/s/cm²/μm)",fontsize = 27, labelpad=22)
    
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
    
        if save_fig:
    
            plt.savefig(cwd+'/interpolacao_'+name+'.png', bbox_inches = 'tight',dpi = 300)
        
        plt.show()



def interp(target, wav_ref, spectra, interpolate, path, show_graphs=False, save_file=False, save_fig=False):
    """
    target: tabela com os parâmetros do objeto a ser interpolado
    wav_ref: passo de comprimento de onda de referência
    spectra: objeto do tipo filter()[0], i.e., lista de espectros que serão usados para interpolar
    interpolate: objeto do tipo filter()[1], i.e., dicionário com as flags para interpolar ou não x parâmetro
    show_graphs: flag para mostrar ou não os gráficos da interpolação
    save_file: flag para salvar o espectro interpolado
    save_fig: flag para salvar os gráficos
    path: caminho da pasta onde os arquivos serão salvos

    Retornáveis:
    comprimento de onda de referência e fluxo interpolado
    gráfico mostrando os passos da interpolação (opcional)
    """
    # Helper functions
    def get_extreme_spectra(params):
        """Get spectra with extreme parameter values based on interpolation flags"""
        conditions = []

        for param in interpolate.items():
            if param[1][0]:
                extreme = 'min' if params[param[0]] == 'min' else 'max'
                col = 'teff' if param[0] == 'teff' else ('logg' if param[0] == 'logg' else 'feh')
                val = min(spectra[col]) if params[param[0]] == 'min' else max(spectra[col])
                conditions.append(spectra[col] == val)
        
        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition &= cond
            
        return spectra.loc[combined_condition]

    def load_and_interpolate_spectrum(spectrum_row):
        """Load spectrum from file and interpolate to reference wavelength"""
        wav, spec = np.loadtxt(cwd+spectrum_row['path'].item(), unpack=True)
        return np.interp(wav_ref, wav, spec)

    def perform_interpolation(spectra_pairs, params, deltas):
        """Perform interpolation steps for given spectra pairs and parameters"""
        spectra_list = [load_and_interpolate_spectrum(pair) for pair in spectra_pairs]
        
        # Perform interpolation for each parameter that needs it
        current_spectra = spectra_list
        interp_steps = {'raw_spec': current_spectra}

        # Initialize parameters tracking with the raw spectra parameters
        current_params = [
            (row['teff'], row['logg'], row['feh']) 
            for row in spectra_pairs
        ]
        
        for i, param in enumerate(interpolate.items()):
            if not param[1][0]:
                continue
                
            if len(current_spectra) % 2 != 0:
                raise ValueError("Odd number of spectra for interpolation")
            
            new_spectra = []
            new_params = []
            
            for j in range(0, len(current_spectra), 2):
                spec1 = current_spectra[j]
                spec2 = current_spectra[j+1]
                
                # Perform the interpolation
                interpolated = interp_partial(
                    spec1, spec2, 
                    params[param[0]] - min(spectra[param[0]]), 
                    deltas[param[0]]
                )
                new_spectra.append(interpolated)
                
                # Create new parameter set for the interpolated spectrum
                param_values = {}
                for p in interpolate:
                    if p == param[0]:
                        param_values[p] = params[p]  # This is the interpolated value
                    else:
                        # Take the value from either spectrum (they should be the same for non-interpolated params)
                        param_values[p] = current_params[j][list(interpolate.keys()).index(p)]
                
                new_params.append(tuple(param_values[p] for p in interpolate))
            
            current_spectra = new_spectra
            current_params = new_params
            interp_steps[f'interp{i+1}_spec'] = current_spectra
            interp_steps[f'interp{i+1}_params'] = current_params
    
        return current_spectra[0], interp_steps

    # Main function logic
    name = target['star'].replace(' ', '').lower()
    params = {
        'teff': target['teff'].item(),
        'logg': target['logg'].item(),
        'feh': target['feh'].item()
    }
    
    # This might change for other models
    deltas = {
        'teff': 100, # [K]
        'logg': 0.5, # [dex]
        'feh': 0.5 # [dex]
    }
    
    # Determine which parameters to interpolate
    interp_params = [p for p in interpolate if interpolate[p][0]]
    
    # Get all combinations of min/max for parameters we're interpolating
    from itertools import product
    extremes = list(product(*[['min', 'max'] if interpolate[p][0] else ['fixed'] for p in interpolate]))
    
    # Get the extreme spectra
    extreme_spectra = []
    for combo in extremes:
        param_values = {p: combo[i] for i, p in enumerate(interpolate)}
        extreme_spectra.append(get_extreme_spectra(param_values))
    
    # Check if we have all required spectra
    if any(len(s) == 0 for s in extreme_spectra):
        print(f'Missing spectra for {name}')
        return None
    
    # Perform the interpolation
    try:
        final_spectrum, interp_steps = perform_interpolation(extreme_spectra, params, deltas)
    except ValueError as e:
        print(f'Interpolation error for {name}: {str(e)}')
        return None
    
    # Prepare the return dictionary
    interp_steps['final_params'] = [params[p] for p in interpolate]
    interp_steps['final_spec'] = final_spectrum
    interp_steps['raw_params'] = [(row['teff'], row['logg'], row['feh']) for _, row in spectra.iterrows()]
    
    # Handle output options
    if show_graphs:
        graph(name, interp_steps, wav_ref, save_fig=save_fig)
    
    if save_file:
        df = pd.DataFrame({'wavelength': wav_ref, 'flux': final_spectrum})
        df.to_csv(f"{path}{name}_interp.csv", index=False)
    
    return interp_steps

