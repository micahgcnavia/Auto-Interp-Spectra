import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import os
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

def graph(name, interp, wav_ref, cwd, save_fig=False):

    """
    Plots raw and interpolated spectra.
    name: string com o nome do alvo
    interp: dicionários com todas as informações dos espectrus crus e interpolados
    """
    x_lower, x_higher = [6552.5/1e4, 6657/1e4]
    y_lower, y_higher = [0.7e7/1e4, 0.9e7/1e4]

    lw=1

    path = cwd+'/output/plots/'

    if 'interp_spec' not in interp.keys():

        x_lower, x_higher = [6552.5/1e4, 6657/1e4]
        y_lower, y_higher = [0.6e7/1e4, 0.73e7/1e4]

        fig, ax = plt.subplots(1,figsize=(14,6))

        fig.set(facecolor='white')

        plt.title('Interpolation for '+name, fontsize='xx-large', y=1)
        
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

            try:

                os.mkdir(path)

            except:
                print('OPS')
        
            plt.savefig(path+'interp_'+name+'.png', bbox_inches = 'tight',dpi = 300)
        
        plt.show()

    else:
    
        fig, (ax1, ax2) = plt.subplots(2,figsize=(18,12), sharex=True)
    
        fig.set(facecolor='white')
    
        fig.suptitle('Interpolation for '+name, fontsize='xx-large',x=0.5,y=0.97)
        
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

            try:

                os.mkdir(path)

            except:
                pass
    
            plt.savefig(path+'interp_'+name+'.png', bbox_inches = 'tight',dpi = 300)
        
        plt.show()