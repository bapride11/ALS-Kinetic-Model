# StaticCell.py
# Author: Greg Jones
# Version: 1.0.0

# This code is designed to be imported and run inside a Jupyter notebook using an iPython kernel.

import numpy as np
import pandas as pd
import ALS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, clear_output


df_photolysis_params = {}
df_photolysis_params['C3H3Br'] = {'xsn':1.0e-19, 'product1':'C3H3', 'product2':'Br'}

class StaticCell(ALS.KineticModel):
    def __init__(self, user_model):
        super().__init__(user_model, err_weight=False, fit_pre_photo=False, apply_IRF=False, apply_PG=False, t_PG=1.0)
    
    def plot_model(self, t_start, t_end, tbin, df_model_params, initial_concentrations, df_photolysis_params, fluence, photolysis_cycles=1, delta_xtick=20.0, save_fn=None):
        df_ALS_params = {'t0': 0}
        #print(initial_concentrations)
        # = df_model_params.to_dict('index').copy()
        #print(type(dict_mod_params))
        c_0 = initial_concentrations.copy()
        
        t_full = np.array([])
        
        
        for cycle in range(0, photolysis_cycles):
            c_after_photo = c_0.copy()
            for species in df_photolysis_params.itertuples():
                dC = species.xsn * fluence * c_0[species.Index]
                c_after_photo[species.Index]    = c_0[species.Index] - dC
                for product, qyield in zip(species.products, species.qyields):
                    c_after_photo[product] = c_0[product] + dC * qyield
            
            #print(c_after_photo)
            #for key in c_after_photo:
            #    dict_mod_params['c_' + key + '_0'] = {'val':c_after_photo[key], 'err':0, 'fit':False}
            
            c_params = {}
            for key in c_after_photo:
                c_params['c_' + key + '_0'] = {'val':c_after_photo[key], 'err':0, 'fit':False}
            
            df_mod_params = df_model_params.append(pd.DataFrame.from_dict(c_params, orient='index'))
            if cycle == 0:
                t_model, c_model = self._model(t_start, t_end, tbin, df_mod_params['val'].to_dict(), df_ALS_params)
                endtime = 0
            else:
                t_model, c_model = self._model(0, t_end, tbin, df_mod_params['val'].to_dict(), df_ALS_params)
            
            t_model = t_model + endtime
            endtime = t_model[-1]
            
            #print("New endtime", endtime)
            
            t_full = np.concatenate((t_full,t_model[:-1]))
            
            #print("Tfull", t_full)
            
            if cycle == 0:
                #c_full = c_model.to_dict('list').copy()
                c_full = c_model.copy().iloc[:-1]
            else:
                c_full = c_full.append(c_model.iloc[:-1], ignore_index=True)
                #for key in c_model:
                #    c_full[key] = np.concatenate((c_full[key], c_model[key]))
            
            
            #for key in c_model:
            #    #print(key)
            #    c_0[key] = list(c_model[key])[-1]
            for (species, concentrations) in c_model.iteritems():
                #print(species)
                #print(type(concentrations))
                c_0[species] = concentrations.iloc[-1]
            #print(c_0['HO2'])
        species_names = list(c_model.columns)
        nSpecies = len(species_names)
        
        # Set up the grid of subplots
        ncols = 3
        nrows = (nSpecies//ncols) if (nSpecies%ncols) == 0 else (nSpecies//ncols)+1
        dpi = 120
        
        plt.rc('font', size=9)
        plt.rc('axes.formatter', useoffset=False)
        f = plt.figure(figsize=(1000/dpi,325*nrows/dpi), dpi=dpi)
        gs = gridspec.GridSpec(nrows, ncols, figure=f, hspace=0.45, wspace=0.3, top=0.9, bottom=0.2)
        
        # Determine x-axis ticks
        tick_low = (t_start//delta_xtick)*delta_xtick
        tick_high = endtime if endtime % delta_xtick == 0. else ((endtime//delta_xtick)+1)*delta_xtick
        ticks = np.linspace(tick_low, tick_high, num=round(((tick_high-tick_low)/delta_xtick)+1), endpoint=True)
        
        # Make the subplots
        s_model = []
        for i, species in enumerate(species_names):
        	mod = c_full[species]
        	s_model.append(mod)
        
        	j = i // 3	# Row index
        	k = i % 3	# Col index
        
        	ax = plt.subplot(gs[j,k])
        	ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        	ax.plot(t_full, mod, linewidth=2)
        
        	# Manually set x-axis ticks
        	ax.set_xticks(ticks)
        
        	# Labels
        	ax.set_title(species, fontweight='bold')	 # Make the title the species name
        	ax.set_xlabel('Time (ms)')					 # Set x-axis label for bottom plot
        	if k == 0:									 # Set y-axis labels if plot is in first column
        		ax.set_ylabel('Concentration ($\mathregular{molc/cm^{3}})$')
        
        plt.show()
        
        # Save the model traces
        if save_fn:
        	df = pd.DataFrame(s_model).T
        	df.insert(0,'t',t_full)
        	df.to_csv(save_fn, index=False)
	
            