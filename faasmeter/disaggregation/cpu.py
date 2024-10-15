import os

import numpy as np 
import pandas as pd

from disaggregation.Linear_Regression import LinearModel

from disaggregation.disagg_i import Disagg_I
from helper_funcs import *
from tags import *
from config import *

class CPU(Disagg_I):

    def __init__( self, lg, N_init, N, delta, threshold, pcol ):
        #P-Suggest: Annotate the args with types. lg==?, or what the typical ranges should be for each 
        self.lg = lg
        self.N_init = N_init #P-Q: Same N_init as KF? 
        self.N = N
        self.delta = delta
        self.threshold = threshold
        self.pcol = pcol
        self.log_loc = lg.log_loc + dir_cpushare 
        os.system('mkdir -p ' + self.log_loc)
        #  os.system('rm ' + self.log_loc + '/* > /dev/null 2>&1')

    ##############################

    def _fit_transform( self,  p_col='perf_rapl' ):
        # This is offline. 
        delta = self.delta 
        N_init = self.N_init
        N = self.N
        lg = self.lg
        power_cpu = p_col
        t = 'timestamp' 
        tg = 'target'
        
        features = lg.cpu_config['cfeatures']                
        threshold = self.threshold 
        
        # separate out the N_init x and y 
        cmb = lg.combined_hwpc                                

        # aggregate - features for each timestamp             
        acmb = cmb.groupby(t).aggregate(sum)        
        acmb_mean = cmb.groupby(t).aggregate('mean')

        x_agg = acmb[features]                               
        y_act = acmb_mean[power_cpu]                      

        x_agg = np.array( x_agg )                           
        y_act = np.array( y_act )                     
        
        m = LinearModel('SVR', lg.cpu_config['history_size'] ) # 'offline'
        
        # build model using N_init initial seconds 
        m.train( x_agg[0:N_init,:], y_act[0:N_init] )
        m.add_history( x_agg[0:N_init,:], y_act[0:N_init] )
        
        yps = None
        
        # traverse through the datapoints one by one 
        # for i in range(N_init, N_init+3):
        for i in range(N_init, x_agg.shape[0]):
            # This needs to be batched and averaged  
            x_all = x_agg[i,:]
            y_a = y_act[i]    
            
            tm = acmb.iloc[i,:]
            
            # add the point to model history 
            m.add_history( x_all, y_a )
            
            # predict using the model 
            yp_a = m.predict( [x_all] ) 
            
            # if error is too much 
            if yp_a - y_a > threshold:
                # rebuild the model  
                m.train_on_history()
                continue
                
            # predict for each target
            x_s = cmb[cmb[t] == tm.name]
            idle_power = m.predict( [np.zeros(x_all.shape)] )
            yos_cp = m.predict( x_s[ x_s['target'] == 'os_cp'][features] )

            for tg_l, vals in x_s.groupby(tg):
                if tg_l == 'os_cp':
                    # we are distributing it among functions
                    continue
                fvals = vals[features]
                fdvals = x_all - fvals
                
                yp = m.predict( fvals )
                ypn = m.predict( fvals )
                ypni = ypn - idle_power
                yp = ypni + (ypni * idle_power)/ yp_a
                yp += (ypni * yos_cp)/ yp_a
                
                r = [tm.name, tg_l, float(yp)]  
                if yps is None:
                    yps = np.array([r])
                else:
                    yps = np.concatenate([yps,[r]], axis=0)
        
        yps = pd.DataFrame( yps, columns=[t,tg,power_cpu] )
        self.cpu_share = yps 

    ########################################
        
    def online_widrow_hoff(self, p_col='perf_rapl', eta=0.1):
        # 
        delta = self.delta 
        N_init = self.N_init
        N = self.N
        lg = self.lg
        power_cpu = p_col
        t = 'timestamp' 
        principals = 'target'
        batch = 10 
        features = lg.cpu_config['cfeatures']                
        threshold = self.threshold 
        
        # separate out the N_init x and y 
        cmb = lg.combined_hwpc                                

        # aggregate - features for each timestamp             
        acmb = cmb.groupby(t).aggregate(sum)        
        acmb_mean = cmb.groupby(t).aggregate('mean')

        x_agg = acmb[features]                               
        y_act = acmb_mean[power_cpu]                      

        x_agg = np.array(x_agg)                           
        y_act = np.array(y_act)   # Y_actual                   
        
        #m = LinearModel('SVR') #, lg.cpu_config['history_size'])
        m = LinearModel('online') # Least Means Squares
        
        # build model using N_init initial seconds 
        m.train(x_agg[0:N_init,:], y_act[0:N_init])
        yps = None
        
        # traverse through the datapoints in a batch 
        # for i in range(N_init, N_init+3):
        for i in range(N_init, x_agg.shape[0], batch):
            # needs to be batched 
            x_all = np.mean(x_agg[i:batch,:], axis=0) # axis? 
            y_a = np.mean(y_act[i:batch])
            
            tm = acmb.iloc[i+batch,:] #Time? What if overflow? 
            # predict using the model 
            Wts = m.update(x_all, y_a)
            
            # predict for each target
            x_s = cmb[cmb[t] == tm.name]
            idle_power = m.predict([np.zeros(x_all.shape)] )
            # This is going to be zero with the dot product 
            yos_cp = m.predict(x_s[ x_s['target'] == 'os_cp'][features])

            for tg_l, vals in x_s.groupby(tg):
                if tg_l == 'os_cp':
                    # we are distributing it among functions
                    continue
                fvals = vals[features]
                fdvals = x_all - fvals
                
                yp = m.predict( fvals )
                ypn = m.predict( fvals )
                ypni = ypn - idle_power
                yp = ypni + (ypni * idle_power)/ yp_a
                yp += (ypni * yos_cp)/ yp_a
                
                r = [tm.name, tg_l, float(yp)]  
                if yps is None:
                    yps = np.array([r])
                else:
                    yps = np.concatenate([yps,[r]], axis=0)
        
        yps = pd.DataFrame(yps, columns=[t,principals,power_cpu])
        self.cpu_share = yps 

        
    ########################################
        
    def condense_to_mins( self, cpu_share, pcol ):
        cpu_share[tag_tm] = translate_t_to_mins( cpu_share )
        uts = set(cpu_share[tag_targets])
        series = []
        for t in uts:
            cpu_share_t = get_givenpattern( cpu_share, tag_targets, t )
            cond_t = cpu_share_t.groupby(tag_tm)[pcol].mean()
            series.append( cond_t )
        s = pd.concat(series, axis=1)
        s.columns = uts
        return s

    def process(self):
        self._fit_transform( self.pcol )
        self.cpu_share_mins = self.condense_to_mins( self.cpu_share, self.pcol )

    def save_dfs(self):
        save_df( self.cpu_share, self.log_loc + '/' + self.pcol )
        save_df( self.cpu_share_mins, self.log_loc + '/' + self.pcol + '_mins' )

