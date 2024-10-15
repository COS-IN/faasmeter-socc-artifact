import os

import numpy as np 
import pandas as pd

from disaggregation.Kalman_Filter import Kalman_Filter

from disaggregation.disagg_i import Disagg_I
from helper_funcs import *
from tags import *
from config import *

class Combined(Disagg_I):

    def __init__( self, lg, N_init, N, delta, o_type, update_type, pcol, kf_type ='x' ):
        self.lg = lg
        self.N_init = N_init
        self.N = N
        self.delta = delta
        self.o_type = o_type
        self.update_type = update_type
        self.kf_type = kf_type
        self.pcol = pcol

        self.log_loc = lg.log_loc + dir_combined 
        os.system('mkdir -p ' + self.log_loc)
        # os.system('rm ' + self.log_loc + '/* > /dev/null 2>&1')

    def run_kf_on_pcol( self, delta, N_init, N, pcol, o_type, update_type, kf_type, no_idle=True):
        
        lg = self.lg
        lg.power_col = pcol
        lg.output_type = o_type # 'indiv'
        lg.full_principals = False # ignore cp #PXXX: Override here? 

        r = pcol
        #PXXX: Not sure about idle correction here. What if its in 'full' mode?
        if no_idle:
            if pcol == 'x_rest':
                lg.power_df[r] = lg.power_df[r] - lg.Widle_xrest
            elif pcol == 'perf_rapl':
                lg.power_df[r] = lg.power_df[r] - lg.Widle_cpu
            else:
                lg.power_df[r] = lg.power_df[r] - lg.Widle

        kf = Kalman_Filter()

        kf.update_type = update_type  # 'kalman'
        kf.kf_type = kf_type
        kf.ldf = lg 
        kf.ks_hist = []
        kf.update_p = False

        kf.kalman_over_time( N_init, N, delta, max_steps=np.inf )

        if no_idle:
            if pcol == 'x_rest':
                lg.power_df[r] = lg.power_df[r] + lg.Widle_xrest
            elif pcol == 'perf_rapl':
                lg.power_df[r] = lg.power_df[r] + lg.Widle_cpu
            else:
                lg.power_df[r] = lg.power_df[r] + lg.Widle

        self.kf = kf
    
    def drop_columns(self, df):
        if self.lg.output_type == 'indiv':
            cols = ['cp']
            return df.drop(cols , axis=1)
        return df
    
    def process(self):
        self.process_for_bars()
        self.process_for_stack()

    def process_for_bars(self):
        self._process( no_idle=True )

    def process_for_stack(self):
        self._process( no_idle=False )

    def _process(self, no_idle=True):
        self.run_kf_on_pcol(
                N_init = self.N_init,
                N = self.N,
                delta = self.delta,
                o_type = self.o_type,
                update_type = self.update_type,
                pcol = self.pcol, 
                kf_type = self.kf_type,
                no_idle = no_idle
        )

        kf = self.kf

        kf_times = kf.ks_over_time( 't' , True )
        kf_times = translate_t_to_mins( kf_times, tag_t=0 )

        # jdf = kf.ks_over_time( 'W_total' , True )
        # jdf = kf.ks_over_time( 'J_contrib' , True ) 

        jdf = kf.ks_over_time( 'J' , True )

        #   jdf = kf.ks_over_time( 'J_contrib' , True )
        #   dd-0-0.0.1                  97.167421
        #   image_processing-0-0.0.1    10.815404
        #   model_training-0-0.0.1      41.446552
        #   pyaes-0-0.0.1               57.687788

        #   jdf = kf.ks_over_time( 'x' , True )
        #   dd-0-0.0.1                  62.958949
        #   image_processing-0-0.0.1    56.971496
        #   model_training-0-0.0.1      81.849544
        #   pyaes-0-0.0.1                5.995032
        
        jdf.columns = self.lg.princip_list
        def set_index_to_kftimes( df, kf_times ):
            df[tag_tm] = kf_times
            return df.set_index(tag_tm)
        jdf = set_index_to_kftimes( jdf, kf_times )
        jdf = self.drop_columns( jdf )

        self.Cnet = kf.ks_over_time( 'Cnet' , True ) # time contributions
        self.Cnet.columns = self.lg.princip_list
        self.Cnet = self.drop_columns( self.Cnet )
        self.Cmean = self.Cnet / self.N
        self.Cmean = set_index_to_kftimes( self.Cmean, kf_times )

        self.Anet = kf.ks_over_time( 'Apersecnet' , True ) # invocations / activation
        self.Anet.columns = self.lg.princip_list
        self.Anet = self.drop_columns( self.Anet )
        self.Amean = self.Anet / self.N
        self.Amean = set_index_to_kftimes( self.Amean, kf_times )

        #print(jdf)
        #exit(-1)
        
        if no_idle:
            self.combined_share_mins = jdf
        else:
            self.combined_share_stacked_mins = jdf

    def save_dfs(self):
        save_df( self.combined_share_mins, self.log_loc + '/' + self.pcol +'_mins' )
        save_df( self.combined_share_stacked_mins, self.log_loc + '/stacked_' + self.pcol +'_mins' )
        save_df( self.Cnet, self.log_loc + '/Cnet' +'_mins' )
        save_df( self.Cmean, self.log_loc + '/Cmean' +'_mins' )
        save_df( self.Anet, self.log_loc + '/Anet' +'_mins' )
        save_df( self.Amean, self.log_loc + '/Amean' +'_mins' )

