import os

import subprocess
from subprocess import CalledProcessError
import pickle

import numpy as np 
import pandas as pd

from tags import *
from config import *
from helper_funcs import *
from postprocessing.base import PostProcessing

class PostProcessSingleExp(PostProcessing):
    
    def __init__(self, log_base, N_init, N, delta, specific_dfs):
        super().__init__( log_base + dir_analysis, N_init, N, delta, specific_dfs )
        self.log_base = log_base 
        self.read_dfs()

    def read_dfs(self):
        dfs = super().read_dfs( self.log_base )
        dfs = self.read_specific_dfs( self.specific_dfs, dfs )
        self.dfs = dfs

    def get_execution_times( self ):
        wdf = self.dfs[tag_lg2df]['worker_df']
        wdf = wdf.copy()
        
        wdf[tag_fn_exec] = wdf[tag_fn_e] - wdf[tag_fn_s]
        wdf[tag_fn_exec] = wdf[tag_fn_exec].astype('int64')/10**9
         
        self.analysis[tag_fn_exec] = wdf.groupby(tag_fqdn)[[tag_fn_exec]].agg([np.mean,np.std])
        return self

    def get_all_power_mins(self):
        pdf = self.dfs[tag_lg2df]['power_df']
        wdf = self.dfs[tag_lg2df]['worker_df']
        N_init = self.N_init
        
        start_t = wdf.iloc[0][tag_fn_s]
        end_t = wdf.iloc[-1][tag_fn_e]
        
        subset = pdf[pdf.index >= start_t+ pd.Timedelta(seconds=N_init)]
        # subset = pdf[pdf.index >= start_t]
        subset = subset[subset.index <= end_t].copy()
        times = np.array(list(map(lambda x: (x.hour*60) + x.minute, subset.index)))
        subset["minute"] = times
        subset = subset.groupby("minute").agg(np.mean)
        self.analysis['all_power'] = subset[subset.columns.intersection(tags_psys)]
        return self

    def get_p_per_ink_mins(self):
        if self.dfs[tag_dis_cpu] is not None:
            cpu_share = self.dfs[tag_dis_cpu][tags_pmins[tag_pcpu]]
            shared_share = self.dfs[tag_dis_combined][tags_pmins[tag_pshared]]
        full_share = self.dfs[tag_dis_combined][tags_pmins[tag_psys[0]]]
        ameans = self.dfs[tag_dis_combined]['Amean_mins']

        def gen_p_per_ink( npower, ameans ):
            r = npower / ameans 
            r.replace([np.inf, -np.inf], np.nan, inplace=True)
            return r

        if self.dfs[tag_dis_cpu] is not None:
            p_per_ink_cpu = gen_p_per_ink( cpu_share, ameans )
            p_per_ink_shared = gen_p_per_ink( shared_share, ameans )
        p_per_ink_full = gen_p_per_ink( full_share, ameans )

        if self.dfs[tag_dis_cpu] is not None:
            self.analysis['p_per_ink_cpu'] = p_per_ink_cpu 
            self.analysis['p_per_ink_shared'] = p_per_ink_shared
        self.analysis['p_per_ink_full'] = p_per_ink_full

    def get_j_per_invk_mins(self):
        ets = self.analysis['exec_time']
        ets = ets[('exec_time','mean')].transpose()

        if self.dfs[tag_dis_cpu] is not None:
            self.analysis['j_per_invk_cpu'] = self.analysis['p_per_ink_cpu'] * ets
            self.analysis['j_per_invk_shared'] = self.analysis['p_per_ink_shared'] * ets
        self.analysis['j_per_invk_full'] = self.analysis['p_per_ink_full'] * ets

    def get_jpt_ratio_mins(self):
        ets = self.analysis['exec_time']

        ets_mean = ets[('exec_time','mean')].transpose()
        ets_std = ets[('exec_time','std')].transpose()
        

        if self.dfs[tag_dis_cpu] is not None:
            jpt_cpu = self.analysis['j_per_invk_cpu']
            jpt_shared = self.analysis['j_per_invk_shared']
            jpt_cmb = jpt_cpu + jpt_shared
            jpt_cmb_std = jpt_cmb.std()
            ratio_cmb = jpt_cmb_std / ets_std
            self.analysis['ratio_jpt_cmb'] = ratio_cmb

        jpt_full = self.analysis['j_per_invk_full']
        jpt_full_std = jpt_full.std()
        ratio_full = jpt_full_std / ets_std
        self.analysis['jpt_full_std'] = jpt_full_std
        self.analysis['jpt_full_std_normalized'] = jpt_full_std / jpt_full.mean()
        self.analysis['ratio_jpt_full'] = ratio_full
    
    def get_delta_sys_error(self):

        analysis = self.analysis
        dfs = self.dfs
        tag = tags_pmins[tag_psys[0]]

        if self.dfs[tag_dis_cpu] is not None:
            df_linear_cpu = dfs[tag_dis_cpu][tags_pmins[tag_pcpu]]
            tag = 'stacked_' + tags_pmins[tag_pshared]
            df_kf_shared = dfs[tag_dis_combined][tag]  
            df_full = df_linear_cpu + df_kf_shared
        else:
            df_full = dfs[tag_dis_combined]['stacked_'+tag]
        sys = analysis['all_power'][tag_psys[0]]
        df_full = df_full.sum(axis=1)
        error = (sys - df_full)*100.0/sys

        analysis['sys_full_error_mins'] = error  

    def get_total_energies(self):
        analysis = self.analysis
        dfs = self.dfs

        pdf = dfs[tag_lg2df]['power_df_secs']
        wdf = dfs[tag_lg2df]['worker_df']

        ts = time_to_seconds(wdf.iloc[0][tag_fn_s])
        te = time_to_seconds(wdf.iloc[-1][tag_fn_e])
        pdf = pdf[pdf.index>=ts]
        pdf = pdf[pdf.index<=te]

        def save_total_energy( pcol ):
            total = pdf[pcol].sum()
            self.analysis['total_energy_'+pcol] = pd.DataFrame([total])
        if self.dfs[tag_dis_cpu] is not None:
            save_total_energy( tag_pcpu )
            save_total_energy( tag_pshared )
        save_total_energy( tag_psys[0] )

    def get_total_invokes(self):
        analysis = self.analysis
        dfs = self.dfs

        wdf = dfs[tag_lg2df]['worker_df']
        funcs = set(wdf[tag_fqdn])

        invokes = []
        col = []
        for fn in funcs:
            wdf_g = get_givenpattern(wdf, tag_fqdn, fn)
            invokes.append( len(wdf_g) )
            col.append( fn )
        invokes = pd.DataFrame( [invokes], columns=col )
        analysis['total_invokes'] = invokes

    def get_scaphandre_overhead(self):
        analysis = self.analysis
        dfs = self.dfs
        dfs_lg2df = dfs['lg2df']

        if 'scaphandre_total_df' not in dfs_lg2df:
            print("Warning: no scaphandre df available")
            return 

        stotal = dfs_lg2df['scaphandre_total_df']
        ssca = dfs_lg2df['scaphandre_scaphandre_df'] 

        texp = stotal[tag_p].sum()
        tsca = ssca[tag_p].sum()

        overhead = (tsca * 100) / texp

        df = pd.DataFrame( [[overhead]] )

        analysis['scaphandre_overhead'] = df

    def get_e_per_invk_scaphandre(self):
        analysis = self.analysis
        dfs = self.dfs
        dfs_lg2df = dfs['lg2df']
        
        if 'scaphandre_funcs_df' not in dfs_lg2df:
            print("Warning: no scaphandre df available")
            return 

        sfuncs = dfs_lg2df['scaphandre_funcs_df']
        total_invokes = analysis['total_invokes'] 
        
        fqdns = set(sfuncs[tag_fqdn])
        
        func = []
        e_per_invk = []
        total_energy = []

        for f  in fqdns:
            sfunc = get_givenpattern( sfuncs, tag_fqdn, f )

            te = sfunc[tag_p].sum()
            t_invokes = int(total_invokes[f][0])

            ep = te / t_invokes

            e_per_invk.append( ep )
            total_energy.append( te )
            func.append( f )
        
        df_scaphandre_e_per_invk = pd.DataFrame( [e_per_invk], columns=func )
        df_scaphandre_total_energy = pd.DataFrame( [total_energy], columns=func )

        analysis['scaphandre_e_per_invk'] = df_scaphandre_e_per_invk
        analysis['scaphandre_total_energy'] = df_scaphandre_total_energy

    def gen_all_analysis(self):
        self.get_execution_times()
        self.get_all_power_mins()
        self.get_p_per_ink_mins()
        self.get_j_per_invk_mins()
        self.get_jpt_ratio_mins()
        self.get_total_energies()
        self.get_total_invokes()
        self.get_e_per_invk_scaphandre()
        self.get_scaphandre_overhead()
        self.get_delta_sys_error()

    
