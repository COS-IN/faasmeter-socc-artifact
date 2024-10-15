import os

import subprocess
from subprocess import CalledProcessError
import pickle
import re

import numpy as np 
import pandas as pd

from tags import *
from config import *
from helper_funcs import *

from postprocessing.distance import L2norm, cosin

from postprocessing.base import PostProcessing

class PostProcessMultiExp(PostProcessing):
    
    def __init__(self, log_bases, N_init, N, delta, specific_dfs ):
        super().__init__( log_bases[0] + dir_analysis, N_init, N, delta, specific_dfs )
        self.log_bases = log_bases 
        self.get_mcx_bases()
        self.read_dfs()
        self.get_mca_bases()
        self.log_loc = self.base_mca + dir_analysis

    def read_dfs(self):
        dfs = {}
        for base in self.log_bases:
            dfs[base] = super().read_dfs( base )
            dfs[base] = self.read_specific_dfs( self.specific_dfs, dfs[base] )
        self.dfs = dfs

        dfs = {}
        for base in self.log_mcxbases:
            dfs[base] = super().read_dfs( base )
            dfs[base] = self.read_specific_dfs( self.specific_dfs, dfs[base] )
        self.mcxdfs = dfs


    def get_mca_bases( self ):
        dfs = self.dfs
        total_funcs = []
        for b in self.log_bases:
            try:
                tinvks = dfs[b][tag_analysis]['total_invokes']
            except TypeError:
                print("Error: Some key has None object in it")
                print("    b {}: {}".format(b, dfs[b]))
                if dfs[b] is not None:
                    print("    tag_analysis {}: {}".format(tag_analysis, dfs[b][tag_analysis]))
                    if dfs[b][tag_analysis] is not None:
                        print("    total_invokes {}: {}".format("total_invokes", dfs[b][tag_analysis]["total_invokes"]))
                exit(-1)
            total_funcs.append(len(tinvks.columns))
        tm = max(total_funcs)
        itm = total_funcs.index(tm)
        bs = set(self.log_bases)
        self.base_mca = self.log_bases[itm]
        self.bases_other = list(bs - set([self.base_mca]))
        print( "MCA is set to: {}".format(self.base_mca) )
        
        bmca = self.base_mca
        if tag_analysis in self.dfs[bmca]:
            self.analysis = self.dfs[bmca][tag_analysis]


    def get_mcx_bases( self ):
        def _is_mcx( string ):
            pat = 'mc_._.*'
            r = re.compile(pat)
            if re.match(r, string):
                return True
            return False
        
        lmcxs = []
        lbases = []

        for base in self.log_bases:
            if _is_mcx( base ):
                lmcxs.append( base )
            else:
                lbases.append( base )
        
        self.log_bases = lbases
        self.log_mcxbases = lmcxs 

    def get_mc( self ):
        def _get_mc( tag ):
            mca_df = self.dfs[self.base_mca]
            mca_te = mca_df[tag_analysis]['total_energy_'+tag]
            mca_total_invokes = mca_df[tag_analysis]['total_invokes'] 
            mca_funcs = set(mca_total_invokes.columns) 
            delta_tes = []
            funcs = []
            for b in self.bases_other:
                df = self.dfs[b]
                mco_te = df[tag_analysis]['total_energy_'+tag]
                delta_te = mca_te - mco_te
                delta_tes.append( delta_te.to_numpy()[0][0] )
                total_invokes = df[tag_analysis]['total_invokes'] 
                bfuncs = set(total_invokes.columns)
                bfunc = mca_funcs - bfuncs
                funcs.append( list(bfunc)[0] )
            delta_tes = pd.DataFrame([delta_tes], columns=funcs)
            mc = delta_tes / mca_total_invokes
            return mc

        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            mc_cpu = _get_mc(tag_pcpu)
            mc_shared = _get_mc(tag_pshared)
        mc_sys = _get_mc(tag_psys[0])
        
        analysis = self.analysis 
        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            analysis['mc_cpu'] = mc_cpu
            analysis['mc_shared'] = mc_shared
        analysis['mc_sys'] = mc_sys

    def get_mcx( self ):
        def _is_mcx_0_0( string ):
            pat = 'mc_._0_0.*'
            r = re.compile(pat)
            if re.match(r, string):
                return True
            return False
        
        no_func_base = None
        for xbase in self.log_mcxbases:
            if _is_mcx_0_0( xbase ):
                no_func_base = xbase 
                break

        # fetch the function we are processing for 
        mca_df = self.dfs[self.base_mca]
        mca_total_invokes = mca_df[tag_analysis]['total_invokes'] 
        mca_funcs = set(mca_total_invokes.columns) 
        
        if no_func_base is not None:
            df = self.mcxdfs[no_func_base]
            total_invokes = df[tag_analysis]['total_invokes'] 
            bfuncs = set(total_invokes.columns)
            bfunc = mca_funcs - bfuncs
            func_to_analyze = list(bfunc)[0] 
        else:
            return

        def _get_mcx( tag ):
            mca_df = self.dfs[self.base_mca]
            mca_te = mca_df[tag_analysis]['total_energy_'+tag]
            mca_total_invokes = mca_df[tag_analysis]['total_invokes'] 
            mca_funcs = set(mca_total_invokes.columns) 
            delta_tes = []
            delta_invokes = []
            e_per_invks = []

            col_names = []

            for xb in self.log_mcxbases:
                col_names.append( xb.split('/')[0] )
                df = self.mcxdfs[xb]
                mco_te = df[tag_analysis]['total_energy_'+tag]
                delta_te = mca_te - mco_te
                delta_tes.append( delta_te.to_numpy()[0][0] )

                total_invokes = df[tag_analysis]['total_invokes'] 
                if func_to_analyze in total_invokes.columns:
                    delta_invokes.append( int(mca_total_invokes[func_to_analyze] - total_invokes[func_to_analyze]) )
                    if delta_invokes[-1] == 0:
                        e_per_invks.append( 0 )
                    else:
                        e_per_invks.append( delta_tes[-1] / delta_invokes[-1] )
                else:
                    delta_invokes.append( int(mca_total_invokes[func_to_analyze]) )
                    e_per_invks.append( delta_tes[-1] / delta_invokes[-1] )
            delta_tes = pd.DataFrame([delta_tes], columns=col_names)
            delta_invokes = pd.DataFrame([delta_invokes], columns=col_names)
            e_per_invks = pd.DataFrame([e_per_invks], columns=col_names)
            return delta_tes, delta_invokes, e_per_invks 

        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            mcx_cpu = _get_mcx(tag_pcpu)
            mcx_shared = _get_mcx(tag_pshared)
        mcx_sys = _get_mcx(tag_psys[0])

        analysis = self.analysis 
        def _save_to_analysis( mcx, tag ):
            delta_te, delta_invks, delta_ep = mcx
            analysis['mcx_'+tag+'_delta_total_eng'] = delta_te 
            analysis['mcx_'+tag+'_delta_total_invks'] = delta_invks 
            analysis['mcx_'+tag+'_delta_e_per_invks'] = delta_ep 
        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            _save_to_analysis( mcx_cpu, 'cpu' )
            _save_to_analysis( mcx_shared, 'shared' )
        _save_to_analysis( mcx_sys, 'sys' )
        analysis['mcx_funcs_analyzed'] = pd.DataFrame([[func_to_analyze]])

    def get_errors_mean( self ):
        d = self.base_mca
        dfs = self.dfs[d]
        analysis = self.analysis
        
        def _errors_mean( tag_mc, j_per_invk ):
            print("-----------------------------")
            mc_truth = analysis[tag_mc]
            j_per_invk = j_per_invk.mean()
            errors = mc_truth.iloc[0] - j_per_invk
            errors = errors / mc_truth.iloc[0]
            errors = np.abs(errors)
            errors = errors * 100.0
            errors_n = errors
            errors = sorted(errors.values.reshape((1,-1))[0])
            errors = [ e for e in errors if not np.isnan(e) ]
            errors = pd.DataFrame(errors)
            return errors, errors_n
        
        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu']
            analysis['errors_mean_cpu'], analysis['errors_mean_cpu_n'] = _errors_mean( 'mc_cpu', j_per_invk )
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_shared']
            analysis['errors_mean_shared'], analysis['errors_mean_shared_n'] = _errors_mean( 'mc_shared', j_per_invk )
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu'] + dfs[tag_analysis]['j_per_invk_shared']
            analysis['errors_mean_sys_f23'], analysis['errors_mean_sys_f23_n'] = _errors_mean( 'mc_sys', j_per_invk )
       
        j_per_invk = dfs[tag_analysis]['j_per_invk_full']
        analysis['errors_mean_sys_full'], analysis['errors_mean_sys_full_n'] = _errors_mean( 'mc_sys', j_per_invk )

    def get_errors_distance( self ):
        d = self.base_mca
        dfs = self.dfs[d]
        analysis = self.analysis
        
        def _errors_distance( tag_mc, j_per_invk ):
            mc_truth = analysis[tag_mc]
            j_per_invk = j_per_invk.mean()
            mct = mc_truth.iloc[0]
            jpt = j_per_invk
            rc =  cosin( mct, jpt )
            rn =  L2norm( mct, jpt ) * 100.0

            rc = pd.DataFrame( [[rc]] )
            rn = pd.DataFrame( [[rn]] )
            return rc, rn

        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu']
            ec, en = _errors_distance( 'mc_cpu', j_per_invk )
            analysis['errors_distance_cpu_cosine'] = ec  
            analysis['errors_distance_cpu_norm'] = en  
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_shared']
            ec, en = _errors_distance( 'mc_shared', j_per_invk )
            analysis['errors_distance_shared_cosine'] = ec  
            analysis['errors_distance_shared_norm'] = en  
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu'] + dfs[tag_analysis]['j_per_invk_shared']
            ec, en = _errors_distance( 'mc_sys', j_per_invk )
            analysis['errors_distance_sys_cosine'] = ec  
            analysis['errors_distance_sys_norm'] = en  

        if 'scaphandre_e_per_invk' in analysis: 
            ec, en = _errors_distance( 'mc_sys', analysis['scaphandre_e_per_invk'] )
            analysis['errors_distance_sca_cosine'] = ec  
            analysis['errors_distance_sca_norm'] = en  
       
        j_per_invk = dfs[tag_analysis]['j_per_invk_full']
        ec, en = _errors_distance( 'mc_sys', j_per_invk )
        analysis['errors_distance_sys_full_cosine'] = ec  
        analysis['errors_distance_sys_full_norm'] = en  

    def get_errors_all( self ):
        d = self.base_mca
        dfs = self.dfs[d]
        analysis = self.analysis
        
        def _errors_all( tag_mc, j_per_invk ):
            mc_truth = analysis[tag_mc]
            errors = mc_truth.iloc[0] - j_per_invk
            errors = errors / mc_truth.iloc[0]
            errors = np.abs(errors)
            errors = errors * 100.0
            errors = sorted(errors.values.reshape((1,-1))[0])
            errors = [ e for e in errors if not np.isnan(e) ]
            errors = pd.DataFrame(errors)
            return errors
        
        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu']
            analysis['errors_all_cpu'] = _errors_all( 'mc_cpu', j_per_invk )
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_shared']
            analysis['errors_all_shared'] = _errors_all( 'mc_shared', j_per_invk )
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu'] + dfs[tag_analysis]['j_per_invk_shared']
            analysis['errors_all_sys_f23'] = _errors_all( 'mc_sys', j_per_invk )
       
        j_per_invk = dfs[tag_analysis]['j_per_invk_full']
        analysis['errors_all_sys_full'] = _errors_all( 'mc_sys', j_per_invk )

    def get_errors_per_func( self ):
        d = self.base_mca
        dfs = self.dfs[d]
        analysis = self.analysis
        
        def _errors_per_func( tag_mc, j_per_invk ):
            mc_truth = analysis[tag_mc]
           
            errors = mc_truth.iloc[0] - j_per_invk
            errors = errors / mc_truth.iloc[0]
            errors = np.abs(errors)
            errors = errors * 100.0
            errors = errors.fillna(1000)

            return errors
        
        if self.dfs[self.base_mca][tag_dis_cpu] is not None:
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu']
            analysis['errors_per_func_cpu'] = _errors_per_func( 'mc_cpu', j_per_invk )
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_shared']
            analysis['errors_per_func_shared'] = _errors_per_func( 'mc_shared', j_per_invk )
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu'] + dfs[tag_analysis]['j_per_invk_shared']
            analysis['errors_per_func_sys_f23'] = _errors_per_func( 'mc_sys', j_per_invk )
       
        j_per_invk = dfs[tag_analysis]['j_per_invk_full']
        analysis['errors_per_func_sys_full'] = _errors_per_func( 'mc_sys', j_per_invk )

    def gen_all_analysis(self):
        self.get_mc()
        self.get_mcx()
        self.get_errors_all()
        self.get_errors_mean()
        self.get_errors_distance()
        self.get_errors_per_func()

