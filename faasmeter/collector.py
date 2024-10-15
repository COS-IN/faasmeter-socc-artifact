import os 

from helper_funcs import *

from postprocessing.multiple_exps import PostProcessMultiExp

class Collector: 

    def __init__( self, dirs ):
        self.dirs = dirs 
        
    def identify_dirs( self ):
        dirs = self.dirs 

        ppmulti = PostProcessMultiExp( dirs, 100, 200, 1, [] )
        self.dfs = ppmulti.dfs

        # mca - golden host dir  
        self.base_mca = ppmulti.base_mca

        # mca_others 
        self.bases_other = ppmulti.bases_other

        # mcx 
        self.bases_mcx = ppmulti.log_mcxbases
        
        # scaphandra 
        bases_scaphandre = []
        for d in dirs:
            blocks = d.split('/')
            for b in blocks:
                if b == 'scaphandre':
                    bases_scaphandre.append( d )
        
        self.bases_scaphandre = bases_scaphandre
   
    def copy_scaphandra_specific( self ): 
        mca = self.base_mca
        sca = self.bases_scaphandre[0] # currently assuming there is only one sca dir 

        specific_srcs = [
            'dfs/scaphandre_*',
            'dfs/analysis/scaphandre_*',
        ]
        destinations = [
            'dfs/',
            'dfs/analysis/',
        ]
        
        print("For {} to {}".format(sca, mca))
        for ss,dst in zip(specific_srcs, destinations):
            r = exec_cmd( 'cp '+ sca + '/' + ss + ' '+ mca + '/' + dst)
            print( "    copied {} to {}".format( ss, dst) )

    def copy_everything( self ):
        self.copy_scaphandra_specific()

    def process( self ):
        self.identify_dirs()
        self.copy_everything()

