import os

import subprocess
from subprocess import CalledProcessError

from tags import *
from config import *
from helper_funcs import *

class PostProcessing:

    def __init__(self, log_loc, N_init, N, delta, specific_dfs):
        self.N_init = N_init
        self.N = N
        self.delta = delta
        self.log_loc = log_loc
        self.analysis = {}
        self.specific_dfs = specific_dfs

        os.system('mkdir -p ' + self.log_loc)
        # os.system('rm ' + self.log_loc + '/* > /dev/null 2>&1')
    
    def dir_to_dfs_tags(self, s ):
        s = s.split('/')
        tags = s[-3:]
        return tags

    def read_specific_dfs(self, dirs, dfs):
        """
        reads the dfs in dirs list 
            and update the current dfs  
        """
        def _check_tag( tags, dfs ):
            if tags is None or len(tags) == 0:
                return
            if tags[0] not in dfs:
                dfs[tags[0]] = {}
                _check_tag( tags[1:], dfs[tags[0]] ) 
        for d in dirs:
            tags = self.dir_to_dfs_tags( d )
            if tags[0] == 'dfs':
                _check_tag( tags[1:], dfs ) 
                dfs[tags[1]][tags[2]] = load_pickle( d + '.pickle' ) 
                print("Specific df loaded: dfs[{}][{}] from {}".format( tags[1], tags[2], d ))
        return dfs


    def read_dfs(self, log_base):
        def read_dfs_dir( d ):
            try:
                fs = subprocess.check_output('ls '+d+'/*.pickle 2> /dev/null', shell=True)
                fs = "".join(map(chr, fs))
                fs = fs.split('\n')
                fs = fs[:-1]
                dfs = {}
                for f in fs:
                    bname = os.path.basename(f)
                    bname = bname.split('.')[0]
                    dfs[bname] = load_pickle( f )
                return dfs
            except CalledProcessError:
                pass
            return None
        base = read_dfs_dir( log_base + dir_lg2df )
        cpu = read_dfs_dir( log_base + dir_cpushare )
        cmb = read_dfs_dir( log_base + dir_combined )
        analysis = read_dfs_dir( log_base + dir_analysis )
        
        dfs = {}
        dfs[tag_lg2df] = base
        dfs[tag_dis_cpu] = cpu
        dfs[tag_dis_combined] = cmb
        dfs[tag_analysis] = analysis
        return dfs

    def save_analysis(self, key ):
        if key in self.analysis:
            print("Saving analysis: {}".format(self.log_loc + '/' + key))
            save_df( self.analysis[key], self.log_loc + '/' + key )

    def dump_all_df_keys(self):
        dump_keys( self.dfs )

    def save_all_analysis(self):
        for k,v in self.analysis.items():
            self.save_analysis(k)
