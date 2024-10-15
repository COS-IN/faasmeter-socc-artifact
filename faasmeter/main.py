from tags import *
from config import *
from parsing.Logs_to_df import Logs_to_df
from disaggregation.cpu import CPU
from disaggregation.combined import Combined 
from postprocessing.single_exp import PostProcessSingleExp
from postprocessing.multiple_exps import PostProcessMultiExp
from plotting.stacked import StackPlot
from plotting.plots import Plots
from collector import Collector

def single_dir_processing( d, mc_type, output_type, specific_dfs, no_parsing=False ):

        if not no_parsing:
            # Parsing
            lg = Logs_to_df( d, mc_type, output_type )
            lg.process_all_logs()
            lg.save_dfs()
            
            # Disaggregation
            #P-Question: What if the CPU power is not present ala Jetson? Default should be system-wide coarse-grained? 
            if mc_type not in dissag_cpu_execptions:
                disagg_cpu = CPU(lg, N_init, N, delta, cpu_threshold, 'perf_rapl')
                disagg_cpu.process()
                disagg_cpu.save_dfs()
            
            def process_cmb( pcol ):
                disagg_cmb = Combined(lg, N_init, N, delta, o_type, update_type, pcol, kf_type)
                disagg_cmb.process()
                disagg_cmb.save_dfs()

            if mc_type not in dissag_cpu_execptions:
                process_cmb( 'perf_rapl' )
                process_cmb( 'x_rest' )

            process_cmb( tag_psys[0] )
        
        # PostProcessing
        ppsingle = PostProcessSingleExp( d, N_init, N, delta, specific_dfs )
        ppsingle.gen_all_analysis()
        ppsingle.save_all_analysis() 


if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="FaasMeter" )
    argparser.add_argument( "--lg_log_dir", '-l', help="Log Directory for single experiment analysis", required=False, type=str)                
    argparser.add_argument( "--lg_log_dirs", '-s', help="Log Directories for analysis over multiple exps", required=False, type=str, nargs='+')                
    argparser.add_argument( "--lg_specific_dfs", '-f', help="list of specific dfs to read", required=False, default='', type=str, nargs='+')                
    argparser.add_argument( "--quirks", '-q', help="list of specific quirks to apply", choices=list(quirks.keys()), required=False, default='', type=str, nargs='+')                
    argparser.add_argument( "--quirks_data", '-d', help="list of data for the given quirks", required=False, default='', type=str, nargs='+')                
    argparser.add_argument( "--collection_only", help="Don't process but collect stuff to mc_a dir", required=False, default=False, action='store_true')                
    argparser.add_argument( "--plots_only", '-p', help="Don't process but plot stuff", required=False, default=False, action='store_true')                
    argparser.add_argument( "--no_singular", '-n', help="Don't perform singular analysis", required=False, default=False, action='store_true')                
    argparser.add_argument( "--no_parsing", help="Don't perform parsing", required=False, default=False, action='store_true')                
    argparser.add_argument( "--mc_type", '-m', help="Platform Type", required=True, type=str, default='desktop')                

    args = argparser.parse_args()

    set_global_mc_type( args.mc_type )
    print( "tag_psys is {}".format(tag_psys) )

    for i,q in enumerate(args.quirks):
        quirks[q] = True
        if i < len(args.quirks_data):
            quirks_data[q] = args.quirks_data[i]

    if args.plots_only:
        if args.lg_log_dirs is not None:
            plots = Plots( args.lg_log_dirs, args.lg_specific_dfs )
        elif args.lg_log_dir is not None:
            plots = Plots( [args.lg_log_dir], args.lg_specific_dfs )
        plots.plot_everything()
        exit(0)

    if args.collection_only:
        collector = Collector( args.lg_log_dirs )
        collector.process()
        exit(0)

    if args.lg_log_dirs is not None:
        if not args.no_singular:
            for d in args.lg_log_dirs:
                single_dir_processing( d, args.mc_type, o_type, args.lg_specific_dfs, args.no_parsing )
        ppmulti = PostProcessMultiExp( args.lg_log_dirs, N_init, N, delta, args.lg_specific_dfs )
        ppmulti.gen_all_analysis()
        ppmulti.save_all_analysis() 
    elif args.lg_log_dir is not None:
        single_dir_processing( args.lg_log_dir, args.mc_type, o_type, args.lg_specific_dfs, args.no_parsing )
    exit(0)

