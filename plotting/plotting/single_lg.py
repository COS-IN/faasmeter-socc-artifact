    ##################################################
    # CPU Share Plotting 

    def plot_hwpc_against_pdf_given_func( self, pat='fetch.*' ):
        lg = self
        t = 'timestamp'
        sw_est = lg.sw_combined
        sw_est = get_givenpattern( sw_est, 'target', pat )
        sw_est = sw_est.groupby(t).agg(sum)
        sw_est = sw_est.reset_index()
        sw_est = lg.timestamp_datetime_to_ts( sw_est )
        lg.plot_hwpc_against_pdf(hwpc_agg=sw_est)

    def plot_hwpc_against_pdf( self, hwpc_agg=None, hwpc_global_agg=None, pdf=None, pdf_col='rapl_pw', scatter=False, zoom=False, title='' ):

        if hwpc_agg is None:
            hwpc_agg = self.sw_core_aggr_time_scaled.reset_index()

        if hwpc_global_agg is None:
            hwpc_global_agg = self.sw_global_scaled_agg

        if pdf is None:
            pdf = self.sw_combined_t

        features = self.sw_config['cfeatures']
        features = features[:-1]
        n = len(features)+1

        fig,axs = plt.subplots( n, 1, figsize=(20,20), sharex=True )
        tname = 'timestamp'

        if scatter:
            axs[0].scatter( pdf[tname], pdf[pdf_col], linewidths=0.1, marker='x' )
        else:
            axs[0].plot( pdf[tname], pdf[pdf_col] )

        axs[0].set_ylabel(pdf_col)
        axs[0].set_title(title)

        for i in range(1,n):
            if scatter:
                axs[i].scatter( hwpc[tname], hwpc[features[i-1]], linewidths=0.1, marker='x')
            else:
                axs[i].plot( hwpc_agg[tname], hwpc_agg[features[i-1]], label='funcs')
                axs[i].plot( hwpc_global_agg[tname], hwpc_global_agg[features[i-1]], label='system')

            axs[i].set_ylabel(features[i-1])
            axs[i].legend()

        axs[-1].set_xlabel('Time (milliseconds)')
        # fig.legend()
        fig.subplots_adjust(hspace=0.5)
        if zoom:
            plt.xlim(left=100000,right=104000)

    def plot_actual_against_predicted( self, power_df=None, pdf_col='perf_rapl', estimate=None, estimate_all=None, scatter=False, zoom=False, title='' ):
        if estimate is None:
            estimate = self.sw_estimate

        if power_df is None:
            power_df = self.power_df

        features = self.sw_config['cfeatures']
        n = 1

        if estimate_all is None:
            estimate_all = self.sw_estimate_all

        fig,axs = plt.subplots( n, 1, figsize=(20,10), sharex=True )
        tname = 'timestamp'
        axs = [axs]

        if scatter:
            axs[0].scatter( estimate[tname], estimate[pdf_col], linewidths=0.1, marker='x', label='Actual' )
        else:
            if pdf_col == 'perf_rapl':
                x = power_df.index
            else:
                x = power_df[tname]
            axs[0].plot( x, power_df[pdf_col], label=pdf_col+'_Actual' )

        pdf_col_e = pdf_col+'_p_all'
        if scatter:
            axs[0].scatter( estimate_all[tname], estimate_all[pdf_col_e], linewidths=0.1, marker='x', label='Total Predicted' )
        else:
            axs[0].plot( estimate_all[tname], estimate_all[pdf_col_e], label=pdf_col_e+'_Total Predicted' )

        pdf_col_e = pdf_col+'_p'
        def plot_predicted( df ):
            if scatter:
                axs[0].scatter( df[tname], df[pdf_col_e], linewidths=0.1, marker='x' )
            else:
                axs[0].plot( df[tname], df[pdf_col_e] )

        # estimate_gzip = get_givenpattern( estimate, 'target', 'gzip.*' )
        estimate_gzip = get_givenpattern( estimate, 'target', 'video.*' )
        # estimate.groupby('target').apply( plot_predicted )
        estimate_gzip.groupby('target').apply( plot_predicted )

        axs[0].set_ylabel('Power (Watts)')
        axs[0].set_title(title)

        # axs[-1].set_xlabel('Time (milliseconds)')
        axs[-1].set_xlabel('Time (date - hrs:mins)')
        fig.legend()
        fig.subplots_adjust(hspace=0.5)
        if zoom:
            plt.xlim(left=estimate['timestamp'].iloc[1000],right=estimate['timestamp'].iloc[1500])
    
    def plot_power_df( self ):
        pcols = [
            'VDD_GPU_SOC_current',
            'VDD_CPU_CV_current',
            'VIN_SYS_5V0_current',
            'VDDQ_VDD2_1V8AO_current',
            'tegra',
            'igpm',
        ]

        lg = self

        t = 'timestamp'

        pd = lg.power_df

        n = 1
        fig, axs = plt.subplots( n,1, figsize=(9,5), sharex=True )

        x = pd[t]

        ax = axs
        for p in pcols:
            ax.plot( x, pd[p], label=p )

        ax.legend(bbox_to_anchor=(1.0, 1.02),  ncol=1)
        ax.set_xlabel('Time')
        plt.show()

    def plot_est_energy(self, target_patterns, sw_est=None, idle_power=None, p_col='perf_rapl' ):

        if sw_est is None:
            sw_est = self.sw_estimate

        if idle_power is None:
            idle_power = self.idle_power
        
        ## 
        # Activations
        
        wdf = self.worker_df
        
        fs = 'fn_start' 
        fe = 'fn_end' 
        fink = wdf.iloc[0][fs]
        eink = wdf.iloc[-1][fe]
        total_sec = int((eink-fink).total_seconds())
        t = (list(range(0,total_sec))) 
        t = [ pd.Timedelta(ts,unit='sec') + fink for ts in t ]
        t = np.array(t) 
        
        if 'cp' in self.princip_list:
            pl = self.princip_list[:-1]
        else:
            pl = self.princip_list
        # A = self.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='C' ) 
        A = self.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='A' ) 
       
        ##
        # Power Stuff
        
        pdf = self.power_df

        n = len(target_patterns)
        fig, axs = plt.subplots( n+1, 1, figsize=(20,20), sharex=True )

        tname = 'timestamp'
        pa = p_col + '_p_active'
        pa_g = p_col 
        idl = 'idle_share'
        sa = 'sys_active'
        sidl = 'sys_idle'

        def plot_on_ax( ax, sw_est_t ):
            ax.plot( sw_est_t[tname], sw_est_t[pa], label=pa)
            ax.plot( sw_est_t[tname], sw_est_t[idl], label=idl)
        
        for i in range(0,n):
            ax = axs[i]
            
            tp = target_patterns[i]
            ip = find_matching_string_index( pl, tp)
            
            sw_est_t = get_givenpattern( sw_est, 'target', tp)
            sw_est_t = sw_est_t.groupby(tname).agg(sum).reset_index()
            
            act = A[:,ip]
            ax.plot( t, act, label='C' )
            
            plot_on_ax( ax, sw_est_t )
            ax.plot( sw_est_t[tname], sw_est_t[sa], label=sa)
            ax.plot( sw_est_t[tname], sw_est_t[sidl], label=sidl)

            ax.set_ylabel('Power (Watts) - '+target_patterns[i])
            ax.legend()

        sw_est_cmb = sw_est.groupby(tname).agg(sum).reset_index()
        ax = axs[-1]
        plot_on_ax( ax, sw_est_cmb )
        ax.plot( pdf.index, pdf[pa_g]-idle_power, label=pa_g)
        ax.set_ylabel('Power (Watts) - Summation of all targets')
        ax.legend()

        axs[-1].set_xlabel('Time (milliseconds)')
        # fig.legend()
        fig.subplots_adjust(hspace=0.5)
    
    def plot_Contributions( self ):

        worker_df = self.worker_df

        fs = 'fn_start'
        fe = 'fn_end'

        wdf = worker_df
        fink = wdf.iloc[0][fs]
        eink = wdf.iloc[-1][fe]
        total_sec = int((eink-fink).total_seconds())

        ts = (list(range(0,total_sec)))
        ts = [ ((pd.Timedelta(tse,unit='sec') + fink)) for tse in ts ]
        ts = np.array(ts)

        if 'cp' in self.princip_list:
            pl = self.princip_list[:-1]
        else:
            pl = self.princip_list
        # display( pl )

        # A = self.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='C' )
        A = self.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='A' )
        A = pd.DataFrame(A)

        n = len(pl)
        fig, axs = plt.subplots( n,1, figsize=(9,5), sharex=True, sharey=True )

        for i,f in enumerate( pl ):
            ax = axs[i]
            ax.plot( ts, A[i], label=f )
            ax.legend()

        plt.ylabel('Contributions')
        plt.xlabel('Time')
        plt.show()

    def plot_activations( self ):
        wdf = self.worker_df
        
        fs = 'fn_start' 
        fe = 'fn_end' 
        fink = wdf.iloc[0][fs]
        eink = wdf.iloc[-1][fe]
        total_sec = int((eink-fink).total_seconds())
        
        if 'cp' in self.princip_list:
            pl = self.princip_list[:-1]
        else:
            pl = self.princip_list

        t = np.array(list(range(0,total_sec)))/60.0 
        
        # A = self.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='A' ) 
        A = self.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='C' ) 
        
        display( A )
        ## Plotting Exec Times
        fig, axs = plt.subplots( len(pl), 1, figsize=(20,10), sharex=True )
        
        for i,p in enumerate(pl):
            
            ax = axs[i]
            ax.plot( t, A[:,i], label=p )
            ax.legend()
        
        axs[-1].set_xlabel('Time (mins)')

    def get_power_from_sw_analysis( self, target_patterns, sw_estimate=None, col='perf_rapl' ):
        p_act = []
        p_idl = []
        for t in target_patterns:
            a,idle = self.get_power_given_target( t, sw_estimate=sw_estimate, col=col )
            p_act.append( a )
            p_idl.append( idle )
        return p_act, p_idl

    def get_energy_from_sw_analysis_power_method( self, target_patterns ):
        lg = self

        pa, pi = lg.get_power_from_sw_analysis( target_patterns )
        cmb = lg.get_execution_times()
        cmb = cmb.reset_index()

        ea = []
        ei = []
        for i,t in enumerate(target_patterns):
            rp = get_givenpattern( cmb, 'fqdn', t )
            temp = rp['exec_time']['mean']

            if len(temp) == 0:
                m = 0.0
            else:
                m = float(temp)

            ea.append( pa[i] * m )
            ei.append( pi[i] * m )

        print("Energy Active P-Method: {}".format( ea ) )
        print("Energy idle P-Method: {}".format( ei ) )
        return ea,ei
 
    def plot_power_bars( self, target_patterns, sw_estimate=None, col='perf_rapl' ):

        p_act, p_idl = self.get_power_from_sw_analysis( target_patterns, sw_estimate=sw_estimate, col=col )
        
        fig,ax = plt.subplots(1,1)
        plt.bar( target_patterns, np.array(p_act)+p_idl, label='Idle' )
        plt.bar( target_patterns, np.array(p_act), label='Active' )
        plt.xticks(rotation='vertical')
        plt.ylabel('Average Power (Watts)')
        plt.legend()

    def get_energies_from_sw_analysis( self, target_patterns, sw_estimate=None, col='perf_rapl' ):
        engs = []
        idls = []
        sys_a = []
        sys_idle = []
        for t in target_patterns:
            a,idle,s_a,s_idl = self.get_energy_given_target( t, sw_estimate=sw_estimate, col=col )
            engs.append( a )
            idls.append( idle )
            sys_a.append( s_a )
            sys_idle.append( s_idl )
        return engs, idls, sys_a, sys_idle

    def get_energies_from_sw_online_cpu_analysis( self, target_patterns, df=None, col='perf_rapl' ):
        engs = []
        for t in target_patterns:
            a = self.get_energy_given_target_sw_online_cpu( t, df=df, col=col )
            engs.append( a )
        return np.array(engs)

    def plot_bars_from_sw_analysis( self, target_patterns, sw_estimate=None, col='perf_rapl' ):

        engs, idls, sys_a, sys_idle = self.get_energies_from_sw_analysis( target_patterns, sw_estimate=sw_estimate, col=col )

        fig,ax = plt.subplots(1,1)
        plt.bar( target_patterns, np.array(engs)+sys_a+sys_idle+idls, label='idle' )
        plt.bar( target_patterns, np.array(engs)+sys_a+sys_idle, label='sys_idle' )
        plt.bar( target_patterns, np.array(engs)+sys_a, label='sys_active' )
        plt.bar( target_patterns, engs, label='active' )
        plt.xticks(rotation='vertical')
        plt.ylabel('Energy per invocation (Joules)')
        plt.legend()

        print( "Sum of active, os active, os idle, idle" )
        print( np.array(engs)+sys_a+sys_idle+idls )
        print( np.array(engs)+sys_a+sys_idle )
        print( np.array(engs)+sys_a )
        print( np.array(engs) )

    def plot_sys_power_rest_cpu( self, sw_estimate=None ):
        if sw_estimate is None:
            sw_estimate = self.sw_estimate

        cpu = 'perf_rapl'
        rest = 'x_rest'
        t = 'timestamp'

        self.append_x_rest(sw_estimate, col_cpu=cpu)

        fig,axs = plt.subplots( 1, 1, figsize=(20,10) )

        ax = axs
        ax.plot( sw_estimate[t], sw_estimate[self.col_sys], label='System' )
        ax.plot( sw_estimate[t], sw_estimate[cpu], label='CPU' )
        ax.plot( sw_estimate[t], sw_estimate[rest], label='REST' )

        fig.legend()

    ##################################################
    # General Plotting 
    def plot_execution_times( self ):

        et = 'exec_time' 

        cmb = self.get_execution_times()

        display( cmb )
        fig, ax = plt.subplots()

        ax.errorbar(cmb.index, list(cmb[et]['mean']), yerr=list(cmb[et]['std']) )
        plt.xticks(rotation='vertical')
        plt.ylabel('Time (seconds)')
        plt.show()

    ##################################################
    # scaphandre related Plotting 
    def plot_scaphandre_dfs( sdf, sca_df, t_df ):
        funcs_set = set( sdf['fqdn'] )
        tag_cns = 'consumption'
        import matplotlib.pyplot as plt
        for f in funcs_set:
            sub_sdf = sdf.loc[ sdf['fqdn'] == f ]
            # plt.scatter( sub_sdf.index, sub_sdf[tag_cns], label=f )
            plt.plot( sub_sdf.index, sub_sdf[tag_cns], label=f )
        plt.plot( sca_df.index, sca_df[tag_cns], label='scaphandre' )
        plt.plot( t_df.index, t_df[tag_cns], label='total_reported' )
        plt.legend()
        # plot_scaphandre_dfs( scaphandre_funcs_df, scaphandre_scaphandre_df, scaphandre_total_df ) 
