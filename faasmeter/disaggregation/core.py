import numpy as np 
import pandas as pd

import cvxpy as cp

class Core():
    #P-Suggest: Needs a better name... matrix_processing?? But also does post processing, alignment etc... 
    debug_timestamps = False
    min_fn_power = 1 # 1 Watt? Or depending on the CP overhead? 

    def __init__(self, lg):
        self.lg = lg

    ##################################################
    ########### MATRIX STUFF #########################

    def c_s_contribution(self, t, delta, kind="A"):
        """ Control plane and system contribution """ 
        if kind == "A":
            # ctrl plane and os are always running 
            return 1
        
        elif kind == "C":
            cplane_cpu_pct, _ = self.vdi(t, delta, self.lg.power_df, "cpu_pct_process")
            # This is the system-wide CPU %.  We need to convert it to time, which is delta seconds
            cp_time = cplane_cpu_pct * 0.01 *float(delta) * self.lg.cpu_cores
            # normalize this by system-cpu
            sys_cpu, _  = self.vdi(t, delta, self.lg.cpu_df, "cpu_pct")
            if self.lg.fix_sys_cpu:
                sys_cpu = max(sys_cpu, cplane_cpu_pct)*0.5 #atmost half 
            ### XXX above for the 0 sys-cpu case due to integer rounding. 
            if sys_cpu > 0:
                cp_time = cp_time/sys_cpu 
            
            sys_time = 0.0 # TODO: From OS sys/kern %.
            
            return cp_time

    ##################################################
    
    def get_A_row(self, t, delta):
        """ Get the functions active from t-delta to t """
        #Vector of size M, number of unique functions
        worker_df = self.lg.worker_df
        M = self.lg.M
        
        ts = t + pd.Timedelta(seconds=-delta)
        te = t
        # This catches all functions that intersect the interval atleast once.
        # Even if fn_start > ts or fn_end < te. DeMorgan  
        # If fn len is more than delta, then function will be in multiple rows. OK.
        # Negations everywhere because of pandas not supporting >=? 
        rfns = worker_df[~(worker_df['fn_end'] < ts) & ~(worker_df['fn_start'] > te)]

        row = np.zeros(M)

        for fn in rfns['fqdn']:
            i = self.lg.get_fn_index(fn)
            row[i]=row[i]+1 

        if self.lg.output_type != "full":
            # 2 extra columns have been added, we need to set them to some value. 
            c = self.c_s_contribution(t, delta, "A")
            row[M-1] = c 
            #row[M-1] = s 

        if self.debug_timestamps:
            print((t, row))
            
        return row 

    ##################################################
    ##################################################

    def vdi(self, t, delta, df, col):
        """ General function for avg, list of values in col during the interval (t-delta, t) in the df """ 
        ts = t + pd.Timedelta(seconds=0-delta)
        #match the rows 
        m = df.index[(df.index >= ts) & (df.index <= t)]
        sdf = df.loc[m]
        avg = np.mean(sdf[col])
        if np.isnan(avg):
            avg = 0.0 
        return avg, sdf[col]

    ##################################################
    
    def tspan_sec(self, ts, te):
        # TODO resolution stuff?
        td = te - ts
        nanosec = float(td.to_numpy())
        ns = 1000000000.0
        return float(nanosec/ns)

    ##################################################
    
    def interval_intersection(self, a, b):
        """ a and b are tuples with start and end times. All explicit timestamps. seconds """
        astart, aend = a
        bstart, bend = b 
        instart = max(astart, bstart)
        inend = min(aend, bend)
        duration_seconds = self.tspan_sec(instart, inend)
        return duration_seconds 
    
    ##################################################

    def resource_contrib_in_interval(self, int_span, fn_span, total_consumption):
        """ Return the resource consumption within the int_span time interval for the function with fn_span start and end times """
        # total_consumption obtained from the worker-log-df and is the recorded value on function completion
        fn_start, fn_end = fn_span 
        fn_len = self.tspan_sec(fn_start, fn_end) 
        usage_len = self.interval_intersection(int_span, fn_span) #in seconds

        return total_consumption*float(usage_len)/fn_len 

    ##################################################

    def get_C_row(self, t, delta):
        """ Contributions: Running time for each function in this interval (seconds) """
        worker_df = self.lg.worker_df
        M = self.lg.M
        
        ts = t + pd.Timedelta(seconds=-delta)
        te = t
        # These are all the functions that are running. What if there are multiple invoks?
        rfns = worker_df[~(worker_df['fn_end'] < ts) & ~(worker_df['fn_start'] > te)]
        common_interval = (ts, te)
        row = np.zeros(M)

        #need to iterate across all invocations/rows in rfns :
        for frow in rfns.itertuples():
            fn = frow.fqdn
            fn_interval = (frow.fn_start, frow.fn_end)
            runtime = self.interval_intersection(common_interval, fn_interval)
            i = self.lg.get_fn_index( fn )
            row[i] = row[i] + runtime

        if self.lg.output_type != "full": 
            # 2 extra columns have been added, we need to set them to some value. 
            c = self.c_s_contribution(t, delta, "C")
            row[M-1] = c 
            #row[M-1] = s

        if self.debug_timestamps:
            print((t, row))

        return row 

    ##################################################

    def single_R_row(self, t, delta, kind="network"):
        """ Assumes worker df will have additional columns for each resource (exec, network, disk, etc). """
        # interval intersection becomes tricky?
        # Maybe just report the fraction of time that was trimmed off and apply the correction here? 
        # basically get_C_row, with interval_intersection replaced by resource_contrib_in_interval
        # can be abstracted out? Can set kind="Exec_time" for the same effect..
        worker_df = self.lg.worker_df
        M = self.lg.M
        
        ts = t + pd.Timedelta(seconds=-delta)
        te = t
        # These are all the functions that are running. What if there are multiple invoks?
        rfns = worker_df[~(worker_df['fn_end'] < ts) & ~(worker_df['fn_start'] > te)]
        common_interval = (ts, te)
        row = np.zeros(M)

        #need to iterate across all invocations/rows in rfns :
        for frow in rfns.itertuples():
            fn = frow.fqdn
            fn_interval = (frow.fn_start, frow.fn_end)
            fdict = frow._asdict()
            total_consumption = fdict[kind] 
            runtime = self.resource_contrib_in_interval(common_interval, fn_interval, total_consumption)
            i = self.lg.get_fn_index( fn )
            row[i] = row[i] + runtime

        if self.lg.output_type != "full": 
            # 2 extra columns have been added, we need to set them to some value. 
            c = self.c_s_contribution(t, delta, "C")
            row[M-1] = c 
            #row[M-1] = s

        if self.debug_timestamps:
            print((t, row))

        return row

    ##################################################

    def get_R_row(self, t, delta):
        """ Vector of resources consumed by each function """ 
        out = []
        M = self.lg.M 
        dimensions = ['exec','network']
        for kind in dimensions:
            one_feat = self.single_R_row(t, delta, kind)
            out.append(one_feat)

        npo = np.array(out)
        npr = np.reshape(npo, (M, 1, 2)) #2 features 
        return npr

    ##################################################
    ##################################################
    
    def build_A_matrix(self, ts, N, delta, ac_type="C"):
        """ ACHTUNG: DO NOT USE FOR A. Double counting if fns cross rows """
        # get_A_row is still OK for accurate counts in an interval! 
        M = self.lg.M 
        A = np.zeros((N,M))
        AR = []
        
        for i in range(N):
            t = ts+pd.Timedelta(seconds=i) #need to check that it exists?
            if ac_type == "A":
                A[i] = self.get_A_row(t, delta)
            elif ac_type == "C":
                A[i] = self.get_C_row(t, delta)
            elif ac_type == "R":
                AR.append(self.get_R_row(t, delta))

        if ac_type == "R":
            A = np.concatenate(AR, axis=1) 
        A = np.nan_to_num(A, 0)
        return A

    ##################################################

    def J_at(self, t, delta, col, lag=0):
        # average energy from t-delta to t. 
        # If delta<=1 second, return W[t]? TODO, not done yet yolo

        #Since the power signal lags, we need to find the power at a future point in time and add 
        if lag > 0:
            t = t + pd.Timedelta(seconds = lag)

        ts = t + pd.Timedelta(seconds=0-delta)

        df = self.lg.power_df

        #match the rows 
        m = df.index[(df.index >= ts) & (df.index <= t)]
        sdf = df.loc[m]

        J_avg = np.mean(sdf[col])

        #possible for this to be nan if missing value.
        if np.isnan(J_avg):
            J_avg = -1.0 #filter out later?

        #Since we want energy, multiply by the delta seconds
        J_avg = J_avg * np.abs(delta)
        if self.debug_timestamps:
            print((ts, J_avg))
        #May want to compute variance of power, samples, confidence intervals etc later
        return J_avg, sdf[col]

    ##################################################

    def build_J_matrix(self, ts, N, delta):
        """ Energy for N time steps """ 
        W = [] 
        for i in range(N):
            t = ts + pd.Timedelta(seconds=i)
            J_avg, all_matched_entries = self.J_at(t, delta, self.lg.power_col, lag=0)
            W.append(J_avg)

        raw_W = np.array(W)
        # TODO: get the signal lag and input this here?
        W = raw_W
        
        if self.lg.output_type != "full":
            # remove idle
            # XXX maybe want to do idle correction elsewhere?
            # W = raw_W - (delta*self.lg.Widle)
            pass

        # XXX We should do some sanity checking and post-processing here,
        # Raw power output may be messed up/negative etc? 
        W[W < 0] = np.nan
        pjm = pd.Series(W)
        pjm = pjm.interpolate(method='nearest')
        W = np.array(pjm)
        W = np.nan_to_num(W, 0)
        return W 

    ##################################################

    def summarize_A_W(self, A, W):
        # Mainly for debugging and sanity-checking of post-processing and the experiment
        # Number of functions and total energy should be some sane number 
        net_activations = np.sum(A, axis=0)
        Wnet = np.sum(W) # Works if its per-second, otherwise needs averaging and time multiplication. 
        return (net_activations, Wnet)

    ##################################################

    def get_start_ts(self):
        ts = self.lg.worker_df['fn_start'][0]
        return ts

    def get_end_te(self):
        te = self.lg.worker_df.iloc[-1]['fn_end']
        return te

    #################################################
    
    def summarize_trace(self):
        # get all activations?
        ts = self.get_start_ts() 
        te = self.get_end_te() 
        trace_len = self.tspan_sec(ts, te)
        Arow = self.get_A_row(te, trace_len)
        return Arow 
        
    ##################################################

    def build_A_W(self, ts, N, delta):
        """ Main interface for getting the A,W matrices """
        W = self.build_J_matrix(ts, N, delta) 
        A = self.build_A_matrix(ts, N, delta, ac_type=self.lg.a_type)
        return (A, W)

    ##################################################
    #################### Output Processing ##########

    def invok_fracs(self, A):
        """ Characteristics of A """
        # col sums not needed if already summarized.
        if len(np.shape(A)) == 1:
            column_sums = A
        else:
            column_sums = np.sum(A, axis=0)
        nc = column_sums/np.sum(column_sums)
        return nc

    ##################################################

    def fn_latencies_in_interval(self, ts, te):
        fn_times = [[] for x in range(self.lg.M)] # list of lists        
        worker_df = self.lg.worker_df 
        rfns = worker_df[~(worker_df['fn_end'] < ts) & ~(worker_df['fn_start'] > te)]

        for row in rfns.itertuples():
            fn = row.fqdn
            fn_latency = self.tspan_sec(row.fn_start, row.fn_end)
            i = self.lg.get_fn_index(fn)
            fn_times[i].append(fn_latency)


        fn_times = np.array(fn_times) #array of lists
        nfuncs = np.shape(fn_times)[0]
        
#        fn_times = np.array([numpy.array(xi) for xi in fn_times])
        #np.mean(fn_times, axis=1) for average 
        return fn_times 

    ##################################################

    def avg_fn_latencies_interval(self, t, delta):
        #avgs = self.get_C_row(t, delta)/self.get_A_row(t, delta)
        avgs = self.div_by_A(self.get_C_row(t, delta), self.get_A_row(t, delta))
        return np.nan_to_num(avgs)

    ##################################################

    def fn_lat_variance_interval(self, t, delta):
        
        pass 

    ##################################################
    
    def fn_power_to_energ(self, X, ts, te):
        """ solving CX-W, X:per-fn power (not per-invok!). 
        Multiply by avg fn latency during the interval """
        
        delta = self.tspan_sec(ts, te)
        avg_fn_times = self.avg_fn_latencies_interval(te, delta)
        #fn_times = self.fn_latencies_in_interval(ts, te)
        #avg_fn_times = np.array([np.mean(x) for x in fn_times])
        #avg_fn_times = np.nan_to_num(avg_fn_times) 
        # Can contain nans, replace by 0 
        #avg_fn_times = np.mean(fn_times, axis=1)  #nested list, doesnt work 
        J  = X * avg_fn_times # because its avg_latency, this is per-invok 
        # average energ consumption for each function. If we want energ contrib, multiply by number of fn invocations.. 
        J_contrib = J * self.get_A_row(te, delta)
        Xcorr = np.array(X)
        
        if self.lg.full_principals:
            #The last element is the control-plane!
            cplane_cpu_pct, _ = self.vdi(te, delta, self.lg.power_df, "cpu_pct_process")
            cp_pow = X[-1] * float(cplane_cpu_pct) * 0.01

            sys_cpu, _  = self.vdi(te, delta, self.lg.cpu_df, "cpu_pct")
            if self.lg.fix_sys_cpu:
                sys_cpu = max(sys_cpu, cplane_cpu_pct)*0.5 #atmost half 

            if sys_cpu > 0:
                cp_pow = cp_pow * 100.0 / sys_cpu
                
            Xcorr[-1] = cp_pow
            J[-1] = cp_pow * delta
            J_contrib[-1] = cp_pow * delta
            
        return Xcorr, J, J_contrib, avg_fn_times  

    ##################################################
    
    # Given some X values, compute the other output types
    def indiv_to_marginals(self, X, ts, te):
        """ Input: Average X values for each function over (ts, ts+span seconds) """ 

        assert(self.lg.output_type == "indiv")
        
        delta = float(self.tspan_sec(ts, te))
        #delta = span 
        # control plane portion needs to be divided by the number of invoks?
        M = self.lg.M # == len(X) 

        #cp_pow = X[M-1] # cp are appended 
        #total_cp_overhead  = delta*np.mean(cp_pow) 
        
        cp_portion = []
        Xcorr, J, J_contrib, avg_fn_times = self.fn_power_to_energ(X, ts, te)
        cp_power = Xcorr[-1]
        cp_energ = cp_power * delta
        
        # invok fraction in this time.
        # Invocations of the function is just A ? # from ts+span to 
        A = np.array(self.get_A_row(te, delta)) # invok counts for each function
        # We already have control-plane ovhead, infact it should be 0!
        A[M-1] = 0 
        invok_frac_per_fn = self.invok_fracs(A)
        cp_portion = np.array(cp_energ*invok_frac_per_fn)
        
        total_idle_overhead = np.array([self.lg.Widle * delta for x in range(len(X))])
        active_principals = np.count_nonzero(A) # or even invok_frac? or some higher thresh? 
        
        idle_portion = total_idle_overhead/active_principals # Array of len N of this value 

        # per_invok=J_contrib+cplane/#invoks, which is J + ...  

        indiv_and_cp = np.array(J_contrib) + np.array(cp_portion)
        indiv_cp_idle = indiv_and_cp + idle_portion

        # should be proportional to A
        per_invok_cp_global = cp_energ/np.sum(A) 
        per_invok_cp = np.repeat(per_invok_cp_global, len(X)) #* A  
        #self.div_by_A(cp_portion, A) #zeroes?
        
        per_invok_idle = self.div_by_A(idle_portion, A) 
        
        return indiv_and_cp, indiv_cp_idle, per_invok_cp, per_invok_idle, cp_portion, A   


    ##################################################
    
    def pow_to_J_total(self, X, ts, te):
        """ Power to energy per invocation and components:
        J(per-invok) + J_cp/\sum(A) + J_idle/M.A_i """
        
        assert(self.lg.output_type == "indiv")        
        delta = float(self.tspan_sec(ts, te))
        M = self.lg.M # == len(X)
        Xcorr, J, J_contrib, avg_fn_times = self.fn_power_to_energ(X, ts, te)
        cp_power = Xcorr[-1] # This can go too high? 
        
        J_cp = cp_power * delta
        J_idle = self.lg.Widle * delta

        A = np.array(self.get_A_row(te, delta)) # invok counts for each function
        # We already have control-plane ovhead, infact it should be 0!
        if self.lg.full_principals:
            A[M-1] = 0
        Asum = np.sum(A)
        cp_share = np.repeat(J_cp/Asum, len(X)) # should ignore cp's own portion while analyzing.
        active_principals = np.count_nonzero(A)
        #can this be zero? nothing at all running.
        if active_principals == 0:
            idle_share = np.repeat(0, len(X))
        
        idle_share = self.div_by_A(np.repeat((J_idle/active_principals), len(X)), A) #makes little sense for per-invok
        return A, J, cp_share, idle_share

      
    ##################################################

    def J_contrib_total(self, X, ts, te):
        """ J_contrib = J*A, which is net energy use by function over a span of time. Power-top like use-cases. """ 
        A, J, cp_share, idle_share = self.pow_to_J_total(X, ts, te)
        return J*A, cp_share*A, idle_share*A 
        
    ##################################################

    def div_by_A(self, T, B):
        """ Aaargh numpy why! T/B no work! """
        T = list(T)
        B = list(B)
        out = []
        for i, b in enumerate(B):
            if b > 0:
                v = T[i]/b
            else:
                v = T[i]
            out.append(v)
            
        return np.array(out)

    ##################################################
    
    def pow_full_breakdown(self, X, ts=-1, te=-1):
        
        if ts == -1:
            ts = self.lg.worker_df['fn_start'][0]
        
        if te == -1:
            te = self.lg.worker_df.iloc[-1]['fn_end']

        outd = dict()
        Xcorr, J, J_contrib, avg_fn_times   = self.fn_power_to_energ(X, ts, te)
        
        outd["Power"], outd["Energy"], outd["Energy-contrib"], outd["Avg-fn-times"] = np.array(Xcorr), np.array(J), np.array(J_contrib), np.array(avg_fn_times)
        
        if self.lg.output_type == 'indiv':
            outd["A"], outd["Energy"], outd["per_invok_cp"], outd["per_invok_idle"]  = self.pow_to_J_total(X, ts, te)
        
        outd["Principals"] = np.array(self.lg.princip_list) 
        return outd 
    
    ##################################################
    #################### Testing ####################

    def init_estimates(self, N_init = -1, delta = 1):
        ts = self.lg.worker_df['fn_start'][0]
        te = self.lg.worker_df.iloc[-1]['fn_end']
        
        if N_init == -1: # perform on the whole trace
            N_init = (te-ts).total_seconds()/delta
            N_init = int( N_init )

        A, W = self.build_A_W(ts, N_init, delta)
        X = self.pow_cvx(A, W)
        return X 

    ##################################################
    ##################################################
    
    def over_time(self, N, delta, opt_fn, opt_args=None):
        """ Iterate over the entire log  """

        ts = self.lg.worker_df['fn_start'][0]
        te = self.lg.worker_df.iloc[-1]['fn_end']
        out = []
        window = N*delta 
        while ts < te + pd.Timedelta(seconds=-window):
            xval, _ = opt_fn(ts, N, delta) #opt_args 
            out.append(xval)
            ts = ts + pd.Timedelta(seconds=window)

        return out 

    ##################################################
    ################## Total Energy Related ###########
    ##################################################

    def get_total_energy_from_first_invoke( self, p_col='perf_rapl' ):
        t = 'timestamp'
        fs = 'fn_start'
        
        pdf = self.lg.power_df
        pdf = pdf.reset_index()
        wdf = self.lg.worker_df
        
        f_invk = wdf.iloc[0][fs]
        
        pdf = pdf[ pdf[t] >= f_invk ]
        s = 'second'
        if s in pdf.columns:
            pdf = pdf.drop_duplicates( s )
        
        pdf = pdf[p_col].cumsum()
        t_eng = pdf.iloc[-1]
        # display( 'Total energy {:.1f} Joules since first invoke'.format( t_eng ) )
        return t_eng  

    ##################################################
    #################### Signal Processing ###########
    ##################################################

    def equalize_array_size(self, array1,array2):
        '''
        reduce the size of one sample to make them equal size. 
        The sides of the biggest signal are truncated
        Args:
            array1 (1d array/list): signal for example the reference
            array2 (1d array/list): signal for example the target
        Returns:
            array1 (1d array/list): middle of the signal if truncated
            array2 (1d array/list): middle of the initial signal if there is a size difference between the array 1 and 2
            dif_length (int): size diffence between the two original arrays 
        '''
        len1, len2 = len(array1), len(array2)
        dif_length = len1-len2
        if dif_length<0:
            array2 = array2[int(np.floor(-dif_length/2)):len2-int(np.ceil(-dif_length/2))]
        elif dif_length>0:
            array1 = array1[int(np.floor(dif_length/2)):len1-int(np.ceil(dif_length/2))]
        return array1,array2, dif_length

    ##################################################

    def chisqr_align(self, reference, target, roi=None, order=1, init=0.1, bound=1):
        '''
        Align a target signal to a reference signal within a region of interest (ROI)
        by minimizing the chi-squared between the two signals. Depending on the shape
        of your signals providing a highly constrained prior is necessary when using a
        gradient based optimization technique in order to avoid local solutions.
        Args:
            reference (1d array/list): signal that won't be shifted
            target (1d array/list): signal to be shifted to reference
            roi (tuple): region of interest to compute chi-squared
            order (int): order of spline interpolation for shifting target signal
            init (int):  initial guess to offset between the two signals
            bound (int): symmetric bounds for constraining the shift search around initial guess
        Returns:
            shift (float): offset between target and reference signal 
        
        Todo:
            * include uncertainties on spectra
            * update chi-squared metric for uncertainties
            * include loss function on chi-sqr
        '''
        reference, target, dif_length = self.lg.equalize_array_size(reference,target)
        if roi==None: roi = [0,len(reference)-1] 
      
        # convert to int to avoid indexing issues
        ROI = slice(int(roi[0]), int(roi[1]), 1)

        # normalize ref within ROI
        reference = reference/np.mean(reference[ROI])

        # define objective function: returns the array to be minimized
        def fcn2min(x):
            shifted = shift(target,x,order=order)
            shifted = shifted/np.mean(shifted[ROI])
            return np.sum( ((reference - shifted)**2 )[ROI] )

        # set up bounds for pos/neg shifts
        minb = min( [(init-bound),(init+bound)] )
        maxb = max( [(init-bound),(init+bound)] )

        # minimize chi-squared between the two signals 
        # print((minb, maxb))
        result = minimize(fcn2min,init,method='L-BFGS-B',bounds=[ (minb,maxb) ])
        # print(result.x)
        
        return result.x[0] + int(np.floor(dif_length/2))

    ##################################################
    ######## BASIC OPTIMIZATION METHODS ##############
    
    def pow_lstsq(self, ts, N, delta):
        """ Simple least squares min of A@X -W """
        A, W = self.build_A_W(ts, N, delta)
        # subtract idle power? 
        sol = scipy.optimize.lsq_linear(A, W, bounds=(0,np.inf))
        return sol.x

    ##################################################

    def pow_cvx_time(self, t, N, delta):
        A, W = self.build_A_W(ts, N, delta)
        return self.pow_cvx(A, W)
    
    
    def pow_cvx(self, A, W):
        # Power estimate using cvxpy https://www.cvxpy.org/examples/basic/least_squares.html
        M = self.lg.M
        
        x = cp.Variable(M)
        cost = cp.sum_squares(A @ x - W)
        
        prob = cp.Problem(cp.Minimize(cost), [x >= self.min_fn_power])
        prob.solve()

        return x.value 


    def pow_cvx_ND(self, A, W):
        # Power estimate using cvxpy https://www.cvxpy.org/examples/basic/least_squares.html
        M = self.lg.M
        D = self.num_features
        
        x = cp.Variable(M, D)
        cost = cp.sum_squares(A @ x - W)
        
        prob = cp.Problem(cp.Minimize(cost), [x >= self.min_fn_power]) # x >=[min, 0, 0, 0]
        prob.solve()

        return x.value 



    ##################################################

    def pow_rglrz_lstsq(self, ts, N, delta, xprev=None):
        """ Min |A@x -W| + r|x-xprev| """

        A, W = self.build_A_W(ts, N, delta)
        M = self.M
        
        x = cp.Variable(M)
        if xprev is None:
            xprev = np.zeros(M)
        r = 0.6 # probably want to decay this over time? or increase it? 

        #princip_importance = importance_wts_per_fn(A)
        princip_importance = np.ones(M)

        regularization_term = cp.sum_squares(cp.multiply(princip_importance, x-xprev)) 
        cost = cp.sum_squares(A @ x - W) + r*regularization_term

        prob = cp.Problem(cp.Minimize(cost), [x >= 0])
        prob.solve()

        return x.value

    ##################################################

    def solv_eliminate(self, A, W, Xorig, Xnew):
        """ Use elimination to solve for new, from Xorig """
        # bounds are the key? 
        M = self.M
        
        x = cp.Variable(M)
        cost = cp.sum_squares(A @ x - W)
        
        prob = cp.Problem(cp.Minimize(cost), [x >= self.min_fn_power])
        prob.solve()

        return x.value 



    ##################################################

    def get_execution_times( self ):

        worker_df = self.lg.worker_df

        e = 'fn_end'
        s = 'fn_start'
        et = 'exec_time'
        f = 'fqdn'

        wdf = worker_df.copy()
        wdf[et] = wdf[e] - wdf[s]
        wdf[et] = wdf[et].astype('int64')/10**9

        cmb = wdf.groupby(f).agg([np.mean,np.std])
        
        return cmb

