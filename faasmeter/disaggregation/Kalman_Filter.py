import os

import pandas as pd
import numpy as np

import cvxpy as cp
from scipy.optimize import minimize

import copy 

class Kalman_Filter():

    # https://www.kalmanfilter.net/kalman1d.html

    ldf = None  # Main log pre-processing, including matrix stuff

    r_c = 0.2 #Measurement error constant  
    p_c = 0.3 #estimation error constant 
    p_prev = p_c
    update_p = False # p = (1-k)p
    K_min = 0.1 
    K_max = 0.9 
    x_prev = None
    x_init = None 
    J_prev = None
    X_prev = None 
    J_init = None 
    A_hist = None # this is some weighted sum, not pure 
    steps = 1
    x_hist = [] # All the power values over time ..
    K_hist = [] # Kalman gains over time?
    time_hist = []
    ks_hist = [] # Kalman properties over time
    t0 = None 
    cumN = 0
    update_type = "kalman" 
    kf_type = 'x'
    P_prev = None
    alpha = 0.8
    beta = 0.2
    gamma = 0.1
    lat_var_init = None 
    
    ##################################################
    
    def init(self):
        self.ks_hist = []
        self.x_hist = []
        self.time_hist = []
        self.A_hist = None
        self.K_hist = []
        self.steps = 0
        self.x_prev = None
        self.N = 10
        self.delta = 1
        self.kf_type = 'x' #classic 
        
        
    ##################################################

    def kalman_over_time(self, N_init, N, delta, update_type="kalman", max_steps=np.inf):
        """ Iterate over the entire log  """
        ldf = self.ldf
        ts = ldf.core.get_start_ts()
        te = ldf.core.get_end_te() 
        self.t0 = ts
        out = []
        window = N*delta
        self.N = N
        self.N_init = N_init
        self.delta = delta 

        # First step may be bigger to collect more data:
        if N_init > 0:
            A, W = ldf.core.build_A_W(ts, N_init, delta)
            self.x_prev = self.ldf.core.pow_cvx(A, W)
            self.x_init = self.x_prev 
            te_init = ts + pd.Timedelta(seconds=N_init) 
            outd = ldf.core.pow_full_breakdown(self.x_prev, ts, te_init)
            
            self.J_prev = outd["Energy"]
            self.J_init = self.J_prev 
            self.time_hist.append(ts)
            self.A_hist = self.ldf.core.invok_fracs(A)
            #print( ldf.core.fn_latencies_in_interval(ts, te) )
            #print( ldf.M )
            self.lat_var_init = np.array([np.std(x) for x in ldf.core.fn_latencies_in_interval(ts, te)])
            #print( self.lat_var_init )
            #exit(-1)

            Anet, Jnet = self.ldf.core.summarize_A_W(A, W)
            Cnet, _ = self.ldf.core.summarize_A_W(A, W) 
            
            # Add the initial step also in ks 
            ks = dict()            
            ks["t"] = ts
            ks["steps"] = 1
            ks["Cnet"] = Cnet
            ks['P'] = 0
            ks['P_next'] = 0
            ks["J_init"] = self.J_init 
            ks["Anet"] = Anet
            ks["W_obs"] = np.mean(W)
            ks['x'], ks['J'], ks['J_contrib'], ks['lat']  = outd['Power'], outd['Energy'], outd['Energy-contrib'], outd['Avg-fn-times']
            self.ks_hist.append(copy.deepcopy(ks))

        t = ts + pd.Timedelta(seconds=N_init)

        while self.steps <= max_steps and t < te:
            # TODO: Determine if we need kalman update or a new least-squares?
            # based on \dA. If small: kalman step. If new fn: elimination. If large: regularized lst squares, or even a reset.
            # if self.diff_in_workload(A) < eps: KF, else ...
            # New function handling here too..
            # if (new function): Any column of A_hist = 0

            if self.update_type == "kalman":
                xval, A, ks = self.kalman_step(t, N, delta)
            elif self.update_type == "memoryless":
                xval, A, ks = self.memoryless_step(t, N, delta)
            elif self.update_type == "cumulative":
                xval, A, ks = self.cumulative_step(t, N, delta)

            self.steps = self.steps + 1
            self.time_hist.append(t)
            self.ks_hist.append(copy.deepcopy(ks))
            self.update_A_hist(A)
            t = t + pd.Timedelta(seconds=window)

        return self.steps

    ##################################################

    def memoryless_step(self, t, N, delta):
        """ memoryless """
        C, W = self.ldf.core.build_A_W(t, N, delta)
        x = self.ldf.core.pow_cvx(C, W)
        A = self.ldf.core.build_A_matrix(t, N, delta, "A")
        ks = dict()
        ks["t"] = t
        ks["steps"] = self.steps
        ks["x"], ks["J"], ks["J_contrib"], ks["lat"]  = self.ldf.core.fn_power_to_energ(x, t, t+pd.Timedelta(seconds=N*delta))
        ks["A_hist"] = self.A_hist
        Anet, Jnet = self.ldf.core.summarize_A_W(A, W)
        Cnet, _ = self.ldf.core.summarize_A_W(A, W) 
        ks["Anet"] = Anet
        ks["Cnet"] = Cnet 
        ks["Jnet"] = Jnet 
        return x, A, ks 

    ##################################################
    
    def regularized_step(self, t, N, delta):
        """ Minimize total power error and changes to X over time """
        pass 
    
    ##################################################

    def cumulative_step(self, t, N, delta):
        """ memoryless """
        cumN = self.cumN + N
        self.cumN = cumN 
        C, W = self.ldf.core.build_A_W(self.t0, cumN, delta)
        x = self.ldf.core.pow_cvx(C, W)
        A = self.ldf.core.build_A_matrix(self.t0, cumN, delta, "A")
        
        ks = dict()
        ks["t"] = t
        ks["steps"] = self.steps
        ks["x"], ks["J"], ks["J_contrib"], ks["lat"]  = self.ldf.core.fn_power_to_energ(x, t, t+pd.Timedelta(seconds=cumN*delta))
        ks["A_hist"] = self.A_hist
        Anet, Jnet = self.ldf.core.summarize_A_W(A, W)
        Cnet, _ = self.ldf.core.summarize_A_W(C, W) 
        ks["Anet"] = Anet
        ks["Cnet"] = Cnet 
        ks["Jnet"] = Jnet 
        return x, A, ks 

    
    ##################################################

    def handle_new_function(self, Xorig, A, C, W):
        """ Returns updated Xorig if new functions are found """
        if self.A_hist is None:
            return False, Xorig 
        Ahist = self.A_hist
        active_fns = np.nonzero(A)[0]
        ahist_active = Ahist[active_fns]
        new_fns = np.nonzero(ahist_active)[0]
        if len(new_fns) > 0:
            Xnew = self.ldf.core.pow_cvx(C, W)
            Xorig[new_fns] = Xnew[new_fns] 
            return True, Xorig 

        return False, Xorig 
        
    ##################################################

    def kalman_step(self, t, N, delta):
        if self.kf_type == 'x':
            return self.kalman_step_x(t, N, delta)
        elif self.kf_type == 'j':
            return self.kalman_step_j(t, N, delta)
        elif self.kf_type == 'x-n':
            return self.kalman_step_x_n(t, N, delta)
            

    ##################################################
                           
    def kalman_step_x(self, t, N, delta): 
        """ Process the next batch of readings and compute x. 
        This is the 1-dimensional version."""
        
        ldf = self.ldf 
        C, W = ldf.core.build_A_W(t, N, delta) # Can either be A/C type 
        x_prev = self.x_prev

        # error with previous predictions
        pred_val = C @ x_prev 
        pe = W - pred_val  # for each timestep 
        observed_err = np.sum(pe)

        A = ldf.core.build_A_matrix(t, N, delta, ac_type="A")
        # XXX FIX the double counting! But doesnt matter because we normalize anyways ... 
        A_sum = np.sum(A, axis=0)

        #newfn, x_prev = self.handle_new_function(x_prev,  A_sum, C, W)


        # This error now has to be distributed to all the principals.
        # error/a takes into account contribution, but can affect small functions and make them negative 
        # Ka = error*x_a/pred_val ? strictly: \sum \sum(a)*x_prev
        # We want delta-x_i/x_i = constant for all i to avoid 

        err_contrib = x_prev*observed_err/np.sum(pred_val)
        
        r =  self.r_c/np.sqrt(N*delta) # Measurement err depends on time-span?        
        # TODO: strictly proportional to time... (t_n - t)?

        wd = self.diff_in_workload(A) # compares A-A_hist
        base_uncertain  = 1.0 #wd is small because normalized.
        p_prev = self.p_prev*(base_uncertain+wd)
        K = p_prev/(p_prev + r) #close to 1

        K = min(self.K_max, K)
        K = max(self.K_min, K)

        #K, err_contrib = self.kf_wt(x_prev, C, W, A, r)
        # Main update rule 
        x = x_prev + K*err_contrib
        # Also clip this to ensure that we dont reduce below 0 ?
        x = np.clip(x, 0, np.max(x))

        new_err = np.sum(W - C @ x)
        # Hopefully smaller than the previous one!

        if self.update_p:
            # This is important. reduces the process error over time.
            # But in our case, it increases the error gap. 
            p_new = (1-K)*p_prev
            self.p_prev = p_new

        self.x_prev = x

        ##### Output of the step 
        
        ks = dict()
        ks["t"] = t
        ks["steps"] = self.steps
        te = t+pd.Timedelta(seconds=N*delta)
        ks["x"], ks["J"], ks["J_contrib"], ks["lat"]  = self.ldf.core.fn_power_to_energ(x, t, te)

        if self.ldf.output_type == 'indiv':
            J, cps, ids = self.ldf.core.J_contrib_total(x, t, te)
            J_total = J + cps + ids
            J_total[J_total<0] = 0.0
            ks['cps'] = cps
            ks['ids'] = ids
            ks['J_con2'] = J
            ks['J_total'] = J_total

            A_1, J_per_invok, cp_share, idle_share = self.ldf.core.pow_to_J_total(x, t, te)
            ks['J_invok'] = J_per_invok
            ks['cp_share'] = cp_share
            ks['idle_share'] = idle_share
            ks['W_total'] = ks['J_total']/float(N*delta)

        ks["W_obs"] = np.mean(W)
        ks['wd'] = wd
        ks['err_contrib'] = err_contrib
        ks['r'] = r
        ks['p_prev'] = p_prev 
        ks["K"] = K
        ks["observed_err"] = observed_err
        ks["new_err"] = new_err
        ks["A_hist"] = self.A_hist
        Anet, Jnet = self.ldf.core.summarize_A_W(A, W)
        Cnet, _ = self.ldf.core.summarize_A_W(C, W) 
        ks["Anet"] = Anet
        ks["Apersecnet"] = Anet
        ks["Cnet"] = Cnet 
        ks["Jnet"] = Jnet 
        return x, A, ks 

    
    ##################################################


    def get_fn_lat_var(self):
        """ Latency in function variance used in Kalman gain."""
        if self.lat_var_init is None: 
            M = self.ldf.M 
            #print( M )
            return 0.1*np.arange(1,M+1) # variance/mean latency of all functions
        else:
            # TODO: Update the variance also... 
            return self.lat_var_init 
        # XXX: Get this from actual data.

    ##############################

    def K_update_simple(self, p, H, r):
        """ p is vector of latency variances . 
        H is vector of number of invocations """
        Knumer = p*H 
        Kdenom = np.dot(p, H) + r 
        return Knumer, Kdenom

    ##############################

    def handle_inactive(self, A):
        """ Adjust the coefficients such that inactive principals are not updated """
        M = len(A)
    
        alpha = self.alpha*np.ones(M)
        beta = self.beta*np.ones(M)
        inactive=np.where(A==0)[0]
        for i in inactive:
            alpha[i] = 1.0
            beta[i] = 0.0
            
        return alpha, beta 
    
    
    def kalman_step_x_n(self, t, N, delta):
        """ Update X using N-D KF"""
        #print("#"*20)
        #print("# Step x {},{},{}".format(t,N,delta))
        ldf = self.ldf 
        C, W = ldf.core.build_A_W(t, N, delta) # Can either be A/C type
        Apersec = ldf.core.build_A_matrix(t, N, delta, ac_type="A")
        Apersecnet, _ = self.ldf.core.summarize_A_W(Apersec, W)
        te = t + pd.Timedelta(seconds=N*delta)
        A = ldf.core.get_A_row(te, N*delta)
        Anet = A 
        #Anet = np.sum(A, axis=0) #use core.invok_fracs for normalizing
        Cnet = np.sum(C, axis=0)
        alpha, beta, gamma = self.alpha, self.beta, self.gamma 
        M = ldf.M 
        p = 0.3 #process noise initial 
        r = 0.1 #measurement variance 
        
        # Three main terms 
        # J = \alpha J_prev + \beta C X_init + \gamma process-noise

        alpha, beta = self.handle_inactive(Anet) 
        
        X1 = alpha * self.x_prev
        X_base = self.x_init # X_base is the baseline power, X_init or X_prev 
        #J2 = beta * (Cnet @ X_base)  # TODO: Convert from total energy to this function's contribution? A/sum(A)?

        #J2 = np.mean(W) * ldf.core.invok_fracs(C)
        x = self.ldf.core.pow_cvx(C, W)

        # XXX : Want to 'freeze' inactive principals .. 
        X2 = beta* x

        fn_lat_CoV = self.get_fn_lat_var() 
        #print("{}".format(fn_lat_CoV))
        X3 = gamma * np.random.normal(scale=fn_lat_CoV) 
        X = X1 + X2 + X3
        X_minus = X1 + X2  # no noise term 
        #J_prev = J
        
        # Eqn2: Z = HX + measurement noise
        # Z==observed energy consumption , HX == AJ 
        Z_observed = np.mean(W)
        Z_minus = np.mean(C @ X_minus) #np.dot(Cnet, X_minus)
        innovation = Z_observed - Z_minus 
        
        # Now update
        if self.P_prev is None:
            P = gamma * fn_lat_CoV #M,1
            #print("{} {}".format(gamma, fn_lat_CoV))
            Pmatrix = P*np.eye(M)
        else:
            P_prev_vec = np.diagonal(self.P_prev)
            P = alpha*P_prev_vec + gamma*fn_lat_CoV #M,1
            Pmatrix = P*np.eye(M)
            
        # TODO: P is supposed to be =  alpha*P + gamma*fn_lat_Cov 

        # TODO: Simplify the weights to just pa/sum(pa)
        H = Anet
        Ht = np.transpose(H)

        #print("{} {}".format(Pmatrix, Ht))
        Knumer = Pmatrix @ Ht # M,1 
        Kdenom = (H @ Pmatrix @ Ht) + r # scalar

        #Knumer, Kdenom = self.K_update_simple(fn_lat_CoV, H, r)

        K = Knumer/Kdenom # M,1
        #print("{} {}".format(Knumer, Kdenom))

        P_next = (1.0 - (K @ H))*Pmatrix # scalar mult M,M
        
        self.P_prev = P_next 

        #P_eff = np.dot(P, Anet) #numerator
        #K = P_eff/(P_eff + r)  #scalar!? 
        
        #print("{} {} {}".format(X_minus, K, innovation))
        X_next = X_minus + K*innovation
        # Main issue is bounds negative, too high, etc.
        # Need robustness? 
        self.x_prev = X_next
        
        # Output
        ks = dict()
        ks["t"] = t
        ks["steps"] = self.steps
        ks["Cnet"] = Cnet
        ks['P'] = P
        ks['P_next'] = P_next 
        ks["J_init"] = self.J_init 
        ks["Anet"] = Anet
        ks["Apersecnet"] = Apersecnet 
        ks["Z_obs"] = Z_observed
        ks["Z_minus"] = Z_minus 
        #ks["J_minus"] = J_minus 
        ks["W_obs"] = np.mean(W)
        ks["observed_err"] = innovation
        ks["K"]  = K
        ks["Jbreakdown"] = (X1, X2, X3)

        ks['x'], ks['J'], ks['J_contrib'], ks['lat']  = self.ldf.core.fn_power_to_energ(X_next, t, te)
        #print("{} {} {}".format(X_next, t, te))
        #print(ks['J'])
        #exit(-1)

        #xval to be returned, never used though 
        return X_next, A, ks 
    
    ###################################################

    def kalman_step_j(self, t, N, delta):
        """ Update J instead of X """
        ldf = self.ldf 
        C, W = ldf.core.build_A_W(t, N, delta) # Can either be A/C type
        ### A = ldf.core.build_A_matrix(t, N, delta, ac_type="A")
        te = t + pd.Timedelta(seconds=N*delta)
        A = ldf.core.get_A_row(te, N*delta)
        Anet = A 
        #Anet = np.sum(A, axis=0) #use core.invok_fracs for normalizing
        Cnet = np.sum(C, axis=0)
        alpha, beta, gamma = 0.6, 0.4, 0.1
        M = ldf.M 
        p = 0.3 #process noise initial 
        r = 0.1 #measurement variance 
        
        # Three main terms 
        # J = \alpha J_prev + \beta C X_init + \gamma process-noise
        J1 = alpha * self.J_prev
        X_base = self.x_init # X_base is the baseline power, X_init or X_prev
        
        #J2 = beta * (Cnet @ X_base)  # TODO: Convert from total energy to this function's contribution? A/sum(A)?

        #J2 = np.mean(W) * ldf.core.invok_fracs(C)
        x = self.ldf.core.pow_cvx(C, W)
        
        x_inst, J_inst, J_contrib_inst, lat_inst  = self.ldf.core.fn_power_to_energ(x, t, te)

        J2 = beta* J_inst 

        fn_lat_CoV = 0.1*np.arange(1,M+1) # variance/mean latency of all functions 
        J3 = gamma * np.random.normal(scale=fn_lat_CoV) 
        J = J1 + J2 + J3
        J_minus = J1 + J2  # no noise term 
        #J_prev = J
        
        # Eqn2: Z = HX + measurement noise
        # Z==observed energy consumption , HX == AJ 
        Z_observed = np.mean(W)*(N*delta)
        Z_minus = np.dot(Anet, J_minus)
        innovation = Z_observed - Z_minus 
        
        # Now update
        if self.P_prev is None:
            P = gamma*fn_lat_CoV #M,1
            Pmatrix = P*np.eye(M)
        else:
            P_prev_vec = np.diagonal(self.P_prev)
            P = alpha*P_prev_vec + gamma*fn_lat_CoV #M,1
            Pmatrix = P*np.eye(M)
            
        # TODO: P is supposed to be =  alpha*P + gamma*fn_lat_Cov 

        H = Anet
        Ht = np.transpose(H)
        Knumer = Pmatrix @ Ht # M,1
        Kdenom = (H @ Pmatrix @ Ht) + r # scalar
        K = Knumer/Kdenom # M,1

        P_next = (1.0 - (K @ H))*Pmatrix # scalar mult M,M
        self.P_prev = P_next 

        #P_eff = np.dot(P, Anet) #numerator
        #K = P_eff/(P_eff + r)  #scalar!? 
        
        J_next = J_minus + K*innovation
        # Main issue is bounds negative, too high, etc.
        # Need robustness? 
        self.J_prev = J_next
        
        # Output
        ks = dict()
        ks["t"] = t
        ks["steps"] = self.steps
        ks["J"] = J_next
        ks["Cnet"] = Cnet 
        ks["J_init"] = self.J_init 
        ks["Anet"] = Anet
        ks["Z_obs"] = Z_observed
        ks["Z_minus"] = Z_minus 
        ks["J_minus"] = J_minus 
        ks["W_obs"] = np.mean(W)
        ks["observed_err"] = innovation
        ks["K"]  = K
        ks["Jbreakdown"] = (J1, J2, J3)
        
        #xval to be returned, never used though 
        return None, A, ks 

    ##################################################    
        
    def kalman_step_x2(self, t, N, delta): 
        """ Process the next batch of readings and compute x """
        ldf = self.ldf 
        C, W = ldf.core.build_A_W(t, N, delta) # Can either be A/C type 
        x_prev = self.x_prev

        # error with previous predictions
        pred_val = C @ x_prev 
        pe = W - pred_val  # for each timestep 
        observed_err = np.sum(pe)

        A = ldf.core.build_A_matrix(t, N, delta, ac_type="A")
        A_sum = np.sum(A, axis=0)

        #newfn, x_prev = self.handle_new_function(x_prev,  A_sum, C, W)


        # This error now has to be distributed to all the principals.
        # error/a takes into account contribution, but can affect small functions and make them negative 
        # Ka = error*x_a/pred_val ? strictly: \sum \sum(a)*x_prev
        # We want delta-x_i/x_i = constant for all i to avoid 

        err_contrib = x_prev*observed_err/np.sum(pred_val)
        # Also want to make this proportional to variance in latency ...
        
        r =  self.r_c/np.sqrt(N*delta) # Measurement err depends on time-span?        
        # TODO: strictly proportional to time... (t_n - t)?

        wd = self.diff_in_workload(A) # compares A-A_hist
        base_uncertain  = 1.0 #wd is small because normalized.
        p_prev = self.p_prev*(base_uncertain+wd)
        K = p_prev/(p_prev + r) #close to 1

        K = min(self.K_max, K)
        K = max(self.K_min, K)

        #K, err_contrib = self.kf_wt(x_prev, C, W, A, r)
        # Main update rule 
        x = x_prev + K*err_contrib
        # Also clip this to ensure that we dont reduce below 0 ?
        x = np.clip(x, 0, np.max(x))

        new_err = np.sum(W - C @ x)
        # Hopefully smaller than the previous one!

        if self.update_p:
            # This is important. reduces the process error over time.
            # But in our case, it increases the error gap. 
            p_new = (1-K)*p_prev
            self.p_prev = p_new

        self.x_prev = x

        ##### Output of the step 
        
        ks = dict()
        ks["t"] = t
        ks["steps"] = self.steps
        te = t+pd.Timedelta(seconds=N*delta)
        ks["x"], ks["J"], ks["J_contrib"], ks["lat"]  = self.ldf.core.fn_power_to_energ(x, t, te)

        if self.ldf.output_type == 'indiv':
            J, cps, ids = self.ldf.core.J_contrib_total(x, t, te)
            J_total = J + cps + ids
            J_total[J_total<0] = 0.0
            ks['cps'] = cps
            ks['ids'] = ids
            ks['J_con2'] = J
            ks['J_total'] = J_total

            A_1, J_per_invok, cp_share, idle_share = self.ldf.core.pow_to_J_total(x, t, te)
            ks['J_invok'] = J_per_invok
            ks['cp_share'] = cp_share
            ks['idle_share'] = idle_share
            ks['W_total'] = ks['J_total']/float(N*delta)

        ks["W_obs"] = np.mean(W)
        ks['wd'] = wd
        ks['err_contrib'] = err_contrib
        ks['r'] = r
        ks['p_prev'] = p_prev 
        ks["K"] = K
        ks["observed_err"] = observed_err
        ks["new_err"] = new_err
        ks["A_hist"] = self.A_hist
        Anet, Jnet = self.ldf.core.summarize_A_W(A, W)
        Cnet, _ = self.ldf.core.summarize_A_W(C, W) 
        ks["Anet"] = Anet
        ks["Cnet"] = Cnet 
        ks["Jnet"] = Jnet 
        return x, A, ks 
    

    ##################################################
    
    def kf_wt(self):
        """ For each principal, the updates should be proportional to weight.
        Weight can be determined by various factors like changes in A, 
        variance in latency, etc. """
        # p is the main process error variance 
        pass

    ##################################################
 
    def update_A_hist(self, A):
        nc = self.ldf.core.invok_fracs(A) # normalized column sums 
        if self.A_hist is None:
            self.A_hist = nc 
        # add this to history
        # or some kind of weighted average of this?
        i = float(self.steps)
        self.A_hist = (i-1/i)*self.A_hist + (nc/i)

    ##################################################
    
    def diff_in_workload(self, A):
        nc = self.ldf.core.invok_fracs(A) #normalized column sums 
        # As A_hist grows, we need to normalize it? A already is.
        norm_a_hist = self.A_hist/np.sum(self.A_hist)
        return np.linalg.norm(norm_a_hist - nc)

    ##################################################

    def ks_over_time(self, key, drop_first=False):
        all_x = []
        if key == 'pow_contrib':
            # special column for stack plot
            pow_contrib = self.ks_over_time("J_contrib")/(self.N*self.delta)
            pow_contrib.loc[0] = np.array(pow_contrib[0:1]/(self.N_init/self.N))
            return pow_contrib 
        if not drop_first:
            s = 0 
        else:
            s = 1
        for ks in self.ks_hist[s:]:
            if key in ks: 
                all_x.append(ks[key])
            else:
                all_x.append(np.nan)
        return pd.DataFrame(np.array(all_x))

    ##################################################

    def pow_cvx(self, A, W):
        # Power estimate using cvxpy https://www.cvxpy.org/examples/basic/least_squares.html
        M = self.ldf.M
        
        x = cp.Variable(M)
        cost = cp.sum_squares(A @ x - W)
        prob = cp.Problem(cp.Minimize(cost), [x >= 0.01])
        prob.solve()

        return x.value 

    ########################################

    def post_KF(self):
        """ Apply any local post-processing  """
        pass 

    def KF_err_metrics(self):
        """ RMSE total, J avg, variance etc """

        outd = dict()
        outd['J_init'] = self.J_init
        
        outd['J-mean'] = np.mean(self.ks_over_time('J'))
        outd['J-var'] = np.std(self.ks_over_time('J'))

        outd['MAE'] = np.mean(np.abs(self.ks_over_time('observed_err')))
        #outd['RMSE']  =
        pow_contrib = self.ks_over_time('pow_contrib')
        estim_pow = pow_contrib.sum(axis=1)
        estim_power = pow_contrib.sum(axis=1)
        observed_power = np.array(self.ks_over_time('W_obs')).flatten()
        outd['RMSE'] = np.sqrt(np.sum(np.square(estim_power - observed_power)))
        return outd 

        
    ##############################

        
