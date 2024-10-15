import os
import json
from json import JSONDecodeError
import re

import math
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np

from dateutil import tz
from datetime import timedelta 

from faasmeter.faasmeter.disaggregation.core import Core
from faasmeter.faasmeter.helper_funcs import * 
from faasmeter.faasmeter.quirks import *
from faasmeter.faasmeter.config import *

class Logs_to_df():
    log_loc = '.'

    #standard filenames for now 
    worker_log = 'worker_worker1.log'
    docker_log = 'docker.log'

    ipmi_log = 'energy-ipmi.log'
    rapl_log = 'rapl.csv'
    perf_log = 'perf.csv'
    cpu_log = 'cpu.csv'
    proc_log = 'process.log'
    battery_log = 'battery.log'
    igpm_log = 'igpm.log'
    tegra_log = 'tegrastats.csv'

    # P-Suggest: Map from src->(log, df, status=Present/None, ...)
    # These are the main df outputs
    worker_df = None
    process_cpu_df = None
    ipmi_df = None
    bat_df = None
    igpm_df = None
    tegra_df = None
    scaphandre_df = None
    rapl_df = None
    cpu_df = None
    perf_df = None 
    power_df = None
    power_df_secs = None
    proc_df = None
    status_df = None
    sw_core = None
    combined_hwpc = None
    scaphandre_rawdf = None
    scaphandre_cdf = None
    scaphandre_funcs_df = None
    scaphandre_scaphandre_df = None
    scaphandre_total_df = None

    ##################################################
    
    collapse_fqdns = False # if true - 'video_processing.3.0-0', 'video_processing.1.0-0' would collapse to video_processing
    fndict = None # dictionary for original name to short name conversion
    order_preference = ['json', 'image', 'video', 'ml_train']
    full_principals = False # Adds control-plane and system as 2 additional columns. Idle is different? 
    should_collapse_similar_funcs = True
    M = 0 #number of principals. functions + <cplane, sys> depending on output_type
    extra_principals = ["cp"]
    fnlist = []
    princip_list = []
    a_type = "C" # "A" for activations - "C" for contributions  
    mc_type = "server" # "desk", "laptop"
    power_src = "sys" # or cpu . Only 2 options 
    power_col = "igpm" # igpm, ipmi, bat_power - based on mc_type, power_src 
    Widle = 0 
    Widle_cpu = 0 
    Widle_xrest = 0 
    cpu_cores = 0 
    min_fn_power = 1 # 1 Watt? Or depending on the CP overhead? 
    cpu_config = None
    first_invoke = None
    first_invoke_t = None
    output_type = "full" # indiv, marginal, total
    fix_sys_cpu = True 
    # full: \sum x = W
    # indiv: \sum x = W - Widle -Wcp
    # marginal: indiv + Wcp/invok-frac
    # total: marginal + idle/N 


    ###############################################

    def __init__(self, log_loc, mc_type='server', output_type='indiv'):
        """ location of logs, and the machine type """
        self.log_loc = log_loc
        self.mc_type = mc_type
        self.output_type = output_type
        self.populate_system_settings()
        self.core = Core( self )

        try:
            os.mkdir(log_loc + dir_lg2df)
        except FileExistsError:
            os.system('rm '+ log_loc + dir_lg2df + '/*') 

        print("Platform {} - analysis output_type {}".format( mc_type, output_type ) )

    ##################################################

    def get_fn_index(self, fn):
        # This is not needed but leaving here for future usecase maybe
        # print(self.fndict)
        if self.fndict is not None:
            i = self.fnlist.index( self.fndict[fn] )
        else:
            i = self.fnlist.index(fn)
        return i

    ##################################################

    def pow_col(self):
        if self.power_src == "sys":
            if self.mc_type == "server":
                return "ipmi"
            elif self.mc_type == "desktop":
                return "igpm"
            elif self.mc_type == "laptop":
                return "bat"
            else:
                return None 

        elif self.power_src == "cpu":
            return "energy_pkg"

        else:
            return None 

    ##################################################
    
    def populate_system_settings(self):
        """ Call this after updating output_type """
        #P-Suggest: These details should be read from a configuration file.
        if self.mc_type == "server":
            self.Widle = 100
            self.cpu_cores = 48 
            cpu_config = {
               'cpu_tdp': 150000000, # cpu thermal specification - should be same unit as power - uj victor
               'cpu_clock_step': 100,
               'cpu_freq_min': 1000,
               'cpu_freq_base': 2100,
               'cpu_freq_max': 3700,
               'history_size': 100, # number of datapoints
            }
            #P-Suggest: Add detailed comment why these and not others. What is "Contr"? 
            cfeatures = [
                        "UNHALTED_CORE_CYCLES",
                        "INSTRUCTION_RETIRED",
                        "Contr",
            ]

        elif self.mc_type == "desktop":
            self.Widle = 30 
            self.cpu_cores = 12 
            cpu_config = {
               'cpu_tdp': 65000000, # cpu thermal specification - should be same unit as power - uj victor
               'cpu_clock_step': 100,
               'cpu_freq_min': 800,
               'cpu_freq_base': 3000,
               'cpu_freq_max': 5900,
               'history_size': 100, # number of datapoints
               'tdp_cpu': 117.0, # max system power on desktop - max turbo power for cpu on desktop is 117W
                                 # /data2/ar/powerapi_personal/cpu_specs/Intel_ARK_SpecificationsChart_2023_02_24.csv
               'tdp_xrest': 80.0,
            }
            cfeatures = [
                        "UNHALTED_CORE_CYCLES",
                        "UNHALTED_REFERENCE_CYCLES",
                        "LLC_MISSES",
                        "INSTRUCTION_RETIRED",
                        "Contr",
            ]
        elif self.mc_type == "jetson":
            self.Widle = 0 
            self.cpu_cores = 12 
            cpu_config = {
               'cpu_tdp': 65000000, # cpu thermal specification - should be same unit as power - uj victor
               'cpu_clock_step': 100,
               'cpu_freq_min': 800,
               'cpu_freq_base': 3000,
               'cpu_freq_max': 5900,
               'history_size': 100, # number of datapoints
               'tdp_cpu': 117.0, # max system power on desktop - max turbo power for cpu on desktop is 117W
                                 # /data2/ar/powerapi_personal/cpu_specs/Intel_ARK_SpecificationsChart_2023_02_24.csv
               'tdp_xrest': 80.0,
            }
            cfeatures = [
                        "UNHALTED_CORE_CYCLES",
                        "UNHALTED_REFERENCE_CYCLES",
                        "LLC_MISSES",
                        "INSTRUCTION_RETIRED",
                        "Contr",
            ]
        else:
            pass 
        cpu_config['cfeatures'] = cfeatures
        self.cpu_config = cpu_config
        self.tegra_pcols = [
            'VDD_GPU_SOC_current',
            'VDD_CPU_CV_current',
            'VIN_SYS_5V0_current',
            'VDDQ_VDD2_1V8AO_current',
        ]
        if self.output_type == "full":
            self.full_principals = False
        else:
            self.full_principals = True

        # self.rnn_config( use_default=self.mc_type )

        return None

    ##################################################

    def timestamp_to_datetime( self, pdf, tname='timestamp', zones=['UTC','US/Eastern'], unit='ms' ):
        sw_cmb_t = pd.to_datetime(pdf[tname], unit=unit)
        sw_cmb_t = sw_cmb_t.dt.tz_localize(zones[0])
        sw_cmb_t = sw_cmb_t.dt.tz_convert(zones[1])
        sw_cmb_t = sw_cmb_t.dt.tz_localize(None)
        return sw_cmb_t

    def start_0_timestampe( self, pdf, start=None, tname='timestamp' ):
        pdf = pdf.copy()
        if start is None:
            start = self.power_df_t['timestamp'][0]

        pdf[tname] = pdf[tname] - start
        return pdf

    def timestamp_datetime_to_ts( self, pdf, tname='timestamp' ):
        df = pdf.copy()
        df[tname] = df[tname].dt.tz_localize('US/Eastern')
        df[tname] = df.timestamp.values.astype(np.int64) // 10 ** 6
        return df
    
    def sca_pid_to_func_dict_add(self, fdict, jmsg ):
        pid = jmsg['pid']
        cpid = jmsg['cpid']
        fqdn = jmsg['fqdn']
        if cpid not in fdict:
            fdict[pid] = fqdn
            fdict[cpid] = fqdn
        else:
            print("Warning: duplicate cpid found in worker.log - shouldn't have happened")

    def collapse_similar_funcs_v1(self, wdf):
        #chameleon.CPU.0-0 -> chamleon.CPU 
        func_set = set(wdf['fqdn'])
        prefixes = {}
        for f in func_set:
            fws = f.split('-')
            fprefix = fws[0]
            fnum = fws[1]
            if fprefix not in prefixes:
                prefixes[fprefix] = []
            prefixes[fprefix].append( fnum )
        rename_dic = {}
        for k,v in prefixes.items():
            if len(v) > 1:
                for vn in v:
                    old_name = '-'.join([k,vn,'0.0.1'])
                    new_name = '-'.join([k,'0','0.0.1'])
                    rename_dic[old_name] = new_name 
        def rename_fqdn( val ):
            if val in rename_dic:
                return rename_dic[val]
            return val
        wdf[tag_fqdn] = wdf[tag_fqdn].apply( rename_fqdn )
        return wdf

    def collapse_similar_funcs(self, wdf):
        #chameleon.CPU.0-0 -> chamleon 
        def rename_fqdn(fname):
            return fname.split('.')[0]
            
        wdf['fqdn'] = wdf['fqdn'].apply(rename_fqdn)
        return wdf

    

    def worker_log_to_df(self):
      """ Determine start and end time for each invocation """   
      INVOKE_TARGET_TRAIT = "iluvatar_worker_library::services::invocation::invoker_trait"
      INVOKE_TARGET_TRAIT_DIS = "iluvatar_worker_library::services::invocation::queueing_dispatcher"

      INVOKE_TARGET = "iluvatar_worker_library::services::containers::containerd::containerdstructs"
      INVOKE_NAME = "ContainerdContainer::invoke"  

      INVOKE_TARGET_S = "iluvatar_worker_library::services::containers::simulation::simstructs"
      INVOKE_NAME_S = "SimulatorContainer::invoke"  

      running = {}
      worker_log = os.path.join(self.log_loc, self.worker_log)
      if not os.path.exists(worker_log):
        print(f"Warning: Worker logs not available: {worker_log}")
        return
      
      
      def timestamp_to_pddate( time ):
        try:
            return pd.to_datetime( time )
        except:
            ptime = pd.to_datetime( time, format="%Y-%m-%d %H:%M:%S:%f+" )
            ptime = ptime.replace(tzinfo=tz.tzutc())
            local = ptime.astimezone(tz.tzlocal())
            local = local.replace(tzinfo=None)
            return local
        
      def timestamp(log):
        time = log["timestamp"]
        return timestamp_to_pddate( time )

      def invoke_start(log):
        fqdn = log["span"]["fqdn"]
        tid = log["span"]["tid"]
        t = timestamp(log)
        running[tid] = (t, fqdn)

      def invoke_end(log):
        tid = log["span"]["tid"]
        stop_t = timestamp(log)
        start_t, fqdn = running[tid]
        duration = stop_t - start_t
        return duration, start_t, stop_t

      worker_df_list = []
      worker_status_list = []
      sca_pid_to_func_dict = {}
      
      with open(worker_log, 'r') as f:
        while True:
          line = f.readline()
          if line == "":
            break
          try:
              log = json.loads(line)
          except Exception as e:
              print(e)
              print(line)
              continue 

          if log["fields"]["message"] == "tag_pid_mapping":
              self.sca_pid_to_func_dict_add( sca_pid_to_func_dict, log["fields"] )

          if log["fields"]["message"] == "new" or log["fields"]["message"] == "close":
              target = log["target"] == INVOKE_TARGET or log["target"] == INVOKE_TARGET_S
              span = log["span"]["name"] == INVOKE_NAME or log["span"]["name"] == INVOKE_NAME_S

          if log["fields"]["message"] == "new" and target and span:
            # invocation starting
            invoke_start(log)
            if self.first_invoke_t is None:
              self.first_invoke_t = timestamp(log)

          if log["target"] == INVOKE_TARGET_TRAIT and log["fields"]["message"] == "function results":
            fqdn = log["fields"]["fqdn"]
            tid = log["fields"]["tid"]
            start_t = timestamp_to_pddate(log["fields"]["t_start"])
            stop_t = timestamp_to_pddate(log["fields"]["t_end"])
            worker_df_list.append((start_t, stop_t, tid, fqdn))

          if (log["target"] == "iluvatar_worker_library::services::invocation::invoker" or log["target"] == INVOKE_TARGET_TRAIT_DIS) and log["fields"]["message"] == "Invocation complete":
            if "fqdn" not in log["fields"]:
              name = log['span']['function_name'].split('.')[0]
              fqdn = f"{name}-{log['span']['function_version']}.0.1"
              fqdn = f"{log['span']['function_name']}-{log['span']['function_version']}"
              # fqdn = log['span']['function_name'].split('.')[0]
              # fqdn = f"{fqdn}-{log['span']['function_version']}."
              # print("calc fqdn:", fqdn)
            else:
              fqdn = log["fields"]["fqdn"]
            tid = log["fields"]["tid"]
            if "t_start" in log["fields"]:
              start_t = timestamp_to_pddate(log["fields"]["t_start"])
              stop_t = timestamp_to_pddate(log["fields"]["t_end"])
            else:
              duration, start_t, stop_t = invoke_end(log)
            worker_df_list.append((start_t, stop_t, tid, fqdn))
              
            
          if log["fields"]["message"] == "current load status":
            ts = timestamp(log)
            status = json.loads(log["fields"]["status"])
            mem_pct = (float(status["used_mem"]) / float(status["total_mem"])) * 100
            if status["cpu_sy"] == None:
                continue
            sys_cpu_pct = status["cpu_us"] + status["cpu_sy"] + status["cpu_wa"]
            load_avg = status["load_avg_1minute"] / status["num_system_cores"]
            if "queue_len" not in status:
                status["queue_len"] = status["cpu_queue_len"]
            gpu_queue_len = 0
            if "gpu_queue_len" in status:
                gpu_queue_len = status["gpu_queue_len"]
            num_containers = 0
            if "num_containers" in status:
              num_containers = status["num_containers"]
            data = (ts, status["queue_len"], mem_pct, sys_cpu_pct, load_avg, status["num_running_funcs"], num_containers, status["cpu_us"], status["cpu_sy"],
                    status["cpu_wa"], gpu_queue_len)
            worker_status_list.append(data)

      worker_df = pd.DataFrame(worker_df_list, columns=['fn_start','fn_end','tid','fqdn'])
      if self.should_collapse_similar_funcs:
          # print("collapsing funcs") 
          worker_df = self.collapse_similar_funcs( worker_df )

      self.status_df = pd.DataFrame.from_records(worker_status_list, columns=["timestamp", "queue_len", "mem_pct", "sys_cpu_pct", "load_avg", "num_running_funcs",
                                                                              "num_containers", "cpu_us", "cpu_sy", "cpu_wa", "gpu_queue_len"], index=["timestamp"])
      if self.collapse_fqdns:
          nset = {}
          # print(worker_df["fqdn"].unique())
          fnlist = list(set(worker_df['fqdn']))
          # print(fnlist)
          for f in fnlist:
            # print("finding:", f)
            r = re.search('(.*)\..[0-9]*\..*', f)
            if r is not None and len(r[0]) > 1:
                nset[f] = r[1]
            
          fnlist = []
          # update worker_df to have fqdns according to this dictionary 
          for k,v in nset.items():
            worker_df[ worker_df['fqdn'] == k ]['fqdn'] = v
            fnlist.append(v)
          
          self.fndict = nset 
          self.fnlist = sorted(list( set(fnlist) ))
          self.princip_list = list( set(fnlist) )
      else:
          self.fnlist = sorted(list(set(worker_df['fqdn'])))
          self.princip_list = list(set(worker_df['fqdn']))

      #P-XXX: Move or remove? Hard-coded fn names 
      if self.order_preference != None:
          new_order = []
          for f in self.order_preference:
              plist = [ function_name_to_paper( fn ) for fn in self.princip_list ]
              if f in plist:
                  new_order.append( self.princip_list.pop( plist.index(f) ) )

          new_order.extend( self.princip_list )
          self.princip_list = new_order 
          self.fnlist = new_order

      self.M = len(self.fnlist)
      
      if self.full_principals:
          self.M = self.M + len(self.extra_principals)
          self.princip_list = self.princip_list + self.extra_principals
      
      if quirks['quirk_trim_end']:
          worker_df = quirk_workerdf_trim_end( worker_df )
      if quirks['quirk_trim_end_by_time']:
          worker_df = quirk_workerdf_trim_end_by_time( worker_df, quirks_data['quirk_trim_end_by_time'] )
      self.worker_df = worker_df
      self.sca_pid_to_func_dict = sca_pid_to_func_dict
      pids = []
      funcs = []
      for k,v in self.sca_pid_to_func_dict.items():
          pids.append( k )
          funcs.append( v )
      self.sca_pid_to_func_df = pd.DataFrame( funcs, index=pids )

    ##################################################

    # columns: t_s, t_e, fqdn, tid
    # Add: Exec_t, Net, Disk, CPU, GPU, ... 
    def worker_log_to_df_ND(self):
      """ New (2024 Feb) log format which has function completion entries with CtrResources (CPU, Disk, Net, Mem) """
      
      INVOKE_TARGET_TRAIT = "iluvatar_worker_library::services::invocation::invoker_trait"
      INVOKE_TARGET_TRAIT_DIS = "iluvatar_worker_library::services::invocation::queueing_dispatcher"

      INVOKE_TARGET = "iluvatar_worker_library::services::containers::containerd::containerdstructs"
      INVOKE_NAME = "ContainerdContainer::invoke"  

      INVOKE_TARGET_S = "iluvatar_worker_library::services::containers::simulation::simstructs"
      INVOKE_NAME_S = "SimulatorContainer::invoke"  

      running = {}
      worker_log = os.path.join(self.log_loc, self.worker_log)
      if not os.path.exists(worker_log):
        print(f"Warning: Worker logs not available: {worker_log}")
        return

      
      def timestamp_to_pddate( time ):
        try:
            return pd.to_datetime( time )
        except:
            ptime = pd.to_datetime( time, format="%Y-%m-%d %H:%M:%S:%f+" )
            ptime = ptime.replace(tzinfo=tz.tzutc())
            local = ptime.astimezone(tz.tzlocal())
            local = local.replace(tzinfo=None)
            return local
        
      def timestamp(log):
        time = log["timestamp"]
        return timestamp_to_pddate( time )


      worker_df_list = []
      worker_status_list = []
      sca_pid_to_func_dict = {}
      tid = 0
      
      with open(worker_log, 'r') as f:
        while True:
          line = f.readline()
          if line == "":
            break
          try:
              log = json.loads(line)
          except Exception as e:
              print(e)
              print(line)
              continue 

          if log["fields"]["message"] == "tag_pid_mapping":
              self.sca_pid_to_func_dict_add( sca_pid_to_func_dict, log["fields"] )

          if log["fields"]["message"] == "new" or log["fields"]["message"] == "close":
              target = log["target"] == INVOKE_TARGET or log["target"] == INVOKE_TARGET_S
              span = log["span"]["name"] == INVOKE_NAME or log["span"]["name"] == INVOKE_NAME_S

    #{"timestamp":"2024-02-21 10:29:31.171202561","level":"INFO","fields":{"message":"Function completion","fname":"chameleon.CPU.2-2","exec":0.017911,"network":6554.0,"char":"warm","time":0.017911,"disk":0.0,"cpu":0.0,"mem":41762816.0},"target":"iluvatar_worker_library::services::invocation"}
              
          if log["target"] == "iluvatar_worker_library::services::invocation" and log["fields"]["message"] == "Function completion":
            if self.first_invoke_t is None:
                self.first_invoke_t = timestamp(log)
              
            fqdn = log["fields"]["fname"]
            stop_t = timestamp(log)
            exec_t = float(log["fields"]["exec"]) #seconds 
            start_t = stop_t + pd.Timedelta(seconds=0.0-exec_t)
            resource_dict = log["fields"] #already a dict? 
            del resource_dict['message']
            del resource_dict['fname']            

            resource_dict['fn_start'] = start_t
            resource_dict['fn_end'] = stop_t
            resource_dict['fqdn'] = fqdn
            tid +=1 
            resource_dict['tid'] = tid # Dont care about it 
            
            worker_df_list.append(resource_dict)

            
          if False and log["fields"]["message"] == "current load status":
            ts = timestamp(log)
            status = json.loads(log["fields"]["status"])
            mem_pct = (float(status["used_mem"]) / float(status["total_mem"])) * 100
            if status["cpu_sy"] == None:
                continue
            sys_cpu_pct = status["cpu_us"] + status["cpu_sy"] + status["cpu_wa"]
            load_avg = status["load_avg_1minute"] / status["num_system_cores"]
            if "queue_len" not in status:
                status["queue_len"] = status["cpu_queue_len"]
            gpu_queue_len = 0
            if "gpu_queue_len" in status:
                gpu_queue_len = status["gpu_queue_len"]
            data = (ts, status["queue_len"], mem_pct, sys_cpu_pct, load_avg, status["num_running_funcs"], status["num_containers"], status["cpu_us"], status["cpu_sy"],
                    status["cpu_wa"], gpu_queue_len)
            worker_status_list.append(data)

            
      worker_df = pd.DataFrame(worker_df_list) #, columns=['fn_start','fn_end','tid','fqdn']) ## XXX : Resource columns 
      
      if self.should_collapse_similar_funcs: 
          worker_df = self.collapse_similar_funcs( worker_df )

      self.status_df = pd.DataFrame.from_records(worker_status_list, columns=["timestamp", "queue_len", "mem_pct", "sys_cpu_pct", "load_avg", "num_running_funcs",
                                                                              "num_containers", "cpu_us", "cpu_sy", "cpu_wa", "gpu_queue_len"], index=["timestamp"])
      if self.collapse_fqdns:
          nset = {}
          fnlist = list(set(worker_df['fqdn']))
          for f in fnlist:
            r = re.search('(.*)\..[0-9]*\..*', f)
            if r is not None and len(r[0]) > 1:
                nset[f] = r[1]
            
          fnlist = []
          # update worker_df to have fqdns according to this dictionary 
          for k,v in nset.items():
            worker_df[ worker_df['fqdn'] == k ]['fqdn'] = v
            fnlist.append(v)
          
          self.fndict = nset 
          self.fnlist = sorted(list( set(fnlist) ))
          self.princip_list = list( set(fnlist) )
      else:
          self.fnlist = sorted(list(set(worker_df['fqdn'])))
          self.princip_list = list(set(worker_df['fqdn']))

      #P-XXX: Move or remove? Hard-coded fn names 
      if self.order_preference != None:
          new_order = []
          for f in self.order_preference:
              plist = [ function_name_to_paper( fn ) for fn in self.princip_list ]
              if f in plist:
                  new_order.append( self.princip_list.pop( plist.index(f) ) )

          new_order.extend( self.princip_list )
          self.princip_list = new_order 
          self.fnlist = new_order

      self.M = len(self.fnlist)
      
      if self.full_principals:
          self.M = self.M + len(self.extra_principals)
          self.princip_list = self.princip_list + self.extra_principals
      
      if quirks['quirk_trim_end']:
          worker_df = quirk_workerdf_trim_end( worker_df )
      if quirks['quirk_trim_end_by_time']:
          worker_df = quirk_workerdf_trim_end_by_time( worker_df, quirks_data['quirk_trim_end_by_time'] )
      self.worker_df = worker_df
      self.sca_pid_to_func_dict = sca_pid_to_func_dict
      pids = []
      funcs = []
      for k,v in self.sca_pid_to_func_dict.items():
          pids.append( k )
          funcs.append( v )
      self.sca_pid_to_func_df = pd.DataFrame( funcs, index=pids )

    ##################################################

    
    # Not sure what this is doing here. Docker expts can be separate?! 
    def docker_log_to_df(self):
      docker_log = os.path.join(self.log_loc, self.docker_log)
      if not os.path.exists(docker_log):
        return

      monitor_log = os.path.join(self.log_loc, "energy_monitor.log")
      with open(monitor_log) as f:
        log = json.loads(f.readline())
        actual_t = pd.to_datetime(log["timestamp"]).to_pydatetime(False)

      def timestamp(ns_stamp):
        seconds, nanoseconds = ns_stamp.split(".")
        nanoseconds = int(nanoseconds)
        nanoseconds += (int(seconds) * pow(10, 9))
        t = pd.to_datetime(nanoseconds, unit='ns', utc=True).to_pydatetime(False).replace(tzinfo=None)
        diff = t.hour - actual_t.hour
        return t - timedelta(hours=diff)

      worker_df_list = []
      start_t, stop_t = 0, 0
      tid=0
      with open(docker_log, 'r') as f:
        while True:
          line = f.readline()
          if line == "":
            break
          if line.startswith("Prewarming container"):
            start_t = timestamp(line.split()[-1])
            if self.first_invoke is None:
              self.first_invoke = start_t
          if line.startswith("killing container"):
            # invocation concluded
            fqdn = os.path.split(self.log_loc.strip("/"))[-1]
            stop_t = timestamp(line.split()[-1])
            worker_df_list.append((start_t, stop_t, tid, fqdn))
            tid += 1

      worker_df = pd.DataFrame(worker_df_list, columns=['fn_start','fn_end','tid','fqdn'])
      self.fnlist = list(set(worker_df['fqdn']))

      self.M = len(self.fnlist)
      self.power_df["cpu_pct_process"] = pd.Series([0]*len(self.power_df))
      # print(self.power_df.columns)
      cpu_log = os.path.join(self.log_loc, "cpu.log")
      with open(cpu_log, 'r') as f:
        lines = f.readlines()
        def parse_cpu_line(line):
          parts = line.split(" ")
          stamp = timestamp(parts[0])
          cpu = parts[-4] + parts[-5]
          return stamp, cpu
        records = map(parse_cpu_line, lines)
      self.cpu_df = pd.DataFrame.from_records(records, index=["timestamp"], columns=["timestamp","cpu_pct"]) #([1]*len(self.power_df), columns=["cpu_pct"], index=self.power_df.index)

      if self.full_principals:
          self.M = self.M + len(self.extra_principals) 

      # if self.worker_df is not None:
      #   raise Exception("Cannot parse both a docker log and worker log, please remove one.")
      self.worker_df = worker_df
    ##################################################

    def ipmi_to_df(self):
        ipmi_log = os.path.join(self.log_loc, self.ipmi_log)
        try:
            idf = pd.read_csv(ipmi_log,parse_dates=['timestamp'], dtype={'ipmi':np.float64})
            print("Read: {}".format(ipmi_log))
            self.col_sys = 'ipmi'
        except FileNotFoundError:
            print(f"Warning: IPMI logs not available: {ipmi_log}")
            return 
        idf = idf.set_index('timestamp')  
        nk =['ipmi']
        for k in nk:
            idf[k] = pd.to_numeric(idf[k])
        
        self.Widle = self.get_idle( idf, 'ipmi' )

        self.ipmi_df = idf

    ##################################################
    
    def get_idle( self, idf, pow_col):
        if self.status_df is None:
            return 0.0
        nidf = self.normalize_time_df_to_first_invoke( idf.copy() )
        nidf = nidf[ nidf['second'] < 0 ]
        nidf = nidf.drop_duplicates( 'second' )
        return nidf[pow_col].mean()

    ##################################################

    def tegra_to_df(self):
        tegra_log = os.path.join(self.log_loc, self.tegra_log)
        try:
            idf = pd.read_csv(tegra_log, header=0)
            print("Read: {}".format(tegra_log))
            self.col_sys = 'tegra'
        except FileNotFoundError:
            print(f"Warning: Instek GPM logs not available: {tegra_log}")
            return
        nk = idf.columns[2:]
        for k in nk:
            idf[k] = pd.to_numeric(idf[k])
        
        idf['timestamp'] = pd.to_datetime( idf['timestamp'] ) 
        idf = idf.set_index('timestamp')
        idf = idf.tz_localize(None)
        
        total = 'total'
        idf[total] = 0
        for f in self.tegra_pcols:
            idf[f] = idf[f].astype('int')
            idf[f] = idf[f]/1000.0
            idf[total] = idf[total] + idf[f]

        col = total 
        idf[col] = idf[col].ewm( alpha=0.3 ).mean()

        cols = { 'total':'tegra' }
        idf = idf.rename( columns=cols )
        idf = idf.drop(columns=['Unnamed: 0'])

        self.Widle = self.get_idle( idf, 'tegra' )

        self.tegra_df = idf

    ##################################################

    def scaphandre_pivot_singular( self, df_pivoted ):
        tag_cns = 'consumption'
        sca_df = df_pivoted
        sca_df = sca_df.droplevel('exe', axis=1)
        sca_df = sca_df[['timestamp', 'consumption', 'pid', 'fqdn']]
        sca_df = sca_df.set_index('timestamp')
        sca_df[tag_cns] = sca_df[tag_cns]/1000000.0
        return sca_df

    def scaphandre_pivot_gunicorn( self, df ):
        tdf = df.loc[ df['exe'] == 'gunicorn' ]
        tdf = tdf.pivot( columns=['exe'] )
        
        gdf = tdf.droplevel('exe', axis=1)
        gdf = gdf.pivot(columns=['fqdn'])
        
        ####
        # Building function df with consumption, pid, fqdn and index of timestamps
        funcs_set = set()
        for d,f in gdf.columns:
            funcs_set.add( f )
        
        def build_func_df( func ):
            ndf = pd.concat( [gdf['timestamp'][func], gdf['consumption'][func], gdf['pid'][func]], axis=1 )
            ndf.columns = ['timestamp', 'consumption', 'pid']
            ndf = ndf.dropna()
            
            def check_duplicated( g ):
                if len(g) > 1:
                    display( g )
            # ndf.groupby('timestamp').apply( check_duplicated )
            # we add up on per function basis - so duplicate timestamps are being added here 
            # to make the subsequent analysis easier 
            ndf = ndf.groupby('timestamp').agg( np.sum )
            ndf['fqdn'] = func
            return ndf
        
        ndfs = []
        funcs = []
        
        for f in funcs_set:
            funcs.append( f )
            ndfs.append( build_func_df(f) )
        
        tag_cns = 'consumption'
        sdf = pd.concat( ndfs )
        sdf[tag_cns] = sdf[tag_cns]/1000000.0
        sdf = sdf.sort_index()
        return sdf


    def sanitize_scaphandre_log( self, raw_f, san_f ):
        # skip first two lines
        with open(raw_f, 'r') as rawf:
            with open(san_f, 'w') as sanf:
                count = 0
                for rline in rawf:
                    if count >= 2:
                        sanf.write( rline )
                    count += 1

    def scaphandre_genrawdf(self, san_f ):
        df = None

        with open(san_f, 'r') as file:
            condfs = []
            for rline in file:
                try:
                    data = json.loads( rline )
                except JSONDecodeError as e:
                    print( "Error: Scaphandre json line not parsed")
                    print( "msg: {}".format(e.msg))
                    print( "doc: {}".format(e.doc))
                    print( "pos: {}".format(e.pos))
                    print( "lineno: {}".format(e.lineno))
                    print( "colno: {}".format(e.colno))
                    exit(-1)

                def append_con( con ):
                    con[tag_t] = int( con[tag_t] )
                    if 'pid' in con:
                        con['pid'] = int( con['pid'] )
                    condf = pd.DataFrame( con, columns=con.keys(), index=[0] )
                    condfs.append( condf )

                con = data['host']
                append_con( con )

                consumers = data['consumers']
                for con in consumers:
                    append_con( con )

            df = pd.concat( condfs, ignore_index=True )
            return df

    def scaphandre_append_fqdns(self, df, fdict):
        not_mapped = {}
        mapped = {}
        def apply_to_rows( row ):
            row['fqdn'] = np.NaN
            if pd.isna(row['pid']): 
                return row
            pid = str(int(row['pid']))
            if pid in fdict:
                row['fqdn'] = fdict[ pid ]
                mapped[pid] = row['exe']
            else:
                not_mapped[pid] = row['exe'] 
            return row
        df = df.apply( apply_to_rows, axis=1 )
        gnotmapped = df.loc[ df['exe'] == 'gunicorn' ]
        gnotmapped = gnotmapped.loc[ gnotmapped['fqdn'].isna() ]
        if len(gnotmapped) != 0:
            print("Warning: some of gunicorn pids not mapped in scaphandre logs")
            print( gnotmapped )
        return df

    def scaphandre_to_df(self):
        raw_f = self.log_loc + '/scaphandre_data.json'
        san_f = self.log_loc + '/scaphandre_sanitized.json'

        try:
            self.sanitize_scaphandre_log( raw_f, san_f )
        except FileNotFoundError:
            print("Warning: {} not present".format( raw_f ))
            return

        rawdf = self.scaphandre_genrawdf( san_f )
        self.scaphandre_rawdf = rawdf
        # rawdf = self.scaphandre_rawdf

        cdf = self.scaphandre_append_fqdns( rawdf, self.sca_pid_to_func_dict )
        self.scaphandre_cdf = cdf
        # cdf = self.scaphandre_cdf

        # functions df
        self.scaphandre_funcs_df = self.scaphandre_pivot_gunicorn( cdf )

        # scaphandra df
        tag = 'scaphandre'
        tdf = cdf.loc[ cdf['exe'] == tag ]
        tdf = tdf.pivot( columns=['exe'] )
        tdf['fqdn'] = tag
        self.scaphandre_scaphandre_df = self.scaphandre_pivot_singular( tdf )

        # total df
        tdf = cdf.loc[ cdf['exe'].isna() ]
        tdf = tdf.pivot( columns=['exe'] )
        tdf['fqdn'] = 'total'
        self.scaphandre_total_df = self.scaphandre_pivot_singular( tdf )

    ##################################################

    def igpm_to_df(self):
        igpm_log = os.path.join(self.log_loc, self.igpm_log)
        try:
            idf = pd.read_csv(igpm_log, header=0)
            print("Read: {}".format(igpm_log))
            self.col_sys = 'igpm'
        except FileNotFoundError:
            print(f"Warning: Instek GPM logs not available: {igpm_log}")
            return
        nk = idf.columns[1:]
        for k in nk:
            idf[k] = pd.to_numeric(idf[k])
        
        idf['timestamp'] = pd.to_datetime( idf['timestamp'] ) 
        idf = idf.set_index('timestamp')
        idf = idf.tz_localize(None)
        
        col = 'power' 
        idf[col] = idf[col].ewm( alpha=0.3 ).mean()

        cols = { 'power':'igpm' }
        idf = idf.rename( columns=cols )
        
        self.Widle = self.get_idle( idf, 'igpm' )

        self.igpm_df = idf
 
    ##################################################

    def battery_to_df(self):
        bat_log = os.path.join(self.log_loc, self.battery_log)
        try:
            idf = pd.read_csv(bat_log)
            print("Read: {}".format(bat_log))
        except FileNotFoundError:
            print(f"Warning: Battery logs not available: {bat_log}")
            return
        idf.columns=['timestamp','bat_energy']
        nk =['bat_energy']
        for k in nk:
            idf[k] = pd.to_numeric(idf[k])
        
        idf['timestamp'] = pd.to_datetime( idf['timestamp'] ) 
        idf['bat_power'] = - idf['bat_energy'].diff()/(1000*idf['timestamp'].diff().dt.total_seconds())
        idf = pd.DataFrame.dropna( idf, axis=0 )
        idf = idf[idf.bat_power >= 0.001 ]
        idf = idf.set_index('timestamp')
        idf = idf.tz_localize(None)
        idf = idf.ewm(com=0.5).mean()
        cols = { 'bat_power':'bat' }
        idf = idf.rename( columns=cols )

        self.bat_df = idf
        
    ##################################################

    def rapl_to_df(self):
        rapl_log = os.path.join(self.log_loc, self.rapl_log)
        
        try:
            idf = pd.read_csv(rapl_log, parse_dates=['timestamp'])
            print("Read: {}".format(rapl_log))
        except FileNotFoundError:
            print(f"Warning: Instek GPM logs not available: {rapl_log}")
            return

        idf = idf.set_index('timestamp')

        df = self.normalize_time_df_to_earliest_time( idf )
        rapl_df = pd.DataFrame( df.drop_duplicates('second') )

        rapl_df['rapl_pw'] = rapl_df.loc[:,'rapl_uj'].diff()/1000000.0
        rapl_df = rapl_df[ rapl_df['minute'] > 0 ]
        rapl_df = rapl_df[ rapl_df['second'] > 0 ]
        rapl_df = rapl_df[ rapl_df['rapl_pw'] > 0 ]
        
        self.rapl_df = rapl_df

    ##################################################

    def perf_to_df(self):
        perf_log = os.path.join(self.log_loc, self.perf_log)
        try:
            idf = pd.read_csv(perf_log, parse_dates=['timestamp'], dtype={'energy_pkg':np.float64, 'energy_ram': np.float64, 'retired_instructions':np.float64})
            print("Read: {}".format(perf_log))
        except FileNotFoundError:
            print(f"Warning: perf logs not available: {perf_log}")
            return

        idf = idf.set_index('timestamp')
        nk =['energy_pkg','retired_instructions', 'energy_ram']
        for k in nk:
            idf[k] = pd.to_numeric(idf[k])

        self.perf_df = idf

    ##################################################
    
    def process_hwpc_csv(self):
        basep = self.log_loc + '/sensor_output/'

        def try_read( path ):
            try:
                f = pd.read_csv( path )
                print("Read: {}".format(path))
                return f
            except FileNotFoundError:
                print('Warning: {} not found'.format(path))
                return None

        self.sw_core = try_read( basep + 'core.csv' )
        self.sw_global = try_read( basep + 'global.csv' )
        self.perf_counters = try_read( basep + 'perf_counters.csv' )

    ##################################################

    def cpu_to_df(self):
        cpu_log = os.path.join(self.log_loc, self.cpu_log)
        if not os.path.exists(cpu_log):
          print(f"Warning: Worker logs not available: {cpu_log}")
          return
        try:
            idf = pd.read_csv(cpu_log, parse_dates=['timestamp'])
        except EmptyDataError:
            print(f"Warning: Worker logs is empty: {cpu_log}")
            return
        print("Read: {}".format(cpu_log))
        idf = idf.set_index('timestamp')
        idf = idf.astype(float)
        self.cpu_df = idf

    ##################################################

    def mk_power_df(self):
        """ Combine all power readings into one with interpolation etc """
        # TODO: We may want to do a similar thing for the CPU readings as well? 
        if self.perf_df is not None:
            perf_rapl = self.perf_df[['energy_pkg', 'energy_ram', 'retired_instructions']]
        else:
            perf_rapl = pd.DataFrame()

        if self.ipmi_df is not None:
          # power_df = self.ipmi_df.join(perf_rapl, how='outer')
          power_df = mergedfs( perf_rapl, self.ipmi_df )
          power_df = power_df.rename(columns={"ipmi":"ipmi", "energy_pkg":"perf_rapl"})
          self.mc_type = 'server'
          self.power_col = 'ipmi'
        else:
          if isinstance(perf_rapl, pd.Series):
              power_df = perf_rapl.to_frame() 
          else:
              power_df = perf_rapl 
          power_df = power_df.rename(columns={"energy_pkg":"perf_rapl"})

        if self.bat_df is not None:
          power_df = power_df.join(self.bat_df, how='outer' )
          self.mc_type = 'laptop'
          self.power_col = 'bat'

        if self.igpm_df is not None:
          if len(power_df) == 0:
              power_df = power_df.join(self.igpm_df, how='outer' )
          else:
              power_df = mergedfs( power_df, self.igpm_df )
          self.mc_type = 'desktop'
          self.power_col = 'igpm'

        if self.tegra_df is not None:
          # power_df = power_df.join( self.tegra_df, how='outer' )
          if len(power_df) == 0:
              power_df = self.tegra_df[self.tegra_pcols+['tegra']]
          else:
              power_df = mergedfs( self.tegra_df[self.tegra_pcols+['tegra']], power_df )
          self.mc_type = 'jetson'
          self.power_col = 'tegra'

        if self.rapl_df is not None:
          # power_df = power_df.join(self.rapl_df, how='outer' )
          power_df = mergedfs( power_df, self.rapl_df )

        # PowerDF should be just power. rest should be elsewhere!   
        if self.cpu_df is not None:
          # power_df = self.cpu_df.join(power_df, how='outer')
          power_df = mergedfs( power_df, self.cpu_df )

        if self.proc_df is not None:
          # power_df = self.proc_df.join(power_df, how='outer', lsuffix='_process')
          proc_df = self.proc_df.copy()
          cols = { c:c+'_process' for c in self.proc_df.columns }
          proc_df = proc_df.rename( columns=cols )
          power_df = mergedfs( power_df, proc_df )

        interp_cols = ["perf_rapl", "energy_ram", "retired_instructions", "load_avg_1minute", "ipmi", "cpu_pct_process", "igpm", "bat_power", "energy_pkg", "rapl_pw"]
        interp_cols = [col for col in interp_cols if col in power_df.columns]
        power_df[interp_cols] = power_df.loc[:, interp_cols].interpolate(limit_direction='both')

        fil_cols = ["cpu_time"]
        fil_cols = [col for col in fil_cols if col in power_df.columns]
        power_df[fil_cols] = power_df.loc[:, fil_cols].fillna( method="bfill")
        power_df[fil_cols] = power_df.loc[:, fil_cols].fillna( method="ffill")
        
        if "cpu_pct_process" not in power_df.columns:
            power_df["cpu_pct_process"] = pd.Series([0]*len(power_df))
        self.power_df = power_df

    ##################################################

    def process_df(self):
        proc_log = os.path.join(self.log_loc, self.proc_log)
        try:
            idf = pd.read_csv(proc_log, parse_dates=['timestamp'])
            print("Read: {}".format(proc_log))
        except FileNotFoundError:
            print(f"Warning: Process logs not available: {proc_log}")
            return 
        idf = idf.set_index('timestamp')
        self.proc_df = idf

    ##################################################

    def normalize_time_df_to_earliest_time( self, df):
      if self.proc_df is not None: 
          earliest_time = min([self.status_df.index[0], self.proc_df.index[0]])
      else:
          earliest_time = self.status_df.index[0]
      self.first_invoke = (self.first_invoke_t - earliest_time).total_seconds() / 60

      df["minute"] = df.index.to_series().apply(lambda x: (x-earliest_time).total_seconds() / 60)
      df["second"] = df.index.to_series().apply(lambda x: math.floor((x-earliest_time).total_seconds()))

      return df

    def normalize_time_df_to_first_invoke( self, df):
      if self.proc_df is not None: 
          earliest_time = min([self.status_df.index[0], self.proc_df.index[0]])
      else:
          earliest_time = self.status_df.index[0]
      self.first_invoke = (self.first_invoke_t - earliest_time).total_seconds() / 60

      df["minute"] = df.index.to_series().apply(lambda x: (x-self.first_invoke_t).total_seconds() / 60)
      df["second"] = df.index.to_series().apply(lambda x: math.floor((x-self.first_invoke_t).total_seconds()))

      return df

    def normalize_times(self):
      earliest_time = min([self.status_df.index[0], self.proc_df.index[0]])
      self.first_invoke = (self.first_invoke_t - earliest_time).total_seconds() / 60

      self.status_df["minute"] = self.status_df.index.to_series().apply(lambda x: (x-earliest_time).total_seconds() / 60)
      self.status_df["second"] = self.status_df.index.to_series().apply(lambda x: math.floor((x-earliest_time).total_seconds()))

      self.proc_df["minute"] = self.proc_df.index.to_series().apply(lambda x: (x-earliest_time).total_seconds() / 60)
      self.proc_df["second"] = self.proc_df.index.to_series().apply(lambda x: math.floor((x-earliest_time).total_seconds()))

      self.power_df["minute"] = self.power_df.index.to_series().apply(lambda x: (x-earliest_time).total_seconds() / 60)
      self.power_df["second"] = self.power_df.index.to_series().apply(lambda x: math.floor((x-earliest_time).total_seconds()))

      def drop_negatives( df ):
          df = df[ df['minute'] > 0 ]
          df = df[ df['second'] > 0 ]
          return df

      self.power_df = drop_negatives( self.power_df )
      self.status_df = drop_negatives( self.status_df )
      self.proc_df = drop_negatives( self.proc_df )

    def append_x_rest( self, power_df=None,  col_cpu='perf_rapl' ):
        if power_df is None:
            power_df = self.sw_estimate
        col_sys = self.col_sys
        rest = 'x_rest'
        power_df[rest] = power_df[col_sys] - power_df[col_cpu]
        power_df.loc[ power_df[rest] < 0 ] = 0.0
        return power_df

    def scale_by_basefreq( self, pdf, cols ):
        core = pdf.copy()
        tsc_nominal = self.cpu_config['cpu_freq_base']*1000000.0  # base frequency in Mhz
        for f in cols:
            if f in core.columns:
                core[f] = core[f] / tsc_nominal
        return core

    def aggregate_sw_data_given_timestamp_target( self, pdf ):
        cum = pdf.groupby(['timestamp','target']).aggregate( sum )
        cum = cum.reset_index()
        return cum

    ##################################################

    def merge_power_df_to_hwpc( self ):

        t  = 'timestamp'
        tg = 'target'
        f = 'fqdn'
        contr = 'Contr'
        cfeatures =  self.cpu_config['cfeatures']
        cfeatures = cfeatures[:-1]
        p_col = 'perf_rapl'

        worker_df = self.worker_df

        ca = self.aggregate_sw_data_given_timestamp_target(  self.sw_core )
        cags = self.scale_by_basefreq(ca, cfeatures)
        cags = cags.sort_values('timestamp')
        cagst = cags.groupby(t).agg(np.sum)

        self.sw_core_aggr_time_scaled = cagst

        swgs = self.scale_by_basefreq(self.sw_global, cfeatures)
        swgs = swgs.groupby(t).agg(np.sum)
        swgs = swgs.reset_index()
        swgs = swgs[ [t] + cfeatures ]
        self.sw_global_scaled_agg = swgs

        sw_swgs_cmb = pd.merge_asof( cagst,  \
                                swgs,  \
                                on='timestamp',  \
                                direction='nearest',  \
                                tolerance=300 )
        for i in range(0,len(cfeatures)):
            c = cfeatures[i]
            sw_swgs_cmb[c] = sw_swgs_cmb[c+'_y'] - sw_swgs_cmb[c+'_x']

        sw_swgs_f = sw_swgs_cmb[[t]+cfeatures].copy()
        sw_swgs_f[tg] = 'os_cp'

        sw_cc = pd.concat( [cags,sw_swgs_f] )
        sw_cc = sw_cc.sort_values(t)
        sw_cc = sw_cc.interpolate() # now sw_cc is a super set of cags

        if self.output_type == 'indiv':
            pl = self.princip_list[:-1]
        else:
            pl = self.princip_list

        ##
        # Activation Calc

        wdf = get_givenpattern( worker_df, f, pl[0] )
        wdf = worker_df

        fs = 'fn_start'
        fe = 'fn_end'

        fink = wdf.iloc[0][fs]
        eink = wdf.iloc[-1][fe]
        total_sec = int((eink-fink).total_seconds())
        ts = (list(range(0,total_sec)))
        ts = [ ((pd.Timedelta(tse,unit='sec') + fink)) for tse in ts ]
        ts = np.array(ts)
        
        if self.output_type == 'indiv':
            pl = self.princip_list[:-1]
        else:
            pl = self.princip_list

        # A = self.core.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='C' )
        A = self.core.build_A_matrix( ts=fink, N=total_sec, delta=1, ac_type='A' )

        sw_cc_ags = []

        for i,p in enumerate(pl):
            sw_cc_t = get_givenpattern( sw_cc, tg, p )
            sw_cc_t = sw_cc_t.groupby(t).agg(np.sum)
            sw_cc_t = sw_cc_t.reset_index()
            sw_cc_t[tg] = p

            ap = A[:,i]

            apd = pd.DataFrame()
            apd[t] = ts
            apd[contr] = ap

            apd = self.timestamp_datetime_to_ts( apd )

            sw_cc_t = pd.merge_asof( sw_cc_t,  \
                                apd,  \
                                on='timestamp',  \
                                direction='nearest',  \
                                tolerance=500 )

            sw_cc_t = sw_cc_t.interpolate()

            sw_cc_ags.append( sw_cc_t )

        sw_cc_t = get_givenpattern( sw_cc, tg, 'os_cp' ).copy()
        sw_cc_t[contr] = 0.0
        sw_cc_ags.append( sw_cc_t )

        sw_cc_ags = pd.concat( sw_cc_ags, axis=0 )
        sw_cc = sw_cc_ags.sort_values(t)

        pdf = self.power_df
        # self.set_widle( pdf, p_col)
        # widle = self.Widle
        # pdf[p_col] = pdf[p_col] - widle

        pow_dft = self.timestamp_datetime_to_ts(  pdf.reset_index() )
        pow_dft = pow_dft.sort_values('timestamp')
        cmb_t = pd.merge_asof(  sw_cc,  \
                                pow_dft,  \
                                on='timestamp',  \
                                direction='nearest',  \
                                tolerance=600 )
        cmb_t = cmb_t.interpolate()
        # cmb_t = cmb_t.dropna()
        f = self.cpu_config['cfeatures']
        cmb_t = cmb_t[cmb_t[f[-1]].notna()]
        self.sw_combined_t = cmb_t.copy()
        cmb_t['timestamp'] = self.timestamp_to_datetime( cmb_t )
        self.combined_hwpc = cmb_t
        return cmb_t

    ##################################################

    def process_all_logs(self):
        """ Top-level function """ 
        self.worker_log_to_df()
        self.perf_to_df()
        self.cpu_to_df()

        self.process_df()

        self.igpm_to_df()
        self.tegra_to_df()
        self.ipmi_to_df()
        self.rapl_to_df()
        self.battery_to_df()
        self.scaphandre_to_df()

        self.mk_power_df()
        self.docker_log_to_df()
        
        self.power_df = self.power_df.set_index('timestamp')
        if quirks['quirk_trim_end'] or quirks['quirk_trim_end_by_time']:
            self.power_df = quirk_df_by_index_trim_end_to_wdf( self.worker_df, self.power_df )

        pr = 'perf_rapl'
        if pr in self.power_df.columns:
            self.Widle_cpu = self.get_idle( self.power_df, pr )

            self.append_x_rest( power_df=self.power_df )

            # both of things below are mathematically same
            # self.Widle_xrest = self.get_idle( self.power_df, 'x_rest' )
            if self.mc_type == 'server':
                self.Widle_xrest = self.Widle - self.Widle_cpu + ipmi_correction
            else:
                self.Widle_xrest = self.Widle - self.Widle_cpu

        self.process_hwpc_csv()
        if self.sw_core is not None: 
            self.merge_power_df_to_hwpc()
            if quirks['quirk_trim_end'] or quirks['quirk_trim_end_by_time']:
                self.combined_hwpc = quirk_df_by_tag_trim_end_to_wdf( self.worker_df, self.combined_hwpc, 'timestamp' )
       
        self.power_df_secs = condense_to_seconds( self.power_df )

        nl = '\n'
        msg = "Generated dfs:"+nl
        if self.worker_df is not None: msg += '  worker_df'+nl
        if self.process_cpu_df is not None: msg += '  process_cpu_df'+nl
        if self.ipmi_df is not None: msg += '  ipmi_df'+nl
        if self.bat_df is not None: msg += '  bat_df'+nl
        if self.igpm_df is not None: msg += '  igpm_df'+nl
        if self.tegra_df is not None: msg += '  tegra_df'+nl
        if self.rapl_df is not None: msg += '  rapl_df'+nl
        if self.cpu_df is not None: msg += '  cpu_df'+nl
        if self.perf_df is not None: msg += '  perf_df' +nl
        if self.power_df is not None: msg += '  power_df'+nl
        if self.power_df_secs is not None: msg += '  power_df_secs'+nl
        if self.proc_df is not None: msg += '  proc_df'+nl
        if self.status_df is not None: msg += '  status_df'+nl
        if self.sw_core is not None: msg += '  sw_core'+nl
        if self.combined_hwpc is not None: msg += '  combined_hwpc'+nl
        #P-Suggest: Print msgs should be logged to control verbosity etc. 
        print(msg)

    def save_dfs(self):
        base = self.log_loc + dir_lg2df + '/' 
        if self.worker_df is not None: save_df(self.worker_df, base + 'worker_df') 
        if self.process_cpu_df is not None: save_df(self.process_cpu_df, base + 'process_cpu_df') 
        if self.ipmi_df is not None: save_df(self.ipmi_df, base + 'ipmi_df') 
        if self.bat_df is not None: save_df(self.bat_df, base + 'bat_df') 
        if self.igpm_df is not None: save_df(self.igpm_df, base + 'igpm_df') 
        if self.tegra_df is not None: save_df(self.tegra_df, base + 'tegra_df') 
        if self.rapl_df is not None: save_df(self.rapl_df, base + 'rapl_df') 
        if self.cpu_df is not None: save_df(self.cpu_df, base + 'cpu_df') 
        if self.perf_df is not None: save_df(self.perf_df, base + 'perf_df') 
        if self.power_df is not None: save_df(self.power_df, base + 'power_df') 
        if self.power_df_secs is not None: save_df(self.power_df_secs, base + 'power_df_secs') 
        if self.proc_df is not None: save_df(self.proc_df, base + 'proc_df') 
        if self.status_df is not None: save_df(self.status_df, base + 'status_df') 
        if self.sw_core is not None: save_df(self.sw_core, base + 'sw_core') 
        if self.combined_hwpc is not None: save_df(self.combined_hwpc, base + 'combined_hwpc') 
        if self.Widle is not None: save_df(pd.DataFrame([[self.Widle]]), base + 'Widle') 
        if self.Widle_cpu is not None: save_df(pd.DataFrame([[self.Widle_cpu]]), base + 'Widle_cpu') 
        if self.Widle_xrest is not None: save_df(pd.DataFrame([[self.Widle_xrest]]), base + 'Widle_xrest') 
        if self.scaphandre_rawdf is not None: save_df(self.scaphandre_rawdf, base + 'scaphandre_rawdf')
        if self.scaphandre_cdf is not None: save_df(self.scaphandre_cdf, base + 'scaphandre_cdf')
        if self.scaphandre_funcs_df is not None: save_df(self.scaphandre_funcs_df, base + 'scaphandre_funcs_df')
        if self.scaphandre_scaphandre_df is not None: save_df(self.scaphandre_scaphandre_df, base + 'scaphandre_scaphandre_df')
        if self.scaphandre_total_df is not None: save_df(self.scaphandre_total_df, base + 'scaphandre_total_df')
        if self.sca_pid_to_func_df is not None: save_df(self.sca_pid_to_func_df, base + 'sca_pid_to_func_df')

    ##################################################
    #################### Testing ####################

    def init_estimates(self, N_init = -1, delta = 1):
        ts = self.worker_df['fn_start'][0]
        te = self.worker_df.iloc[-1]['fn_end']
        
        if N_init == -1: # perform on the whole trace
            N_init = (te-ts).total_seconds()/delta
            N_init = int( N_init )

        A, W = self.build_A_W(ts, N_init, delta)
        X = self.pow_cvx(A, W)
        return X 
      ##################################################

    def build_A_W(self, ts, N, delta):
        """ Main interface for getting the A,W matrices """
        W = self.build_J_matrix(ts, N, delta) 
        A = self.build_A_matrix(ts, N, delta, ac_type=self.a_type)
        return (A, W)
    
    def build_J_matrix(self, ts, N, delta):
        """ Energy for N time steps """ 
        W = [] 
        for i in range(N):
            t = ts + pd.Timedelta(seconds=i)
            J_avg, all_matched_entries = self.J_at(t, delta, self.power_col, lag=0)
            W.append(J_avg)

        raw_W = np.array(W)
        # TODO: get the signal lag and input this here?
        W = raw_W
        
        if self.output_type != "full":
            # remove idle
            W = raw_W - (delta*self.Widle)
                        
        return W   
    def build_A_matrix(self, ts, N, delta, ac_type="C"):
        M = self.M 
        A = np.zeros((N,M))

        for i in range(N):
            t = ts+pd.Timedelta(seconds=i) #need to check that it exists?
            if ac_type == "A":
                A[i] = self.get_A_row(t, delta)
            elif ac_type == "C":
                A[i] = self.get_C_row(t, delta)

        return A

    def get_A_row(self, t, delta):
        """ Get the functions active from t-delta to t """
        #Vector of size M, number of unique functions
        worker_df = self.worker_df
        M = self.M
        
        ts = t + pd.Timedelta(seconds=-delta)
        te = t
        # This catches all functions that intersect the interval atleast once.
        # Even if fn_start > ts or fn_end < te. DeMorgan  
        # If fn len is more than delta, then function will be in multiple rows. OK.
        # Negations everywhere because of pandas not supporting >=? 
        rfns = worker_df[~(worker_df['fn_end'] < ts) & ~(worker_df['fn_start'] > te)]

        row = np.zeros(M)

        for fn in rfns['fqdn']:
            i = self.get_fn_index(fn)
            row[i]=row[i]+1 

        if self.output_type != "full":
            # 2 extra columns have been added, we need to set them to some value. 
            c = self.c_s_contribution(t, delta, "A")
            row[M-1] = c 
            #row[M-1] = s 
            
        return row 
    def c_s_contribution(self, t, delta, kind="A"):
        """ Control plane and system contribution """ 
        if kind == "A":
            # ctrl plane and os are always running 
            return 1
        
        elif kind == "C":
            cplane_cpu_pct, _ = self.vdi(t, delta, self.power_df, "cpu_pct_process")
            # This is the system-wide CPU %.  We need to convert it to time, which is delta seconds
            cp_time = cplane_cpu_pct * 0.01 *float(delta) * self.cpu_cores
            # normalize this by system-cpu
            sys_cpu, _  = self.vdi(t, delta, self.cpu_df, "cpu_pct")
            if self.fix_sys_cpu:
                sys_cpu = max(sys_cpu, cplane_cpu_pct)*0.5 #atmost half 
            ### XXX above for the 0 sys-cpu case due to integer rounding. 
            if sys_cpu > 0:
                cp_time = cp_time/sys_cpu 
            
            sys_time = 0.0 # TODO: From OS sys/kern %.
            
            return cp_time
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
    def J_at(self, t, delta, col, lag=0):
        # average energy from t-delta to t. 
        # If delta<=1 second, return W[t]? TODO, not done yet yolo

        #Since the power signal lags, we need to find the power at a future point in time and add 
        if lag > 0:
            t = t + pd.Timedelta(seconds = lag)

        ts = t + pd.Timedelta(seconds=0-delta)

        df = self.power_df

        #match the rows 
        m = df.index[(df.index >= ts) & (df.index <= t)]
        sdf = df.loc[m]

        J_avg = np.mean(sdf[col])

        #possible for this to be nan if missing value.
        if np.isnan(J_avg):
            J_avg = -1.0 #filter out later?

        #Since we want energy, multiply by the delta seconds
        J_avg = J_avg * np.abs(delta)

        #May want to compute variance of power, samples, confidence intervals etc later
        return J_avg, sdf[col]

    def get_C_row(self, t, delta):
        """ Contributions: Running time for each function in this interval (seconds) """
        worker_df = self.worker_df
        M = self.M
        
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
            i = self.get_fn_index( fn )
            row[i] = row[i] + runtime

        if self.output_type != "full": 
            # 2 extra columns have been added, we need to set them to some value. 
            c = self.c_s_contribution(t, delta, "C")
            row[M-1] = c 
            #row[M-1] = s

        return row 
    
    def interval_intersection(self, a, b):
        """ a and b are tuples with start and end times. All explicit timestamps. seconds """
        astart, aend = a
        bstart, bend = b 
        instart = max(astart, bstart)
        inend = min(aend, bend)
        duration_seconds = self.tspan_sec(instart, inend)
        return duration_seconds 
    def tspan_sec(self, ts, te):
        # TODO resolution stuff?
        td = te - ts
        nanosec = float(td.to_numpy())
        ns = 1000000000.0
        return float(nanosec/ns)
    
    def pow_cvx(self, A, W):
        import cvxpy as cp
        # Power estimate using cvxpy https://www.cvxpy.org/examples/basic/least_squares.html
        M = self.M
        
        x = cp.Variable(M)
        cost = cp.sum_squares(A @ x - W)
        
        prob = cp.Problem(cp.Minimize(cost), [x >= self.min_fn_power])
        prob.solve()

        return x.value 
    
    def pow_full_breakdown(self, X, ts=-1, te=-1):
        
        if ts == -1:
            ts = self.worker_df['fn_start'][0]
        
        if te == -1:
            te = self.worker_df.iloc[-1]['fn_end']

        outd = dict()
        Xcorr, J, J_contrib, avg_fn_times   = self.fn_power_to_energ(X, ts, te) 
        outd["Power"], outd["Energy"], outd["Energy-contrib"], outd["Avg-fn-times"] = np.array(Xcorr), np.array(J), np.array(J_contrib), np.array(avg_fn_times)
        
        if self.output_type == 'indiv':
            outd["A"], outd["Energy"], outd["per_invok_cp"], outd["per_invok_idle"],  = self.pow_to_J_total(X, ts, te)
        
        outd["Principals"] = np.array(self.princip_list) 
        return outd 
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
        
        if self.full_principals:
            #The last element is the control-plane!
            cplane_cpu_pct, _ = self.vdi(te, delta, self.power_df, "cpu_pct_process")
            cp_pow = X[-1] * float(cplane_cpu_pct) * 0.01

            sys_cpu, _  = self.vdi(te, delta, self.cpu_df, "cpu_pct")
            if self.fix_sys_cpu:
                sys_cpu = max(sys_cpu, cplane_cpu_pct)*0.5 #atmost half 

            if sys_cpu > 0:
                cp_pow = cp_pow * 100.0 / sys_cpu
                
            Xcorr[-1] = cp_pow
            J[-1] = cp_pow * delta
            J_contrib[-1] = cp_pow * delta
            
        return Xcorr, J, J_contrib, avg_fn_times  
    
    def avg_fn_latencies_interval(self, t, delta):
        #avgs = self.get_C_row(t, delta)/self.get_A_row(t, delta)
        avgs = self.div_by_A(self.get_C_row(t, delta), self.get_A_row(t, delta))
        return np.nan_to_num(avgs)
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

    
    def pow_to_J_total(self, X, ts, te):
        """ Power to energy per invocation and components:
        J(per-invok) + J_cp/\sum(A) + J_idle/M.A_i """
        
        assert(self.output_type == "indiv")        
        delta = float(self.tspan_sec(ts, te))
        M = self.M # == len(X)
        Xcorr, J, J_contrib, avg_fn_times = self.fn_power_to_energ(X, ts, te)
        cp_power = Xcorr[-1] # This can go too high? 
        
        J_cp = cp_power * delta
        J_idle = self.Widle * delta

        A = np.array(self.get_A_row(te, delta)) # invok counts for each function
        # We already have control-plane ovhead, infact it should be 0!
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
    #Getters 

    def get_realpower_series(self):
        return self.power_df[self.power_col]
    
    def get_realpower_means(self):
        return self.power_col

    #P-Suggest: This should have a __name__ == "__main__" clause to run it separately for modularity, testing, etc. 
    
    ############################################################
    #######################  END    ############################
    ############################################################
