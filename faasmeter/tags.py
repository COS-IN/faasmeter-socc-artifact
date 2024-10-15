from faasmeter.faasmeter.config import *

tag_t = 'timestamp'
tag_tm = 'minutes'

tag_fqdn = 'fqdn' 
tag_targets = 'target' # function names in dfs same as fqdn

tag_p = 'consumption'

tag_fn_s = 'fn_start'
tag_fn_e = 'fn_end'
tag_fn_exec = 'exec_time'

tag_pcpu = 'perf_rapl'
tag_pshared = 'x_rest'

tag_psys_desktop = 'igpm'
tag_psys_server = 'ipmi'
tag_psys_jetson = 'tegra'
tags_psys = [ 
    tag_psys_desktop,
    tag_psys_jetson,
    tag_psys_server,
    tag_pcpu,
    tag_pshared,
]

tags_pmins = { 
    t: t + '_mins' for t in [tag_pcpu, tag_pshared] + tags_psys
}

tag_lg2df = 'lg2df'
tag_dis_cpu = 'cpu'
tag_dis_combined = 'combined'
tag_analysis = 'analysis'

tag_cns = 'consumption'
