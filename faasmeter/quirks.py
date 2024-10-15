import pandas as pd

def quirk_workerdf_trim_end( wdfo ):
    def select_last( group ):
        return group.loc[group['fn_end'].idxmax()]
    wdf = wdfo.groupby('fqdn').apply(select_last)
    last = wdf.loc[wdf['fn_end'].idxmin()]
    return wdfo.loc[ wdfo['fn_end'] < last['fn_end'] ]

def quirk_workerdf_trim_end_by_time( wdfo, time_mins ):
    time_mins = float(time_mins)
    last = wdfo.loc[wdfo['fn_end'].idxmax()]
    return wdfo.loc[ wdfo['fn_end'] <= last['fn_end'] - pd.Timedelta( minutes=time_mins ) ]

def quirk_df_by_index_trim_end_to_wdf( wdfo, pdf ):
    last = wdfo.loc[wdfo['fn_end'].idxmax()]
    return pdf.loc[ pdf.index <= last['fn_end'] ]

def quirk_df_by_tag_trim_end_to_wdf( wdfo, pdf, tag ):
    last = wdfo.loc[wdfo['fn_end'].idxmax()]
    return pdf.loc[ pdf[tag] < last['fn_end'] ]

