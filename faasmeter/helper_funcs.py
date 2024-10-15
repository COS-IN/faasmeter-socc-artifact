import pandas as pd
from faasmeter.faasmeter.tags import *
import pickle
import subprocess

def function_name_to_paper(name: str) -> str:
  """
  Tries to convert a functon name into one suitable for the paper
  It will raise an error if one could not be found
  """
  map_dict = {
                "hello":                           "hello",
                "gzip_compression":                "gzip",
                "pyaes":                           "AES",
                "pyaes_small":                     "AES_small",
                "dd":                              "dd",
                "lin_pack":                        "lin_pack",
                "float_operation":                 "float",
                "chameleon":                       "web",
                "json_dumps_loads":                "json",
                "cnn_image_classification":        "cnn",
                "cnn_image_classification_gpu":        "cnn_gpu",
                "image_processing":                "image",
                "video_processing":                "video",
                "model_training":                  "ml_train",
                "lookbusy":                        "bubble",
  }

  if name in map_dict:
    return map_dict[name]
  if "." in name:
    split_name = name.split(".")
    for part in split_name:
      if part in map_dict:
        return map_dict[part]
  if "-" in name:
    split_name = name.split("-")
    for part in split_name:
      if part in map_dict:
        return map_dict[part]
  #raise Exception(f"Could not convert function name '{name}' to a paper-appropriate one")
  return name

def get_field( x, kargs, obj ):
    if x not in kargs:
        return getattr(obj, x)
    return kargs[x]
def get_givencol( df, col, value ):
    return df.loc[ df[col] == value ]
def get_giventimestamp( df, time ):
    return df.loc[ df['timestamp'] == time ]
def get_givensocket( df, sock ):
    return df.loc[ df['socket'] == sock ]
def get_giventarget( df, target ):
    return df.loc[ df['target'] == target ]
def get_givenpattern( pd, column, pattern ):
    return pd[pd[column].str.contains(pat=pattern, regex=True)]
def get_col_names_btw( pd, start='cpu', end='time_enabled' ):
    features = list(pd.columns)
    s = features.index(start) + 1
    e = features.index(end)
    features = features[s:e]
    return features 

def fixtcol(p):
    t='timestamp'
    p[t]=p[t].astype('<M8[ns]')

def mergedfs( a, b, nearest_sec=1 ):
    sw_est = pd.merge_asof( a,  \
      b,  \
      left_on='timestamp', \
      right_on='timestamp',  \
      direction='nearest',  \
      tolerance=pd.Timedelta(nearest_sec,unit='sec') )
    return sw_est

def handle_outliers( ss: pd.Series ) -> pd.Series:
    Q1 = np.percentile(ss, 25, method='midpoint')
    Q3 = np.percentile(ss, 75, method='midpoint')
    IQR = Q3 - Q1

    # Above Upper bound
    upper=Q3+1.5*IQR
    upper_array=np.array(ss>=upper)
    ss.loc[ ss >= upper ] = upper

    #Below Lower bound
    lower=Q1-1.5*IQR
    lower_array=np.array(ss<=lower)
    ss.loc[ ss <= lower ] = lower
    return ss

def time_to_seconds( t ):
    return t.hour*60*60 + t.minute*60 + t.second

def time_to_mins( t ):
    return t.hour*60 + t.minute

def translate_t_to_mins(df, tag_t=tag_t):
    return df[tag_t].apply(time_to_mins)

def translate_t_to_seconds(df, tag_t=tag_t):
    return df[tag_t].apply(time_to_seconds)

def condense_to_seconds( pdf ):
    pdf = pdf.copy()
    tms = translate_t_to_seconds( pdf.reset_index() )
    pdf.index = tms
    pdf = pdf.groupby(tag_t).mean()
    pdf = pdf.reset_index()
    pdf = pdf.rename(columns={tag_t: 'seconds'})
    pdf = pdf.set_index('seconds')
    return pdf

def save_df(df, name):
    df.to_csv( name + '.csv' )
    df.to_pickle(  name + '.pickle' )

def load_pickle(f):
    with open(f, 'rb') as f:
        return pickle.load(f)

def dump_keys(dfs):
    for k in dfs.keys():
        print("{} ----".format(k))
        for k2 in dfs[k].keys():
            print("     {}".format(k2))

def reformat_exec_time_df( df ):
    df = df['exec_time']['mean']
    df = pd.DataFrame( df )
    df = df.transpose()
    # print( df )
    # exit(0)
    return df

def exec_cmd( cmd ):
    r = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                      )
    if r.returncode != 0:
        exit(r)
    return r.stdout.decode()[:-1]

