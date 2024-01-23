import os
import itertools
import pandas as pd
import urllib
import numpy as np
from sklearn.metrics import pairwise_distances, roc_auc_score
import time
import math
from numba import jit
from sklearn.metrics import roc_auc_score

#Download drug-target interaction flat file from DrugCentral
df_dti = pd.read_csv('drug.target.interaction.tsv.gz', compression='gzip', sep='\t')
#df_dti.to_csv('drug.target.interaction.csv', sep='\t', index=False)
df_dti.to_excel('drug.target.interaction.xlsx', index=False)