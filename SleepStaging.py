
# coding: utf-8

# In[2]:

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#get_ipython().magic('matplotlib inline')
#plt.rcParams['figure.figsize'] = (15, 9)


# In[3]:

import glob
import os
file_names = glob.glob('C:/Users/Ostyk/Desktop/hackaton_data/*.h5')
dataframes = {
    os.path.basename(fname) : pd.HDFStore(fname)
    for fname in file_names
    if not fname.endswith('KD_050616.h5')
}


# In[4]:

def get_timeseries_by_epoch(df, field='signal', common_label=None):
    grouped = df.groupby('epoch')
    tmp = grouped.aggregate(lambda x: tuple(x))[[field]]
    if common_label:
        tmp['label'] = common_label
    return tmp

def get_labels_by_epoch(df, common_label=None):
    tmp = df.loc[:,['epoch','stage']]
    if common_label:
        tmp['label'] = common_label
    return tmp

def get_X_y(dfs):
    X = pd.concat([
        get_timeseries_by_epoch(dfs[df_name]['/eeg_neuroon_cleaned'], 'signal', df_name)
        for df_name in dfs
    ])
    y = pd.concat([
        get_labels_by_epoch(dfs[df_name]['/reference_alice'], df_name)
        for df_name in dfs
    ])
    X.set_index(['label'], inplace=True, append=True)
    y.set_index(['epoch', 'label'], inplace=True)
    result = pd.concat([X, y], axis=1, join_axes=[X.index])
    return result


# In[5]:

data = get_X_y(dataframes)

data1=data.copy()
# In[6]:

def has_no_nans(x):
    return not np.isnan(x).any()

data = data[data['signal'].map(has_no_nans)]


# In[7]:

data.head()


# In[8]:

import scipy.signal, scipy.fftpack
x = np.array(data.iloc[1]['signal'])
plt.plot(x)
plt.xlabel('Tick, 1/125 s')
plt.ylabel('Signal, units')
plt.title('Stage: {}'.format(data.iloc[100]['stage']))


# In[9]:

from scipy.signal import welch, filtfilt, butter

def welch_func(x, Fs):
    N = len(x)
    Nseg = 3
    N_s = N/Nseg
    our_window = np.hamming(N_s)
    our_window /= np.linalg.norm(our_window)
    (F, P) = welch(x, Fs, window=our_window)
    return F, P

def filtering(signal,Fs=125):
    Nq=Fs/2
    [b,a] = butter(N=3, Wn=[0.5/Nq, 25./Nq], btype='bandpass')
    return filtfilt(b,a,signal)

def get_power(x, low, high, fs=125):
    F2, P2 = welch_func(x, fs)
    mask = (F2>=low)&(F2<=high)
    power = np.trapz(P2[mask], x=F2[mask])
    return power

freqs = [
    ('Alpha_power', 8, 13),
    ('Beta_power', 16, 31),
    ('Theta_power', 4, 7),
    ('Mu_power', 7.5, 12.5),
    ('Delta_power', 0.3, 3),
    ('SMR_power', 12.5, 15.5),
    ]

for freq in freqs:
    field_name, low, high = freq
    data[field_name] = data['signal'].map(lambda x: get_power(x, low, high))

data['All_power'] = data[[freq[0] for freq in freqs]].sum(axis=1)

#for freq in freqs:
#    field_name, low, high = freq
#    data[field_name+'_rel'] = data[field_name] / data['All_power']


# In[10]:

data['Mean'] = data['signal'].map(np.mean)
data['Std'] = data['signal'].map(np.std)
data['Skew'] = data['signal'].map(scipy.stats.skew)
data['Kurtosis'] = data['signal'].map(scipy.stats.kurtosis)


# In[47]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import interp
from sklearn.metrics import roc_curve, auc




X = data.drop(['signal', 'stage'], axis=1)
y = np.abs(data['stage'])
groups = data.index.get_level_values('label').tolist()
a=0
stage_1_fpr, stage_2_fpr, stage_3_fpr, stage_4_fpr = [],[],[],[]
stage_1_tpr, stage_2_tpr, stage_3_tpr, stage_4_tpr = [],[],[],[]
aucs = []
accs_test = []

for q, (train_index, test_index)  in enumerate(LeaveOneGroupOut().split(X=X, y=y, groups=groups)):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    cls = RandomForestClassifier(max_depth=4).fit(X_train, y_train)
    
    probas_ = cls.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    a, b= [] , []
    for i, stage in enumerate([0,1,3,4]):
        fpr, tpr, thresholds = roc_curve(y_test==stage, probas_[:, i])
        #print(fpr.shape)
        a.append(fpr)
        b.append(tpr)
    if q==8:
        ee=y_test*-1
        plt.plot(np.array(y_test*-1))
    stage_1_fpr.append(a[0])
    stage_2_fpr.append(a[1])
    stage_3_fpr.append(a[2])
    stage_4_fpr.append(a[3])
    
    stage_1_tpr.append(b[0])
    stage_2_tpr.append(b[1])
    stage_3_tpr.append(b[2])
    stage_4_tpr.append(b[3])
    
    acc_test = accuracy_score(y_test, cls.predict(X_test))
    accs_test.append(acc_test)
    acc_train = accuracy_score(y_train, cls.predict(X_train))
    print('Acc test: {}, Acc train: {}'.format(acc_test, acc_train))

print('Test accuracy: {} +- {}'.format(np.mean(accs_test), np.std(accs_test)))
our =data[data.index.get_level_values('label')=='TG_180616.h5']
our_ind =our.index.get_level_values('epoch')
 

de=[[stage_1_fpr,stage_1_tpr,"Awake"],[stage_2_fpr,stage_2_tpr,"REM"],
    [stage_3_fpr,stage_3_tpr,"Light Sleep"],
    [stage_4_fpr,stage_4_tpr,"Deep Sleep"]]
for j in range(4):
    plt.subplot(2,2,j+1)
    for i in range(len(de[0][0])):
        fpr, tpr = de[j][0][i], de[j][1][i]
       # roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3)#, alpha=0.3,
                 #label='Rec #%d (AUC = %0.2f)' % (i+1, roc_auc))
        #plt.legend()
        plt.title(str(de[j][2]))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
plt.show()