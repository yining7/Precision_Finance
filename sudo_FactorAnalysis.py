#!/usr/bin/env python
# coding: utf-8

# ## Factor Analysis

# In[1]:


# !pip install factor_analyzer
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt    
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn import preprocessing

import factor_analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('5_FactorAna.csv')


# In[3]:


df.columns


# In[200]:


df_fa = df[['NRC_positive','compound', 'neutral', 'NRC_negative','rating', 'review_count',
            'price1', 'price2', 'price3', 'price4','delivery', 'pickup', 'reservation']]  # 18
df_fa = pd.DataFrame(preprocessing.scale(df_fa),columns=df_fa.columns)


# In[201]:


fac_num=8
fa = FactorAnalyzer(n_factors=fac_num,rotation='varimax',method='principal', impute='drop' )  #n_factors=10,
fa.fit(df_fa)

fa.get_communalities()  #

total_var = pd.DataFrame(np.matrix(fa.get_factor_variance()).T,columns=['variance','proportion_var_explain','cumulative_var_explain'])
total_var
# variance – The factor variances.
# proportional_variance – The proportional factor variances.
# cumulative_variances 


# In[202]:


chi_square_value,p_value=calculate_bartlett_sphericity(df_fa.dropna())
kmo = calculate_kmo(df_fa.dropna())
chi_square_value, p_value
kmo[1]
# Represents the degree to which each observed variable is predicted by the other variables in the dataset.
# In general, a KMO > 0.6 is considered adequate.


# In[203]:


ev, v = fa.get_eigenvalues()  #  eigen values
plt.scatter(range(1,df_fa.shape[1]+1),ev)
plt.plot(range(1,df_fa.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[204]:


factor_loads = pd.DataFrame(np.matrix(fa.loadings_),index=df_fa.columns) # Factor Loadings Matrix
plt.figure(figsize=(6,6))
sns.heatmap(factor_loads, xticklabels= True, yticklabels= True,cmap="bwr")  
# factor0: positive adj factor, 0: positive review factor, 1: neg fac, 2: polarity pos-neg, 3: Michelin specific, 4: services, 
# 5: nonMi specific, 6: reviews, 7: reservation
 


# In[205]:


df_newfa = pd.DataFrame(np.matrix(fa.transform(df_fa.fillna(0))))  # Get the factor scores for new data set

y = df['is_Michelin']
X = sm.add_constant(df_newfa)
model = sm.OLS(y, X).fit()

model.summary()

tvalues = model.tvalues[1:]#/sum(model.tvalues[1:])
# tvalues = model.params[1:]/sum(model.params[1:])

def sort_factors(series, t):
    if t > 0: # positive
        return np.argsort(np.argsort(-series))/sum(np.argsort(-series))
    else: return np.argsort(np.argsort(series))/sum(np.argsort(-series))

for i in range(fac_num):
    df_newfa['factor'+str(i)+'order']= sort_factors(df_newfa.iloc[:,i], tvalues[i])
    
df_newfa['total_order'] = df_newfa.iloc[:,-fac_num:].apply(lambda row: np.dot(row,tvalues),axis=1)

df['FactorAnaScore']=df_newfa['total_order']


# In[206]:


model.summary()


# In[207]:


def get_potentials(df,top=10):
    potent=[]
    for i in df.index:
        potent.append([df['FactorAnaScore'][i],df['name'][i]])
    potential_list = sorted(potent)[:top]
    return potential_list

pre_list = get_potentials(df,top=202)
wall = pre_list[-1][0]

df['reco'] = np.where(df['FactorAnaScore']<wall,1,0)


# In[208]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
fa_cm = metrics.confusion_matrix(df['is_Michelin'], df['reco'])
sns.heatmap(fa_cm, annot=True, cmap = "BuPu", fmt='.2f',xticklabels = ["Non-Michelin", "Michelin"] , yticklabels = ["Non-Michelin", "Michelin"] )
import matplotlib.pyplot as plt
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Factor Analysis')
plt.savefig('factor_ana_cm')

def c_m_analysis(y_test,logreg_y_pred,fa_cm):
    tn, fp, fn, tp = fa_cm.ravel()
    tpr = tp/(tp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fpr = fp/(fp+tn)
    f_score = 2*precision*tpr/(precision+tpr)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print("Precision:\t\t\t%1.2f identified as Michelin are Michelin"%(precision))
    print("Recall/TPR:\t\t\t%1.2f proportion of Michelin identified"%(recall))
    print("False Positive Rate:\t\t%1.2f proportion of Non-Michelin identified as Michelin"%fpr)
    print("f-score:\t\t\t%1.2f tradeoff between precision and recall"%(f_score))
    print("Accuracy:\t\t\t%1.2f how well the model has classified"%(accuracy))

c_m_analysis(df['is_Michelin'],df['reco'],fa_cm)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


pd.DataFrame(pre_sf)


# In[269]:


y = df['is_Michelin']
x = df_pca

pca=PCA(n_components=5)
reduced_x=pca.fit_transform(x)

mi_x,mi_y = [],[]
nonmi_x,nonmi_y = [],[]
for i in range(len(reduced_x)):
    if y[i] ==1:
        mi_x.append(reduced_x[i][0])
        mi_y.append(reduced_x[i][1])
    else:
        nonmi_x.append(reduced_x[i][0])
        nonmi_y.append(reduced_x[i][1])

plt.figure(figsize=(8,5))
plt.scatter(mi_x,mi_y,c='r',marker='.',s=50,zorder=2)
plt.scatter(nonmi_x,nonmi_y,c='g',marker='^')
# plt.ylim(-10,10) # plt.xlim(-10,10)
plt.show()


# #### The result didn't show significant distinction between the two kinds of restaurants, so we didn't go deeper into this method.

# In[199]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
fa_smote_cm = metrics.confusion_matrix(y_resampled, pre_list)
sns.heatmap(fa_smote_cm, annot=True, cmap = "BuPu", fmt='.2f',xticklabels = ["Non-Michelin", "Michelin"] , yticklabels = ["Non-Michelin", "Michelin"] )
import matplotlib.pyplot as plt
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Factor Analysis')
plt.savefig('factor_ana_cm')

def c_m_analysis(y_test,logreg_y_pred,fa_cm):
    tn, fp, fn, tp = fa_cm.ravel()
    tpr = tp/(tp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fpr = fp/(fp+tn)
    f_score = 2*precision*tpr/(precision+tpr)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print("Precision:\t\t\t%1.4f identified as Michelin are Michelin"%(precision))
    print("Recall/TPR:\t\t\t%1.4f proportion of Michelin identified"%(recall))
    print("False Positive Rate:\t\t%1.4f proportion of Non-Michelin identified as Michelin"%fpr)
    print("f-score:\t\t\t%1.4f tradeoff between precision and recall"%(f_score))
    print("Accuracy:\t\t\t%1.4f how well the model has classified"%(accuracy))

c_m_analysis(y_resampled, pre_list,fa_smote_cm)


# In[ ]:




