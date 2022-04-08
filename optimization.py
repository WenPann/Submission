import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error,mean_squared_error
import time
import marshal, types

params = {'font.weight': 'bold',
              'font.size': 14,
              'legend.fontsize': 'medium',
              'axes.labelsize': 'large',
              'axes.labelweight': 'bold',
              'axes.titlesize': 'large',
              'axes.titleweight': 'bold',
              'xtick.labelsize': 'medium',
              'ytick.labelsize': 'medium', 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

data=pd.read_csv('Project_data.csv')

data['month']=pd.to_datetime(data.month, format='%b').dt.month
data['date_abs']=data['year']*12+data['month']
data['count']=data.groupby('API')['date_abs'].transform('count')
data=data[data['count']>5]
data['date_diff']=data['date_abs'].sub(data.groupby('API')['date_abs'].transform('first'))
# %%
#outlier detection with median filter and gradient
wells=data['API'].unique()
data['oil_diff']=data.groupby('API')['oil'].diff()/data.groupby('API')['date_diff'].diff()

# remove extremly low values
data=data[data['oil']>=10]
data_group=data.groupby('API')

#%% regression
def type_curve(x,t,oil):
    logq0,D0,D1,b,t1=x
    logq1=logq0+D0*t1
    ft=np.zeros(t.size)
    early=t<t1
    late=t>=t1
    ft[early]=logq0+D0*t[early]
    ft[late]=logq1-(1/b)*np.log(1+b*D1*(t[late]-t1))
    error=np.mean((oil-ft)**2)
    return error

def type_curve_output(x,t,oil):
    logq0,D0,D1,b,t1=x
    logq1=logq0+D0*t1
    ft=np.zeros(t.size)
    early=t<t1
    late=t>=t1
    ft[early]=logq0+D0*t[early]
    ft[late]=logq1-(1/b)*np.log(1+b*D1*(t[late]-t1))
    return ft

# failed to parallelize, will examine it
# code_string = marshal.dumps(type_curve.__code__)
# code = marshal.loads(code_string)
# func = types.FunctionType(code, globals(), "some_func_name")

t0 = time.time()

results=[]
for i,well in enumerate(wells):
    t,oil=data_group.get_group(well)[['date_diff','oil']].values.T
    t1_loc=np.argmax(oil)
    print(i)
    result = differential_evolution(type_curve, bounds=[(0,10*np.log(oil.max())),
                                                 (0,np.log(oil.max())),
                                                 (0,np.log(oil.max())),
                                                 (0,2),
                                                 (np.max([0,t1_loc-10]),t1_loc+2)],
                                    args=(t,np.log(oil)))
    
    results.append(result.copy())

t1 = time.time()

opt_time=t1-t0
print('Optimization Time {}s'.format(str(t1-t0)))

for i in range(4):
    well=wells[i]
    t,oil=data_group.get_group(well)[['date_diff','oil']].values.T
    y_pred=type_curve_output(results[i]['x'],t,oil)
    plt.figure()
    plt.title('well #{}'.format(str(i)))
    plt.scatter(t,oil,label='cleaned data')
    plt.plot(t,np.exp(y_pred))
    plt.savefig('img/fig'+str(i)+'.pdf')

myerror_log=[i['fun'] for i in results]

myerror_mse=[]
for i in range(len(wells)):
    well=wells[i]
    t,oil=data_group.get_group(well)[['date_diff','oil']].values.T
    y_pred=np.exp(type_curve_output(results[i]['x'],t,oil))
    myerror_mse.append(mean_squared_error(oil, y_pred))

plt.figure()
plt.hist(myerror_mse)
plt.title('MSE')
plt.savefig('img/mse.png')
plt.close()

plt.figure()
plt.hist(myerror_log)
plt.title('MSE_log')
plt.savefig('img/log_mse.png')
plt.close()



#%%
# evaluation/validation
# first 70% as training last 30% as test

# training
results_val=[]
for i,well in enumerate(wells):
    t,oil=data_group.get_group(well)[['date_diff','oil']].values.T
    t=t[:int(len(t)*0.7)]
    oil=oil[:int(len(oil)*0.7)]
    t1_loc=np.argmax(oil)
    print(i)
    result = differential_evolution(type_curve, bounds=[(0,10*np.log(oil.max())),
                                                 (0,np.log(oil.max())),
                                                 (0,np.log(oil.max())),
                                                 (0,2),
                                                 (np.max([0,t1_loc-10]),t1_loc+2)],
                                    args=(t,np.log(oil)))
    
    results_val.append(result.copy())

# test
myerror_mse_val=[]
myerror_log_val=[]
for i in range(len(wells)):
    well=wells[i]
    t,oil=data_group.get_group(well)[['date_diff','oil']].values.T
    t=t[int(len(t)*0.7):]
    oil=oil[int(len(oil)*0.7):]    
    y_pred=np.exp(type_curve_output(results[i]['x'],t,oil))
    myerror_mse_val.append(mean_squared_error(oil, y_pred))
    myerror_log_val.append(mean_squared_error(np.log(oil), np.log(y_pred)))

plt.figure()
plt.hist(myerror_mse_val)
plt.title('MSE_VAL')
plt.savefig('img/mse_val.png')
plt.close()

plt.figure()
plt.hist(myerror_log_val)
plt.title('MSE_LOG_VAL')
plt.savefig('img/log_mse_val.png')
plt.close()

#%%
# iterate on outlier detection
# EDA of outlier wells with high mse
outlier_well_id=np.argsort(myerror_log)[-5:]
for i in outlier_well_id:
    well=wells[i]
    t,oil=data_group.get_group(well)[['date_diff','oil']].values.T
    y_pred=type_curve_output(results[i]['x'],t,oil)
    plt.figure()
    plt.title('well #{}'.format(str(i)))
    plt.scatter(t,oil,label='cleaned data')
    plt.plot(t,np.exp(y_pred))

# well 36 has two production cycles, probably fracking happens in the middle.
# should fit two models for the first and send halfs of the production respectively.
# should use log diff
data['log_oil']=np.log(data['oil'])
data['oil_diff_log']=data.groupby('API')['log_oil'].diff()/data.groupby('API')['date_diff'].diff()
# check oil diff for outlier detection
i=0;data[data['API']==wells[i]]['oil_diff_log'].plot()



#%%
from cvxopt import matrix, log, div, spdiag, solvers

def F(x = None, z = None):
     if x is None:  return 0, matrix(0.0, (3,1))
     if max(abs(x)) >= 1.0:  return None
     u = 1 - x**2
     val = -sum(log(u))
     Df = div(2*x, u).T
     if z is None:  return val, Df
     H = spdiag(2 * z[0] * div(1 + x**2, u**2))
     return val, Df, H