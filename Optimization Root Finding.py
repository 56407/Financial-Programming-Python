
# coding: utf-8

# In[5]:

import pandas as pd
from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import log,sqrt,exp
from scipy import stats
from math import pow


# In[6]:

get_ipython().magic(u'matplotlib notebook')


# In[7]:

pwd


# # remove missing data

# In[4]:

df=pd.read_csv('Option2017.csv')


# In[5]:

df.head()


# In[8]:

cleandf=df.dropna(axis=0,how='any')
len(cleandf)


# In[9]:

cleandf.head()


# In[10]:

cleandf.to_csv('Option2017Clean.csv')


# # merge two datasets

# In[11]:

df2=pd.read_csv('Option2017_2.csv')


# In[12]:

df1=cleandf


# In[13]:

df1.rename(columns={'implied volatility':'Implied volatility'},inplace=True)


# In[14]:

df2=df2.drop('Unnamed: 0',axis=1)


# In[15]:

df2.head()


# In[16]:

len(df2)


# In[17]:

df2=df2.drop_duplicates()


# In[18]:

len(df2)


# In[19]:

Option2017_2_Clean = pd.merge(df1,df2,how='left',on=['Ticker','Type','Last','StrikePrice','Implied volatility','currentDate'])


# In[20]:

len(Option2017_2_Clean)


# In[21]:

Option2017_2_Clean.to_csv('Option2017_2_Clean.csv')


# In[22]:

Option2017_2_Clean.head()


# # Visualize and generate eps/tiff files

# In[23]:

AAPL=Option2017_2_Clean[Option2017_2_Clean['Ticker']=='AAPL']


# In[25]:

fig=plt.figure(figsize=(10,5))
ax=Axes3D(fig)
ax.scatter(AAPL['StrikePrice'],AAPL['Expiration time'],AAPL['Implied volatility'],alpha=0.3)

# set x,y,z label
ax.set_xlabel('strike price')
ax.set_ylabel('Expiration time')
ax.set_zlabel('implied volatility')
ax.set_title('AAPL implied volatility')
plt.show()
fig.savefig('AAPL.eps', format='eps', dpi=1000)


# In[26]:

GOOG=Option2017_2_Clean[Option2017_2_Clean['Ticker']=='GOOG']


# In[27]:

fig=plt.figure(figsize=(10,5))
ax=Axes3D(fig)
plot=ax.scatter(GOOG['StrikePrice'],GOOG['Expiration time'],GOOG['Implied volatility'],alpha=0.3)

# set x,y,z label
ax.set_xlabel('strike price')
ax.set_ylabel('Expiration time')
ax.set_zlabel('implied volatility')
ax.set_title('GOOG implied volatility')

fig.savefig('GOOG.eps', format='eps', dpi=1000)


# In[28]:

# the plot of their bid-ask differencez


# In[29]:

fig=plt.figure(figsize=(10,5))
plt.scatter((AAPL['Ask']-AAPL['Bid']),AAPL['Implied volatility'],alpha=0.3)
plt.xlabel('ask-bid difference')
plt.ylabel('implied volatility')
plt.title('AAPL ask-bid difference with implied volatility')

plt.show()
plt.savefig('AAPLdifference.eps', format='eps', dpi=1000)


# In[30]:

fig=plt.figure(figsize=(10,5))
plt.scatter((GOOG['Ask']-GOOG['Bid']),GOOG['Implied volatility'],alpha=0.3)
plt.xlabel('ask-bid difference')
plt.ylabel('implied volatility')
plt.title('GOOG ask-bid difference with implied volatility')
plt.savefig('GOOGdifference.eps', format='eps', dpi=1000)


# # Visualize and generate eps/tiff files

# In[31]:

fig=plt.figure(figsize=(10,5))
plt.hist(Option2017_2_Clean['Implied volatility'],bins=len(np.arange(min(Option2017_2_Clean['Implied volatility']),max(Option2017_2_Clean['Implied volatility']),0.1)))
plt.tick_params(axis='x', which='major', labelsize=7)
plt.xlabel('Implied Volatility')
plt.ylabel('Number of Option')
plt.title('Histogram of Implied Volatility')
plt.xticks(np.arange(min(Option2017_2_Clean['Implied volatility']),max(Option2017_2_Clean['Implied volatility']),0.1))
plt.show()
plt.savefig('Histogram of Implied Volatility.eps',format='eps',dpi=1000)


# In[32]:

fig=plt.figure(figsize=(10,5))
plt.hist(Option2017_2_Clean['Expiration time'].dropna(),bins=len(np.arange(min(Option2017_2_Clean['Expiration time']),max(Option2017_2_Clean['Expiration time']),0.1)))
plt.tick_params(axis='x', which='major', labelsize=6)
plt.xlabel('Expiration time')
plt.ylabel('Number of Option')
plt.title('Histogram of Expiration time')
plt.xticks(np.arange(min(Option2017_2_Clean['Expiration time']),max(Option2017_2_Clean['Expiration time']),0.1))
plt.show()
plt.savefig('Histogram of Expiration time.eps',format='eps',dpi=1000)


# In[33]:

fig=plt.figure(figsize=(10,5))
plt.hist(Option2017_2_Clean['Underlaying asset price'].dropna(),bins=np.arange(0,1000,100))
plt.tick_params(axis='x', which='major', labelsize=10)
plt.xlabel('Underlaying asset price')
plt.ylabel('Number of Option')
plt.title('Histogram of Underlaying asset price')

plt.show()
plt.savefig('Histogram of Underlaying asset price.eps',format='eps',dpi=1000)


# In[34]:

fig=plt.figure(figsize=(10,5))
plt.hist(Option2017_2_Clean['Last'],bins=np.arange(0,22,2))
plt.tick_params(axis='x', which='major', labelsize=7)
plt.xlabel('Option Price')
plt.ylabel('Number of Option')
plt.title('Histogram of Option Price')
plt.xticks(np.arange(0,22,2))
plt.show()
plt.savefig('Histogram of Option Price.eps',format='eps',dpi=1000)


# # Finish the following analytics work

# In[8]:

Option2017_2_Clean=pd.read_csv('Option2017_2_Clean.csv')


# In[9]:

Option2017_2_Clean.drop('Unnamed: 0',axis=1).head()


# In[10]:

def bsm_pricing(S, K, T, r, sigma, option_type):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type.strip().lower()=='call':
        N_d1 = stats.norm.cdf(d1, 0.0, 1.0)
        N_d2 = stats.norm.cdf(d2, 0.0, 1.0)
        call_price = (S * N_d1 - K * exp(-r * T) * N_d2)
        return call_price
        
    else:
        N_n_d2=stats.norm.cdf(-d2,0.0,1.0)
        N_n_d1=stats.norm.cdf(-d1,0.0,1.0)
        put_price=K*exp(-r*T)*N_n_d2-S*N_n_d1
        return put_price
        


# In[11]:

#Bisection Algorithm Code


# In[12]:

fig=plt.figure(figsize=(10,5))
Option2017_2_Clean['Implied volatility'].hist()
plt.show()


# In[13]:

Option2017_2_Clean=Option2017_2_Clean.dropna()


# In[14]:

# we try with (0,2)


# In[15]:

len(Option2017_2_Clean)


# In[16]:

Option2017_2_Clean.reset_index(inplace=True)
Option2017_2_Clean.tail()


# In[17]:

# we can't set 0 as the small interval
def bisection(fs,fb,small,big,S,K,T,r,option_type,market_price,eps):
    
    count=0
    if fs*fb==0:
        if fs==0:
            return small,count
        else:
            return big,count
    if fs*fb>0:
        return np.NaN,count
    while (fs*fb<0):
        count=count+1
        middle=(small+big)/2.0
        fm= (bsm_pricing(S,K,T,r,middle,option_type)- market_price)
        if (fs*fm<0):
            big=middle
            fb=(bsm_pricing(S,K,T,r,big,option_type)- market_price)
        if (fm*fb<0):
            small=middle
            fs=(bsm_pricing(S,K,T,r,small,option_type)- market_price)
        if ( abs(fm) < eps):
            return middle,count


# In[18]:

index_list=[]
for i in  Option2017_2_Clean.index:
    
    S= Option2017_2_Clean.iloc[i]['Underlaying asset price']
    K= Option2017_2_Clean.iloc[i]['StrikePrice']
    T= Option2017_2_Clean.iloc[i]['Expiration time']
    r=0.03
    market_price=Option2017_2_Clean.iloc[i]['Last']
    small=pow(10,-10)
    big=2.5
    option_type= Option2017_2_Clean.iloc[i]['Type']
    eps= pow(10,-10)
    
    fs = (bsm_pricing(S,K,T,r,small,option_type)- market_price)
    fb = (bsm_pricing(S,K,T,r,big,option_type)- market_price)
    
    middle,count=bisection(fs,fb,small,big,S,K,T,r,option_type,market_price,eps)
    if abs( middle - Option2017_2_Clean.iloc[i]['Implied volatility'])<0.005:
        index_list.append(i)
        print i
    
   
     


# In[19]:

len(index_list)


# In[20]:

European_option= Option2017_2_Clean.iloc[index_list]


# In[21]:

European_option=European_option.drop(['index','Unnamed: 0'],axis=1)


# In[22]:

European_option.head()


# In[30]:


fig,ax = plt.subplots()
ax.hist(European_option['Implied volatility'],bins=np.arange(0,2,0.2))
ax.tick_params(axis='x', which='major', labelsize=7)
ax.set_xlabel('Implied volatility')
ax.set_ylabel('Number of Option')
ax.set_title('Histogram of European option Implied volatility')
ax.set_xticks(np.arange(0,2,0.2))
for value, tick in zip(np.histogram(European_option['Implied volatility'],bins=np.arange(0,2,0.2))[0],np.histogram(European_option['Implied volatility'],bins=np.arange(0,2,0.2))[1]):
    ax.text(tick+0.05,value+5,value)
fig.savefig('Histogram of European_option Implied volatility.eps',format='eps',dpi=1000)


# In[31]:

fig,ax = plt.subplots()
ax.hist(European_option['Expiration time'],bins=np.arange(0,1,0.1))
ax.tick_params(axis='x', which='major', labelsize=7)
ax.set_xlabel('Expiration time')
ax.set_ylabel('Number of Option')
ax.set_title('Histogram of European option Expiration time')
ax.set_xticks(np.arange(0,1,0.1))
for value, tick in zip(np.histogram(European_option['Expiration time'],bins=np.arange(0,1,0.1))[0],np.histogram(European_option['Expiration time'],bins=np.arange(0,1,0.1))[1]):
    ax.text(tick+0.03,value+5,value)
fig.savefig('Histogram of European_option Expiration time.eps',format='eps',dpi=1000)


# In[32]:

fig,ax = plt.subplots()
ax.hist(European_option['Underlaying asset price'],bins=np.arange(0,200,10))
ax.tick_params(axis='x', which='major', labelsize=7)
ax.set_xlabel('Underlaying asset price')
ax.set_ylabel('Number of Option')
ax.set_title('Histogram of European option Underlaying asset price')
ax.set_xticks(np.arange(0,200,10))
for value, tick in zip(np.histogram(European_option['Underlaying asset price'],bins=np.arange(0,200,10))[0],np.histogram(European_option['Underlaying asset price'],bins=np.arange(0,200,10))[1]):
    ax.text(tick+1,value+0.5,value)
fig.savefig('Histogram of European_option Underlaying asset price.eps',format='eps',dpi=1000)


# In[60]:

fig,ax = plt.subplots()
ax.hist(European_option['Last'],bins=np.arange(0,50,5))
ax.tick_params(axis='x', which='major', labelsize=7)
ax.set_xlabel('Last')
ax.set_ylabel('Number of Option')
ax.set_title('Histogram of European option Last')
ax.set_xticks(np.arange(0,50,5))
for value, tick in zip(np.histogram(European_option['Last'],bins=np.arange(0,50,5))[0],np.histogram(European_option['Last'],bins=np.arange(0,50,5))[1]):
    ax.text(tick+1,value+0.5,value)
fig.savefig('Histogram of European_option Last.eps',format='eps',dpi=1000)


# In[61]:

# apply the following methods to estimate the implied volatilityn for the option|


# In[33]:

eps=pow(10,-12)
r=0.03


# In[34]:

def f(S,K,T,r,sigma,option_type,market_price):
    return bsm_pricing(S,K,T,r,sigma,option_type)- market_price


# In[35]:

European_option.head()


# # Bisection

# In[36]:

est_volatility_Bi=[]
count_list=[]
for i in  range(len(European_option.index)):
    
    S= European_option.iloc[i]['Underlaying asset price']
    K= European_option.iloc[i]['StrikePrice']
    T= European_option.iloc[i]['Expiration time']
    market_price=European_option.iloc[i]['Last']
    small=pow(10,-10)
    big=1.91
    option_type= European_option.iloc[i]['Type']
    
    
    fs = (bsm_pricing(S,K,T,r,small,option_type)- market_price)
    fb = (bsm_pricing(S,K,T,r,big,option_type)- market_price)
    
    middle,count=bisection(fs,fb,small,big,S,K,T,r,option_type,market_price,eps)
   
    est_volatility_Bi.append(middle)
    count_list.append(count)
   


# In[37]:

print "average iteration number to converge for Bisection is "+ str(sum(count_list)/len(count_list))


# In[38]:

European_option['est_volatility_Bi']=est_volatility_Bi
fig=plt.figure(figsize=(10,5))
((European_option['est_volatility_Bi']-European_option['Implied volatility'])**2).hist()
plt.title('SE for Bisection')


# In[39]:

print "MSE of Bisection is "+str(((European_option['est_volatility_Bi']-European_option['Implied volatility'])**2).mean())


# In[ ]:




# # Brent method

# In[40]:

from scipy.optimize import brentq


# In[41]:

def f(x):
    return bsm_pricing(S,K,T,r,x,option_type)- market_price
small=pow(10,-10)
big=1.91
est_volatility_Br=[]
count_list=[]
eps=pow(10,-12)
r=0.03
for i in  range(len(European_option.index)):
    
    S= European_option.iloc[i]['Underlaying asset price']
    K= European_option.iloc[i]['StrikePrice']
    T= European_option.iloc[i]['Expiration time']
    market_price=European_option.iloc[i]['Last']
    
    option_type= European_option.iloc[i]['Type']
    
    root,rootresult=brentq(f,small,big,full_output = True)
    
    
    est_volatility_Br.append(root)
    count_list.append(rootresult.iterations)


# In[42]:

print "average iteration number to converge for Brent is "+ str(sum(count_list)/len(count_list))


# In[43]:

European_option['est_volatility_Br']=est_volatility_Br
fig=plt.figure(figsize=(10,5))
((European_option['est_volatility_Br']-European_option['Implied volatility'])**2).hist()
plt.title('SE for Brent')


# In[74]:

print "MSE of Brent is "+str(((European_option['est_volatility_Br']-European_option['Implied volatility'])**2).mean())


# In[ ]:




# # Muller-Bisection

# In[44]:

def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.
    else:
        return x


# In[50]:

def f(x):
    return (bsm_pricing(S,K,T,r,x,option_type)- market_price)


# In[56]:

def muller_bisection(fs,fb,small,big,S,K,T,r,option_type,market_price,eps):
    count=0
    a=small
    b=big
    c=(small+big)/2
    if fs*fb==0:
        if fs==0:
            return small,count
        else:
            return big,count
    if fs*fb>0:
        return np.NaN,count  
    while f(small)* f(big)< 0:    
        #f(small)-C = (small-middle)^2*A + (small-middle)*B  #f(big)-C = (big-middle)^2 *A + (big-middle)*B        
        count=count+1
        if f(a)*f(c)<0:
            a2=a
            b2=c
            
        if f(c)*f(b)<0:
            a2=c
            b2=b           
        #C=f(c) #solve_array = np.array([[((a-c)**2),((b-c)**2)], [(a-c),(b-c)]]) #outcome_array = np.array([ (f(a)-C),(f(b)-C)])    
        #outcome = np.linalg.solve(solve_array, outcome_array  #A=outcome[0]  #B=outcome[1]
        C=f(c)
        B=((a-c)**2*(f(b)-f(c))-(b-c)**2*(f(a)-f(c)))/((a-c)*(b-c)*(a-b))
        A=((b-c)*(f(a)-f(c))-(a-c)*(f(b)-f(c)))/((a-c)*(b-c)*(a-b))
        
        c2=c- (2*C)/(B+sign(B)*sqrt(B**2-4*A*C))   
        
        if f(a)*f(c)<0:
            c=b
        if f(c)*f(b)<0:
            c=a        
        if  (c2<=b2 and c2>=a2):
            c2=c2
        else:
            c2=(a2+b2)/2
        c=c2 
        a=a2
        b=b2
        if (f(c2)==0 or abs(f(c2))<eps):
            return c2,count


# In[57]:

small=pow(10,-10)
big=1.91
est_volatility_Mu=[]
eps=pow(10,-12)
r=0.03

count_list=[]

for i in  range(len(European_option.index)):
    small=pow(10,-10)
    big=1.91
    fs=f(small)
    fb=f(big)
    S= European_option.iloc[i]['Underlaying asset price']
    K= European_option.iloc[i]['StrikePrice']
    T= European_option.iloc[i]['Expiration time']
    market_price=European_option.iloc[i]['Last']
    
    option_type= European_option.iloc[i]['Type']
    
    
    
    est,count = muller_bisection(fs,fb,small,big,S,K,T,r,option_type,market_price,eps)
    est_volatility_Mu.append(est)
    count_list.append(count)


# In[58]:

print "Average iteration number to converge for Muller-Bisection is "+ str(sum(count_list)/len(count_list))


# In[59]:

European_option['est_volatility_Mu']=est_volatility_Mu
fig=plt.figure(figsize=(10,5))
((European_option['est_volatility_Mu']-European_option['Implied volatility'])**2).hist()
plt.title('SE for Muller-Bisection')


# In[60]:

print "MSE of Muller-Bisection is "+str(((European_option['est_volatility_Mu']-European_option['Implied volatility'])**2).mean())


# # Newton method

# In[86]:

def bsm_vega(S,K,T,r,sigma):
    d1= (log(S/K)+ (r+0.5*sigma**2)*T)/ (sigma*sqrt(T))
    vega= S*stats.norm.pdf(d1,0.0,1.0)* sqrt(T)
    return vega


# In[87]:

def bsm_call_imp_vol(S,K,T,r,C_star, sigma_est,option_type):
    count=0
    if abs(bsm_pricing(S,K,T,r,sigma_est,option_type)- C_star) < eps:
        return sigma_est,count
    else:
        while abs(bsm_pricing(S,K,T,r,sigma_est,option_type)- C_star) > eps:
            count=count+1
            f1 = bsm_pricing(S,K,T,r,sigma_est,option_type)-C_star
            f_prime = bsm_vega(S,K,T,r,sigma_est)
            sigma_1=sigma_est
            sigma_est=sigma_est-(f1/f_prime)
            if ( abs(bsm_pricing(S,K,T,r,sigma_est,option_type)-C_star) < eps or abs(sigma_est-sigma_1)< eps):
                return sigma_est,count


# In[88]:


r=0.03

est_volatility_Ne=[]
count_list=[]
for i in  range(len(European_option.index)):
    
    sigma=0.5
    
    S= European_option.iloc[i]['Underlaying asset price']
    K= European_option.iloc[i]['StrikePrice']
    T= European_option.iloc[i]['Expiration time']
    market_price=European_option.iloc[i]['Last']
    
    option_type= European_option.iloc[i]['Type']
    
    
    est,count= bsm_call_imp_vol(S,K,T,r,market_price, sigma,option_type)
    
    est_volatility_Ne.append(est)
    count_list.append(count)
   
    


# In[89]:

print "average iteration number to converge for Newton method is "+ str(sum(count_list)/len(count_list))


# In[90]:

European_option['est_volatility_Ne']=est_volatility_Ne
fig=plt.figure(figsize=(10,5))
((European_option['est_volatility_Ne']-European_option['Implied volatility'])**2).hist()
plt.title('SE for Newton method')


# In[91]:

print "MSE of Newton method is "+str(((European_option['est_volatility_Ne']-European_option['Implied volatility'])**2).mean())


# # New newton (brent as fill-in)

# In[100]:

def f(x):
    return bsm_pricing(S,K,T,r,x,option_type)- market_price


# In[101]:

r=0.03
small=-0.9
big=2.

est_volatility_Nn=[]
count_list=[]
for i in  range(len(European_option.index)):
    
    
    
    S= European_option.iloc[i]['Underlaying asset price']
    K= European_option.iloc[i]['StrikePrice']
    T= European_option.iloc[i]['Expiration time']
    market_price=European_option.iloc[i]['Last']
    
    option_type= European_option.iloc[i]['Type']
    
    sigma=brentq(f,small,big,maxiter=5,disp=False)
    
    est,count= bsm_call_imp_vol(S,K,T,r,market_price, sigma,option_type)
    
    est_volatility_Nn.append(est)   
    count_list.append(count)


# In[102]:

print "average iteration number to converge for New Newton is "+ str(sum(count_list)/len(count_list))


# In[103]:

European_option['est_volatility_Nn']=est_volatility_Nn
fig=plt.figure(figsize=(10,5))
((European_option['est_volatility_Nn']-European_option['Implied volatility'])**2).hist()
plt.title('SE for New Newton')


# In[104]:

print "MSE of New Newton is "+str(((European_option['est_volatility_Nn']-European_option['Implied volatility'])**2).mean())


# # New Harley (brent as fill-in)

# In[105]:

def f(x):
    return bsm_pricing(S,K,T,r,x,option_type)- market_price


# In[106]:

def bsm_vomma(S,K,T,r,sigma):
    d1= (log(S/K)+ (r+0.5*sigma**2)*T)/ (sigma*sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    vomma= S*stats.norm.pdf(d1,0.0,1.0)* sqrt(T)*d1*d2/sigma
    
    return vomma


# In[107]:

def harley(S,K,T,r,sigma,option_type,market_price):
    count=0
    if abs(f(sigma)) < eps:
        return sigma,count
    while abs(f(sigma)) > eps:
        count=count+1
        vomma=bsm_vomma(S,K,T,r,sigma)
        vega=bsm_vega(S,K,T,r,sigma)
        sigma=sigma-(f(sigma)/vega)* (1-(f(sigma)/vega)*(vomma/(2*vega)))**(-1)    
        if  abs(f(sigma))< eps :
            return sigma,count


# In[108]:

r=0.03
small=-0.999
big=2.

est_volatility_Ha=[]
eps=pow(10,-12)
count_list=[]

for i in  range(len(European_option.index)):
    
    
    
    S= European_option.iloc[i]['Underlaying asset price']
    K= European_option.iloc[i]['StrikePrice']
    T= European_option.iloc[i]['Expiration time']
    market_price=European_option.iloc[i]['Last']
    
    option_type= European_option.iloc[i]['Type']
    
    sigma=brentq(f,small,big,maxiter=5,disp=False)
    
    est,count=harley(S,K,T,r,sigma,option_type,market_price)
    est_volatility_Ha.append(est)
    count_list.append(count)
   


# In[109]:

print "average iteration number to converge for Harley is "+ str(sum(count_list)/len(count_list))


# In[110]:

European_option['est_volatility_Ha']=est_volatility_Ha
fig=plt.figure(figsize=(10,5))
((European_option['est_volatility_Ha']-European_option['Implied volatility'])**2).hist()
plt.title('SE for Harley')


# In[111]:

print "MSE of Harley is "+str(((European_option['est_volatility_Ha']-European_option['Implied volatility'])**2).mean())


# # Halley's irrational formula

# In[112]:

def irr_harley(S,K,T,r,sigma,option_type,market_price):
    count=0
    if abs(f(sigma)) < eps:
        return sigma,count
    while abs(f(sigma)) > eps:
        count=count+1
        vomma=bsm_vomma(S,K,T,r,sigma)
        vega=bsm_vega(S,K,T,r,sigma)
        #sigma=sigma+(-vega+sqrt((vega**2)-2*f(sigma)*vomma))/vomma  
        sigma=sigma-(f(sigma)/vega)* (1-(f(sigma)/vega)*(vomma/(2*vega)))**(-1)
        if  abs(f(sigma))< eps :
            return sigma,count


# In[113]:

r=0.03
small=-0.999
big=2.
count_list=[]
est_volatility_Ih=[]
eps=pow(10,-12)


for i in  range(len(European_option.index)):
    
    
    
    S= European_option.iloc[i]['Underlaying asset price']
    K= European_option.iloc[i]['StrikePrice']
    T= European_option.iloc[i]['Expiration time']
    market_price=European_option.iloc[i]['Last']
    
    option_type= European_option.iloc[i]['Type']
    
    sigma=0.5
    
    est,count=irr_harley(S,K,T,r,sigma,option_type,market_price)
    est_volatility_Ih.append(est)
    count_list.append(count)


# In[114]:

print "average iteration number to converge for Irrational Harley is "+ str(sum(count_list)/len(count_list))


# In[115]:

European_option['est_volatility_Ih']=est_volatility_Ih
fig=plt.figure(figsize=(10,5))
((European_option['est_volatility_Ih']-European_option['Implied volatility'])**2).hist()
plt.title('SE for Irrational Harley')


# In[116]:

print "MSE of Irrational Harley is "+str(((European_option['est_volatility_Ih']-European_option['Implied volatility'])**2).mean())


# In[ ]:



