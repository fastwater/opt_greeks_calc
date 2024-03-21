# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:34:39 2019

@author: fastwater

BS Model option IV Greeks and price calculation ......

cp_sign: 1 for call, -1 for put

"""

import numpy as np
import scipy.stats as sps

# ------------------------计算时间价值------------------------
def timevalue(cp_sign, premium, k, s0):
    innerValue = max(cp_sign * (s0 - k), 0)
    timevalue = premium - innerValue
    if (abs(timevalue) < 0.000001):
        return 0
    else:
        return timevalue

def impVolByNewton(cp_sign, premium, tyear, k, s0, timevalue, sigma=0.9, n=10):

    if timevalue<=0: return 0
        
    for i in range(n):
        d_1 = calcD1(tyear, sigma, k, s0)
        d_2 = calcD2(d_1, sigma, tyear)
        v = getvega(tyear, s0, d_1) 
        if abs(v) < 1e-6: 
            #return impVolByBSearch(cp_sign, premium, tyear, k, s0) # because min price should be min tick_size = 1e-4
            return 0 # 为了速度，否则可以用impVolByBSearch
        sigma -= (bsprice(cp_sign, d_1, d_2, tyear, k, s0) - premium) / getvega(tyear, s0, d_1)
        if (sigma < 0): 
            sigma = 0
            break
    return sigma

def impVolByBSearch(cp_sign, premium, tyear, k, s0, r=0.0):
    smax = 2
    smin = 0
    while(smax - smin > 0.0001):
        sigma = (smax + smin) * 0.5
        d_1 = calcD1(tyear, sigma, k, s0, r=r)
        d_2 = calcD2(d_1, sigma, tyear)
        if bsprice(cp_sign, d_1, d_2, tyear, k, s0, r=r) > premium:
            smax = sigma
        else: smin = sigma
    return sigma


# Black-Scholes
def bsprice(cp_sign, d_1, d_2, tyear, k, s0, r=0.0, dv=0):
    return cp_sign * s0 * np.exp(-dv * tyear) * sps.norm.cdf(cp_sign * d_1) \
            - cp_sign * k * np.exp(-r * tyear) * sps.norm.cdf(cp_sign * d_2)


# ------------------ 计算 d1 和 d2----------------------------------
def calcD1(tyear, sigma, k, s0, r=0.0, dv=0):
    if sigma == 0: sigma = 1e-5 # 为了减少warning输出
    return (np.log(s0/k)+(r-dv+.5*sigma**2)*tyear)/sigma/np.sqrt(tyear)

def calcD2(d_1, sigma, tyear):
    return d_1-sigma*np.sqrt(tyear)
        
# -------------------get Greeks and IV-------------------------
def getdelta(cp_sign, d_1):
    return cp_sign * sps.norm.cdf(cp_sign * d_1)
    
# 未考虑股息！gamma = N＇(d1)/(st*sigma*sqrt(T))
def getgamma(tyear, s0, d_1, sigma):
    if sigma == 0: sigma = 1e-5 # 为了减少warning输出
    return sps.norm.pdf(d_1)/(s0*sigma*np.sqrt(tyear))

# 未考虑股息！Theta --注意N'用的是pdf！导数？
# call: theta = -1*(st*N＇(d1)*sigma)/(2*sqrt(T))-r×k*exp(-r *T)*N(d2)
# put:theta = -1*(st*N＇(d1)*sigma)/(2*sqrt(T))+r×k*exp(-r *T)*N(-1*d2)
def gettheta(cp_sign, tyear, k, s0, d_1, d_2, sigma, r=0.0):
    return -(s0*sps.norm.pdf(d_1)*sigma) / (2*np.sqrt(tyear)) - \
            cp_sign*r*k*np.exp(-r*tyear) * sps.norm.cdf(cp_sign * d_2)
                    
# 未考虑股息！vega = st*sqrt(T)*N＇(d1)
def getvega(tyear, s0, d_1):
    return s0*np.sqrt(tyear)*sps.norm.pdf(d_1)