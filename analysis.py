# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.animation import FuncAnimation
from numpy import exp, pi, sinh, tanh, cosh, abs, log, log10
from matplotlib.ticker import ScalarFormatter
from scipy import integrate
from scipy import stats
from scipy.special import ellipk, ellipe
import matplotlib.cm as cm

import os
import csv


############### parameter ################
n = 100
Nsample = int(1e4)
Nspin = n*n
# 0:熱浴法 1:メトロポリス法
METHOD = 0
# 0:熱平衡は全てサンプリング 1:サンプリングの配位間を開ける 2: その2
DISTANCE = 2
# plt.show() するかしないか
PLT = 1
tau_array = np.arange(0.1,5.01,0.01)
tau_c = 2.0/(log(1+sqrt(2)))
print("Critical temperature : ", tau_c)
xticks = [0,1,2,tau_c,3,4,5]
xticklabels = ["0","1","2",r"$\tau_c$","3","4","5"]
############### parameter ################

##############################################################
HEAT_PATH = "./data/heat_bath/spin%dsample%.1e"%(n,(float(Nsample)))
METR_PATH = "./data/metropolis/spin%dsample%.1e"%(n,(float(Nsample)))
COMPARE_PATH = "./figs/compare/spin%dsample%.1e"%(n,(float(Nsample)))
if(DISTANCE != 0):
  COMPARE_PATH += "distance%d"%DISTANCE
if(METHOD == 0):
  method = "heat_bath"
if(METHOD == 1):
  method = "metropolis"
DATA_PATH = "./data/%s/spin%dsample%.1e"%(method,n,(float(Nsample)))
# DIST_PATH = DATA_PATH + "/distribution"
SAVE_PATH = "./figs/%s/spin%dsample%.1e"%(method,n,(float(Nsample)))
if(DISTANCE != 0):
  DATA_PATH += "distance%d"%DISTANCE
  SAVE_PATH += "distance%d"%DISTANCE
  HEAT_PATH += "distance%d"%DISTANCE
  METR_PATH += "distance%d"%DISTANCE
  print(SAVE_PATH)
TITLE = r"%s (N$_{spin}=%d\times%d$, N$_{sample}=10^%d$)"%(method,n,n,log10(Nsample))
os.makedirs(SAVE_PATH+"/distribution",exist_ok=True)
os.makedirs(COMPARE_PATH,exist_ok=True)
# os.makedirs(DIST_PATH+"/relaxation",exist_ok=True)
df_result = pd.read_csv("%s/result.csv"%DATA_PATH)
##############################################################

def set_ax(ax):
  # ax.set_title("Voight Profile", fontsize=24)
  # ax.set_ylabel("Voight Profile", fontsize=21)
  # ax.set_ylabel(r"$\phi$(x)", fontsize=21)
  # ax.set_xlabel("x", fontsize=21)
  ax.grid(linestyle="dotted")
  ax.xaxis.set_tick_params(direction='in')
  ax.yaxis.set_tick_params(direction='in')
  # ax.tick_params(labelsize=21)
  # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  # ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
  # ax.yaxis.offsetText.set_fontsize(16)

def my_K(k):
  ret = []
  for k_ in k:
    f = lambda theta : 1.0/np.sqrt(1.0-k_*k_*np.sin(theta)**2)
    ret.append(integrate.quad(f, 0.0, pi/2.0)[0])
  return np.array(ret)

def my_E(k):
  ret = []
  for k_ in k:
    f = lambda theta : np.sqrt(1.0-k_*k_*np.sin(theta)**2)
    ret.append(integrate.quad(f, 0.0, pi/2.0)[0])
  return np.array(ret)

def coth(x):
  return 1/tanh(x)

def u_ana(tau):
  k = 2.0/(cosh(2.0/tau)*coth(2.0/tau))
  k2 = 2.0*tanh(2.0/tau)*tanh(2.0/tau)-1
  # tmp = 1+2.0/pi*k2*ellipk(k)
  tmp = 1+2.0/pi*k2*my_K(k)
  return -1.0/tau*coth(2.0/tau)*tmp

def c_ana(tau):
  k = 2.0/(cosh(2.0/tau)*coth(2.0/tau))
  k2 = 2.0*tanh(2.0/tau)*tanh(2.0/tau)-1
  # tmp = 2.0*ellipk(k)-2.0*ellipe(k)-(1-k2)*(pi/2.0+k2*ellipk(k))
  tmp = 2.0*my_K(k)-2.0*my_E(k)-(1-k2)*(pi/2.0+k2*my_K(k))
  return 2.0/(tau*tau*pi)*coth(2.0/tau)*coth(2.0/tau)*tmp

def m_ana(tau):
  ret = [] 
  for t in tau:
    if(t < tau_c):
      tmp = 1-sinh(2.0/t)**(-4.0)
      ret.append(tmp**(1.0/8.0))
    else:
      ret.append(0)
  return ret

def u_plot():
  tau = df_result["T"]
  u = df_result["e"]
  u_sig = df_result["e_sig"]
  
  fig = plt.figure(figsize=(8, 8))
  fig.subplots_adjust(left=0.2)
  fig.suptitle("internal energy %s"%TITLE)
  ax1 = fig.add_subplot(211,xticks=xticks,xticklabels=xticklabels)
  ax2 = fig.add_subplot(212,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax1)
  set_ax(ax2)
  ax1.axvline(x=tau_c,color="r",ls="--")
  ax2.axvline(x=tau_c,color="r",ls="--")
  ax1.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax2.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)

  ax1.set_title("internal energy")
  ax2.set_title("error")
  ax1.set_ylabel(r"internal energy U/(k$_{\mathrm{B}}$TN$_{\mathrm{spin}}$)",fontsize=16)
  ax2.set_ylabel(r"error $\sigma_u$",fontsize=16)
  ax1.plot(tau_array, u_ana(tau_array), label="analytic")
  ax1.errorbar(tau, u, yerr=u_sig, fmt=".", label="numeric")
  ax2.plot(tau, abs(u-u_ana(tau)), ".", label="absolute error")
  ax2.plot(tau, u_sig, ".", label="estimated error")
  ax2.set_yscale("log")
  ax1.legend()
  ax2.legend()
  plt.tight_layout(rect=[0,0,1,0.96])

  plt.savefig("%s/u.png"%SAVE_PATH)

def c_plot():
  tau = df_result["T"]
  c = df_result["c"]
  c_sig = df_result["c_sig"]

  fig = plt.figure(figsize=(8, 8))
  fig.subplots_adjust(left=0.2)
  ax = fig.add_subplot(111,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax)
  ax.set_title(TITLE)
  ax.axvline(x=tau_c,color="r",ls="--")
  ax.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax.set_ylabel(r"heat capacity C/(k$_\mathrm{B}$N$_{\mathrm{spin}}$)",fontsize=16)
  ax.plot(tau_array, c_ana(tau_array), label="analytic")
  # ax.errorbar(tau, c, yerr=c_sig, fmt=".", label="numeric")
  ax.errorbar(tau, c, yerr=c_sig, fmt=".", label="numeric", markersize=7)
  
  ax.legend()
  plt.savefig("%s/c.png"%SAVE_PATH)
  # plt.show()

def m_plot():
  tau = np.array(df_result["T"])
  m = np.array(df_result["m"])
  m_sig = np.array(df_result["m_sig"])

  fig = plt.figure(figsize=(8, 8))
  fig.subplots_adjust(left=0.2)
  fig.suptitle("magnetization %s"%TITLE)
  ax1 = fig.add_subplot(211,xticks=xticks,xticklabels=xticklabels)
  ax2 = fig.add_subplot(212,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax1)
  set_ax(ax2)
  ax1.axvline(x=tau_c,color="r",ls="--")
  ax2.axvline(x=tau_c,color="r",ls="--")
  ax1.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax2.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)

  ax1.set_title("magnetization")
  ax2.set_title("error")
  ax1.set_ylabel(r"magnetiaztion |M|/N$_{\mathrm{spin}}$",fontsize=16)
  ax2.set_ylabel(r"error $\sigma_m$",fontsize=16)
  ax1.plot(tau_array, m_ana(tau_array), label="analytic")
  ax1.errorbar(tau, m, yerr=m_sig, fmt=".", label="numeric")
  ax2.plot(tau, abs(m-m_ana(tau)), ".", label="absolute error")
  ax2.plot(tau, m_sig, ".", label="estimated error")
  ax2.set_yscale("log")
  ax1.legend()
  ax2.legend()

  plt.tight_layout(rect=[0,0,1,0.96])

  plt.savefig("%s/m.png"%SAVE_PATH)

def K_E_compare():
  df = pd.read_csv("./data/ellips.csv")
  k = df["k"]
  K_pp = df["K"]
  E_pp = df["E"]
  fig = plt.figure(figsize=(8, 8))
  # fig.subplots_adjust(left=0.2)
  ax = fig.add_subplot(111)
  set_ax(ax)
  ax.set_xlabel(r"k",fontsize=16)
  ax.set_ylabel("K(k),E(k)",fontsize=16)
  ax.plot(k, ellipk(k), "b", label="scipy K(k)")
  ax.plot(k, my_K(k), "b--" ,label="my K(k)")
  ax.plot(k, K_pp, "b.-" ,label="C++ K(k)")
  ax.plot(k, ellipe(k), "r", label="scipy E(k)")
  ax.plot(k, my_E(k), "r--" ,label="my E(k)")
  ax.plot(k, E_pp, "r.-" ,label="C++ E(k)")
  ax.legend()
  # plt.show()
  plt.savefig("./figs/EK_compare.png")

def dist_plot():
  tau_list = df_result["T"]
  for tau in tau_list:
    # if(tau != 1.0):
    #   continue
    TE_df = pd.read_csv("%s/TE/T%.3f.csv"%(DATA_PATH,tau))
    TE_No = TE_df["No"]
    TE_ene = TE_df["Energy"]
    TE_mag = TE_df["Magnetization"]
    dist_df = pd.read_csv("%s/distribution/T%.3f.csv"%(DATA_PATH,tau))
    dist_No = dist_df["No"]
    dist_ene = dist_df["Energy"]
    dist_mag = dist_df["Magnetization"]
    # dist_ene = np.array(dist_df["Energy"])
    # dist_mag = np.array(dist_df["Magnetization"])

    # fig = plt.figure(figsize=(4, 4))
    fig = plt.figure(figsize=(8, 8))
    # fig.subplots_adjust(top=0.2)
    fig.suptitle(r"$\tau$=%.2f %s"%(tau,TITLE))
    ax_ustep = fig.add_subplot(221)
    ax_mstep = fig.add_subplot(222)
    ax_udist = fig.add_subplot(223)
    ax_mdist = fig.add_subplot(224)
    set_ax(ax_ustep)
    set_ax(ax_mstep)
    set_ax(ax_udist)
    set_ax(ax_mdist)
    ax_ustep.set_title("internal energy step")
    ax_mstep.set_title("magnetization step")
    ax_udist.set_title("internal energy distribution")
    ax_mdist.set_title("magnetization distribution")

    ax_ustep.set_xlabel("MC step",fontsize=16)
    ax_ustep.set_ylabel(r"internal energy U/(k$_{\mathrm{B}}$TN$_{\mathrm{spin}}$)",fontsize=16)
    ax_ustep.plot(TE_No,TE_ene,"b",label="thermal relaxation")
    ax_ustep.plot(dist_No,dist_ene,"r",label="sampling")
    ax_ustep.legend()

    ax_mstep.set_xlabel("MC step",fontsize=16)
    ax_mstep.set_ylabel(r"magnetiaztion M/N$_{\mathrm{spin}}$",fontsize=16)
    ax_mstep.set_ylim(-1.1,1.1)
    ax_mstep.plot(TE_No,TE_mag,"b",label="thermal relaxation")
    ax_mstep.plot(dist_No,dist_mag,"r",label="sampling")
    ax_mstep.legend()

    print(tau)
    bins = 2501
    if(tau == 1.0):
      bins = 29
      # bins = np.array()
      # print("ou")
    # print(bins)
    ax_udist.hist(dist_ene, ec="black",density=True,bins=bins,align="mid")
    # ax_udist.hist(dist_ene, ec="black",density=True,align="mid")
    ax_udist.set_ylabel("distribution", fontsize=16)
    ax_udist.set_xlabel(r"internal energy U/(k$_{\mathrm{B}}$TN$_{\mathrm{spin}}$)",fontsize=16)
    # ax_udist.hist(Energys, bins=33, ec="black", align="left", range=(-100,32), 
    #   stacked=False, label=labels, rwidth=30)
    ax_mdist.set_xlabel(r"magnetiaztion M/N$_{\mathrm{spin}}$",fontsize=16)
    # ax_mdist.hist(dist_mag, ec="black", align="left")
    ax_mdist.hist(dist_mag, ec="black" ,density=True, bins=bins, align="mid",range=(-1,1))
    ax_mdist.set_ylabel("distribution", fontsize=16)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig("%s/distribution/T%.3f.PNG"%(SAVE_PATH,tau), bbox_inches="tight")
    # plt.savefig("%s/distribution/T%.3f.png"%(SAVE_PATH,tau), bbox_inches="tight")
    # plt.savefig("%s/distribution/T%.3f.png"%(SAVE_PATH,tau))

def Corr_func(t_list,X_array):
  num = len(X_array)
  ret = []
  for t in t_list:
    X_array1 = X_array[0:num-t]
    X_array2 = X_array[t:]
    corr = (X_array1*X_array2).mean()
    tmp = (corr-X_array.mean()**2)/np.var(X_array)
    ret.append(tmp)
  return ret

def Self_correlation():
  step_list = np.arange(0,101,1,dtype=int)
  tau_list = np.array(df_result["T"])
  tau_list = sorted(tau_list)
  tau_min = tau_list[2]
  tau_max = tau_list[-1]
  
  fig = plt.figure(figsize=(8, 8))
  # fig.subplots_adjust(top=0.2)
  # fig.suptitle(r"$\tau$=%.2f %s"%(tau,TITLE))
  ax = fig.add_subplot(111)
  set_ax(ax)
  ax.set_ylim(-0.1,1.1)
  ax.set_xlabel("step distance t", fontsize=16)
  ax.set_ylabel("self correlation function R(t)", fontsize=16)
  ax.set_title(TITLE)
  for tau in tau_list:
    if(tau < 1.1):
      continue
    norm = (tau-tau_min)/(tau_max-tau_min)
    dist_df = pd.read_csv("%s/distribution/T%.3f.csv"%(DATA_PATH,tau))
    dist_ene = np.array(dist_df["Energy"])
    dist_mag = np.array(dist_df["Magnetization"])

    if(tau == tau_list[2]):
      ax.plot(step_list,Corr_func(step_list,dist_ene),"r--",color=cm.jet(norm) ,label="energy")
      ax.plot(step_list,Corr_func(step_list,dist_mag),"b-",color=cm.jet(norm) ,label="magnetization")
    
    ax.plot(step_list,Corr_func(step_list,dist_ene),"r--",color=cm.jet(norm) ,label=r"$\tau$=%.2f"%tau)
    ax.plot(step_list,Corr_func(step_list,dist_mag),"b-",color=cm.jet(norm))
  ax.legend()
  plt.savefig("%s/selfcorretion.png"%SAVE_PATH)

def ucm_compare():
  df_result_heat = pd.read_csv("%s/result.csv"%HEAT_PATH)
  df_result_metr = pd.read_csv("%s/result.csv"%METR_PATH)
  tau = np.array(df_result["T"])

  m_heat = np.array(df_result_heat["m"])
  m_sig_heat = np.array(df_result_heat["m_sig"])
  m_metr = np.array(df_result_metr["m"])
  m_sig_metr = np.array(df_result_metr["m_sig"])

  u_heat = np.array(df_result_heat["e"])
  u_sig_heat = np.array(df_result_heat["e_sig"])
  u_metr = np.array(df_result_metr["e"])
  u_sig_metr = np.array(df_result_metr["e_sig"])

  c_heat = np.array(df_result_heat["c"])
  c_metr = np.array(df_result_metr["c"])

  fig = plt.figure(figsize=(10, 12))
  # fig.subplots_adjust(left=0.2)
  fig.suptitle(r"comparing m,u,c result of heat bath and metropolis method (N$_{spin}=%d\times%d$, N$_{sample}=10^%d$)"%(n,n,log10(Nsample)))
  
  # m のプロット
  ax_m = fig.add_subplot(321,xticks=xticks,xticklabels=xticklabels)
  ax_m_sig = fig.add_subplot(322,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_m)
  set_ax(ax_m_sig)
  ax_m.axvline(x=tau_c,color="r",ls="--")
  ax_m_sig.axvline(x=tau_c,color="r",ls="--")
  ax_m.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_m_sig.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)

  ax_m.set_title("magnetization")
  ax_m_sig.set_title("magnetization error")
  ax_m.set_ylabel(r"magnetiaztion |M|/N$_{\mathrm{spin}}$",fontsize=16)
  ax_m_sig.set_ylabel(r"error $\sigma_m$",fontsize=16)
  ax_m.plot(tau_array, m_ana(tau_array), label="analytic")
  ax_m.errorbar(tau, m_heat, yerr=m_sig_heat, fmt="r.", label="heatbath")
  ax_m.errorbar(tau, m_metr, yerr=m_sig_metr, fmt="y.", label="metropilis")

  # ax_m_sig.plot(tau, abs(m_heat-m_ana(tau)), "r.", label="absolute (heat)")
  # ax_m_sig.plot(tau, abs(m_metr-m_ana(tau)), "y.", label="absolute (metr)")
  ax_m_sig.plot(tau, m_sig_heat, "r*", label="estimated (heat)")
  ax_m_sig.plot(tau, m_sig_metr, "y*", label="estimated (metr)")
  ax_m_sig.set_yscale("log")
  ax_m.legend()
  ax_m_sig.legend()
  
  # uのプロット
  ax_u = fig.add_subplot(323,xticks=xticks,xticklabels=xticklabels)
  ax_u_sig = fig.add_subplot(324,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_u)
  set_ax(ax_u_sig)
  ax_u.axvline(x=tau_c,color="r",ls="--")
  ax_u_sig.axvline(x=tau_c,color="r",ls="--")
  ax_u.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_u_sig.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)

  ax_u.set_title("internal energy")
  ax_u_sig.set_title("internal energy error")
  ax_u.set_ylabel(r"internal energy U/(k$_{\mathrm{B}}$TN$_{\mathrm{spin}}$)",fontsize=16)
  ax_u_sig.set_ylabel(r"error $\sigma_u$",fontsize=16)
  ax_u.plot(tau_array, u_ana(tau_array), label="analytic")
  ax_u.errorbar(tau, u_heat, yerr=u_sig_heat, fmt="r.", label="heatbath")
  ax_u.errorbar(tau, u_metr, yerr=u_sig_metr, fmt="y.", label="metropilis")

  # ax_u_sig.plot(tau, abs(u_heat-u_ana(tau)), "r.", label="absolute (heat)")
  # ax_u_sig.plot(tau, abs(u_metr-u_ana(tau)), "y.", label="absolute (metr)")
  ax_u_sig.plot(tau, u_sig_heat, "r*", label="estimated (heat)")
  ax_u_sig.plot(tau, u_sig_metr, "y*", label="estimated (metr)")
  ax_u_sig.set_yscale("log")
  ax_u.legend()
  ax_u_sig.legend()

  # c のプロット
  ax_c = fig.add_subplot(325,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_c)
  ax_c.set_title("heat capacity")
  ax_c.axvline(x=tau_c,color="r",ls="--")
  ax_c.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_c.set_ylabel(r"heat capacity C/(k$_\mathrm{B}$N$_{\mathrm{spin}}$)",fontsize=16)
  ax_c.plot(tau_array, c_ana(tau_array), label="analytic")
  ax_c.plot(tau, c_heat, "r.", label="heat bath",markersize=7)
  ax_c.plot(tau, c_metr, "y.", label="metrolpolis",markersize=7)
  
  ax_c.legend()

  plt.tight_layout(rect=[0,0,1,0.96])

  plt.savefig("%s/result.png"%COMPARE_PATH)

def m_hist():
  tau_list = np.array(df_result["T"])
  tau_list = sorted(tau_list)
  tau_min = tau_list[0]
  tau_max = tau_list[-1]
  fig = plt.figure(figsize=(8, 8))
  # fig.subplots_adjust(top=0.2)
  # fig.suptitle(r"$\tau$=%.2f %s"%(tau,TITLE))
  ax_mdist = fig.add_subplot(111)
  set_ax(ax_mdist)
  ax_mdist.set_title("magnetization distribution")
  ax_mdist.set_xlabel(r"magnetiaztion M/N$_{\mathrm{spin}}$",fontsize=16)
  ax_mdist.set_ylabel("distribution", fontsize=16)
  for tau in tau_list:
    norm = (tau-tau_min)/(tau_max-tau_min)
    dist_df = pd.read_csv("%s/distribution/T%.3f.csv"%(DATA_PATH,tau))
    dist_mag = dist_df["Magnetization"]

    bins = 101
    ax_mdist.hist(dist_mag, ec="black" ,density=True, bins=bins, align="mid"
      ,range=(-1,1), label=r"$\tau$=%.2f"%tau, color=cm.jet(norm))

  ax_mdist.legend()
  plt.tight_layout(rect=[0,0,1,0.96])
  plt.savefig("%s/distribution/magnetization.png"%(SAVE_PATH))

def u_hist():
  tau_list = np.array(df_result["T"])
  tau_list = sorted(tau_list)
  tau_min = tau_list[0]
  tau_max = tau_list[-1]
  fig = plt.figure(figsize=(8, 8))
  # fig.subplots_adjust(top=0.2)
  # fig.suptitle(r"$\tau$=%.2f %s"%(tau,TITLE))
  ax_udist1 = fig.add_subplot(121)
  ax_udist2 = fig.add_subplot(122)
  set_ax(ax_udist1)
  set_ax(ax_udist2)
  fig.suptitle("internal energy distribution")
  ax_udist1.set_xlabel(r"internal energy U/(k$_{\mathrm{B}}$TN$_{\mathrm{spin}}$)",fontsize=16)
  ax_udist2.set_xlabel(r"internal energy U/(k$_{\mathrm{B}}$TN$_{\mathrm{spin}}$)",fontsize=16)
  ax_udist1.set_title(r"$\tau$=0.1-1.0")
  ax_udist2.set_title(r"$\tau$=2.0-5.0")

  ax_udist1.set_ylabel("distribution", fontsize=16)
  ax_udist2.set_ylabel("distribution", fontsize=16)
  ax_udist2.set_xlim(-1.1,0.1)
  for tau in tau_list:
    norm = (tau-tau_min)/(tau_max-tau_min)
    dist_df = pd.read_csv("%s/distribution/T%.3f.csv"%(DATA_PATH,tau))
    dist_ene = dist_df["Energy"]

    if(tau < 1.9):
      bins = 20
      ax_udist1.hist(dist_ene,density=True, bins=bins, align="mid"
        , label=r"$\tau$=%.2f"%tau, color=cm.jet(norm))
    else:
      ax_udist2.hist(dist_ene,density=True,  align="mid"
      , label=r"$\tau$=%.2f"%tau, color=cm.jet(norm))
    # ax_udist.hist(dist_ene, ec="black" ,density=True, bins=bins, align="mid"
    #   , label=r"$\tau$=%.2f"%tau, color=cm.jet(norm))
    # ax_udist.set_yscale("log")

  # ax_udist1.set_xscale("symlog")
  ax_udist1.set_yscale("log")
  ax_udist1.legend()
  ax_udist2.legend()

  plt.tight_layout(rect=[0,0,1,0.96])
  plt.savefig("%s/distribution/energy.png"%(SAVE_PATH))

def ucm_result():
  df_result_heat = pd.read_csv("%s/result.csv"%HEAT_PATH)
  df_result_metr = pd.read_csv("%s/result.csv"%METR_PATH)
  tau = np.array(df_result["T"])

  m_heat = np.array(df_result_heat["m"])
  m_sig_heat = np.array(df_result_heat["m_sig"])
  m_metr = np.array(df_result_metr["m"])
  m_sig_metr = np.array(df_result_metr["m_sig"])

  u_heat = np.array(df_result_heat["e"])
  u_sig_heat = np.array(df_result_heat["e_sig"])
  u_metr = np.array(df_result_metr["e"])
  u_sig_metr = np.array(df_result_metr["e_sig"])

  c_heat = np.array(df_result_heat["c"])
  c_metr = np.array(df_result_metr["c"])

  fig = plt.figure(figsize=(8, 8))
  # fig.subplots_adjust(left=0.2)
  fig.suptitle(r"m,u,c value of heat bath and metropolis method (N$_{spin}=%d\times%d$, N$_{sample}=10^%d$)"%(n,n,log10(Nsample)))
  
  # m のプロット
  ax_m = fig.add_subplot(221,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_m)
  ax_m.axvline(x=tau_c,color="r",ls="--")
  ax_m.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)

  ax_m.set_title("magnetization")
  ax_m.set_ylabel(r"magnetiaztion |M|/N$_{\mathrm{spin}}$",fontsize=16)
  ax_m.plot(tau_array, m_ana(tau_array), label="analytic")
  ax_m.errorbar(tau, m_heat, yerr=m_sig_heat, fmt="r.", label="heatbath")
  ax_m.errorbar(tau, m_metr, yerr=m_sig_metr, fmt="y.", label="metropilis")

  # ax_m_sig.plot(tau, abs(m_heat-m_ana(tau)), "r.", label="absolute (heat)")
  # ax_m_sig.plot(tau, abs(m_metr-m_ana(tau)), "y.", label="absolute (metr)")
  ax_m.legend()
  
  # uのプロット
  ax_u = fig.add_subplot(222,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_u)
  ax_u.axvline(x=tau_c,color="r",ls="--")
  ax_u.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)

  ax_u.set_title("internal energy")
  ax_u.set_ylabel(r"internal energy U/(k$_{\mathrm{B}}$TN$_{\mathrm{spin}}$)",fontsize=16)
  ax_u.plot(tau_array, u_ana(tau_array), label="analytic")
  ax_u.errorbar(tau, u_heat, yerr=u_sig_heat, fmt="r.", label="heatbath")
  ax_u.errorbar(tau, u_metr, yerr=u_sig_metr, fmt="y.", label="metropilis")

  # ax_u_sig.plot(tau, abs(u_heat-u_ana(tau)), "r.", label="absolute (heat)")
  # ax_u_sig.plot(tau, abs(u_metr-u_ana(tau)), "y.", label="absolute (metr)")
  ax_u.legend()

  # c のプロット
  ax_c = fig.add_subplot(223,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_c)
  ax_c.set_title("heat capacity")
  ax_c.axvline(x=tau_c,color="r",ls="--")
  ax_c.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_c.set_ylabel(r"heat capacity C/(k$_\mathrm{B}$N$_{\mathrm{spin}}$)",fontsize=16)
  ax_c.plot(tau_array, c_ana(tau_array), label="analytic")
  ax_c.plot(tau, c_heat, "r.", label="heat bath",markersize=7)
  ax_c.plot(tau, c_metr, "y.", label="metrolpolis",markersize=7)
  
  ax_c.legend()

  plt.tight_layout(rect=[0,0,1,0.96])

  plt.savefig("%s/ucm_result.png"%COMPARE_PATH)

def ucm_error():
  df_result_heat = pd.read_csv("%s/result.csv"%HEAT_PATH)
  df_result_metr = pd.read_csv("%s/result.csv"%METR_PATH)
  tau = np.array(df_result["T"])

  m_heat = np.array(df_result_heat["m"])
  m_sig_heat = np.array(df_result_heat["m_sig"])
  m_metr = np.array(df_result_metr["m"])
  m_sig_metr = np.array(df_result_metr["m_sig"])

  u_heat = np.array(df_result_heat["e"])
  u_sig_heat = np.array(df_result_heat["e_sig"])
  u_metr = np.array(df_result_metr["e"])
  u_sig_metr = np.array(df_result_metr["e_sig"])


  fig = plt.figure(figsize=(8, 8))
  # fig.subplots_adjust(left=0.2)
  fig.suptitle(r"comparing m,u,c error of heat bath and metropolis method (N$_{spin}=%d\times%d$, N$_{sample}=10^%d$)"%(n,n,log10(Nsample)))
  
  # sigma_m_heat のプロット
  ax_msig_heat = fig.add_subplot(221,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_msig_heat)
  ax_msig_heat.axvline(x=tau_c,color="r",ls="--")
  ax_msig_heat.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_msig_heat.set_yscale("log")
  ax_msig_heat.set_ylim(10**-6,1)

  ax_msig_heat.set_title("magnetization error (heat bath)")
  ax_msig_heat.set_ylabel(r"$\sigma_m$",fontsize=16)
  ax_msig_heat.plot(tau, m_sig_heat, "r.", label="estimated error")
  ax_msig_heat.plot(tau, abs(m_heat-m_ana(tau)), "b.", label="absolute error")
  ax_msig_heat.legend()

    # sigma_m_metr のプロット
  ax_msig_metr = fig.add_subplot(222,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_msig_metr)
  ax_msig_metr.axvline(x=tau_c,color="r",ls="--")
  ax_msig_metr.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_msig_metr.set_yscale("log")
  ax_msig_metr.set_ylim(10**-6,1)

  ax_msig_metr.set_title("magnetization error (metropolis)")
  ax_msig_metr.set_ylabel(r"$\sigma_m$",fontsize=16)
  ax_msig_metr.plot(tau, m_sig_metr, "r.", label="estimated error")
  ax_msig_metr.plot(tau, abs(m_metr-m_ana(tau)), "b.", label="absolute error")
  ax_msig_metr.legend()
  
    # sigma_u_heat のプロット
  ax_usig_heat = fig.add_subplot(223,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_usig_heat)
  ax_usig_heat.axvline(x=tau_c,color="r",ls="--")
  ax_usig_heat.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_usig_heat.set_yscale("log")
  ax_usig_heat.set_ylim(10**-5,5*10**-2)

  ax_usig_heat.set_title("internal energy error (heat bath)")
  ax_usig_heat.set_ylabel(r"$\sigma_u$",fontsize=16)
  ax_usig_heat.plot(tau, u_sig_heat, "r.", label="estimated error")
  ax_usig_heat.plot(tau, abs(u_heat-u_ana(tau)), "b.", label="absolute error")
  ax_usig_heat.legend()

    # sigma_m_metr のプロット
  ax_usig_metr = fig.add_subplot(224,xticks=xticks,xticklabels=xticklabels)
  set_ax(ax_usig_metr)
  ax_usig_metr.axvline(x=tau_c,color="r",ls="--")
  ax_usig_metr.set_xlabel(r"temperature $\tau$ (= k$_{\mathrm{B}}$T/J)",fontsize=16)
  ax_usig_metr.set_yscale("log")
  ax_usig_metr.set_ylim(10**-5,5*10**-2)

  ax_usig_metr.set_title("internal energy error (metropolis)")
  ax_usig_metr.set_ylabel(r"$\sigma_u$",fontsize=16)
  ax_usig_metr.plot(tau, u_sig_metr, "r.", label="estimated error")
  ax_usig_metr.plot(tau, abs(u_metr-u_ana(tau)), "b.", label="absolute error")
  ax_usig_metr.legend()



  plt.tight_layout(rect=[0,0,1,0.96])

  plt.savefig("%s/ucm_error.png"%COMPARE_PATH)

def all_plot():
  u_plot()
  c_plot()
  m_plot()
  dist_plot()
  Self_correlation()
  ucm_compare()
  m_hist()
  u_hist()
  ucm_result()

# u_plot()
# c_plot()
# m_plot()
# dist_plot()
# Self_correlation()
# ucm_compare()
# m_hist()
# u_hist()
# ucm_result()
ucm_error()
# K_E_compare()

# all_plot()

if(PLT == 1):
  plt.show()