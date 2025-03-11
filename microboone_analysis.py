# Comparing the data from MicroBooNE to MiniBooNE and LSND

import numpy as np
import uproot as uproot3
import pickle

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import Neutrino_functions

from math import *
import scipy as sci

import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

BINS = 20

"""Single selection on the dataset"""
def ApplySelection(selection, frame, purity_efficiency, totalCount):

  frame = frame[selection]

  signalCount = len(frame[(frame['category'] == 21) | (frame['category'] == 10)])

  purity = signalCount / len(frame)
  efficiency = len(frame) / totalCount

  if purity_efficiency != None:
    purity_efficiency.append((purity, efficiency))

  return frame

def Selections(frame, show_plots=True):
    
  # Basic variables present in dataframe 
  trk_start_x_v = frame['trk_sce_start_x_v']        # cm
  trk_start_y_v = frame['trk_sce_start_y_v']        # cm
  trk_start_z_v = frame['trk_sce_start_z_v']        # cm
  trk_end_x_v = frame['trk_sce_end_x_v']            # cm
  trk_end_y_v = frame['trk_sce_end_y_v']            # cm
  trk_end_z_v = frame['trk_sce_end_z_v']            # cm
  reco_x = frame['reco_nu_vtx_sce_x']               # cm
  reco_y = frame['reco_nu_vtx_sce_y']               # cm
  reco_z = frame['reco_nu_vtx_sce_z']               # cm
  topological = frame['topological_score']          # N/A
  trk_score_v = frame['trk_score_v']                # N/A
  trk_dis_v = frame['trk_distance_v']               # cm
  trk_len_v = frame['trk_len_v']                    # cm
  trk_energy_tot = frame['trk_energy_tot']          # GeV 
  
  # Select the conditions you want to apply, here is an initial condition to get you started.
  purity_efficiency = []
  totalCount = len(frame)

  frame = ApplySelection(((frame['trk_len_v'] > -1000) & (frame['trk_len_v'] < 1000)), frame, None, totalCount)
  frame = ApplySelection((frame['trk_energy_tot'] < 5000.0), frame, None, totalCount)

  frame = ApplySelection((frame['trk_energy_tot'] < 2.0), frame, purity_efficiency, totalCount)
  frame = ApplySelection((frame['topological_score'] > 0.4), frame, purity_efficiency, totalCount)
  frame = ApplySelection((frame['trk_distance_v'] < 5), frame, purity_efficiency, totalCount)
  frame = ApplySelection((frame['reco_nu_vtx_sce_x'] > 50), frame, purity_efficiency, totalCount)
  frame = ApplySelection((frame['reco_nu_vtx_sce_x'] < 200), frame, purity_efficiency, totalCount)

  if not show_plots:
    return frame

  # # Plot purity
  # purity_values = [val[0] for val in purity_efficiency]

  # plt.plot([(i + 1) for i in range(len(purity_efficiency))], purity_values, marker='o', linestyle='-')
  # plt.xlabel("Selection Cut")
  # plt.ylabel(f"Purity")
  # plt.title(f"Purity change over cuts")
  # plt.show()

  # # Efficiency
  # eff_values = [val[1] for val in purity_efficiency]

  # plt.plot([(i + 1) for i in range(len(purity_efficiency))], eff_values, marker='o', linestyle='-')
  # plt.xlabel("Selection Cut")
  # plt.ylabel(f"Efficiency")
  # plt.title(f"Efficiency change over cuts")
  # plt.show()

  return frame


"""
Length: km
Energy: GeV
"""
def oscillation_probability(sin2theta, delta_m2, length, energy):
  return sin2theta * np.sin(1.27 * delta_m2 * length / energy) ** 2

"""Calculates the chi squared of the energy data fit"""
def get_chi_squared(sim_data, real_data):

  real_weights = None 
  if "weight" in real_data:
    real_weights = real_data["weight"]

  # Bin the data
  sim_hist, _ = np.histogram(sim_data["trk_energy_tot"], BINS, 
                             weights=sim_data["weight"]*sim_data["dis_prob"])
  
  real_hist, _ = np.histogram(real_data["trk_energy_tot"], BINS, weights=real_weights)

  # Calculate chi squared
  chi_squared = 0.0
  for i,_ in enumerate(sim_hist):
    chi_squared += ((sim_hist[i] - real_hist[i])**2) / ((sim_hist[i] * 0.15)**2)

  return chi_squared

def main():

  # Load MicroBooNE data 
  data_file = './data/data_flattened.pkl'
  data = pd.read_pickle(data_file)
  data = data.drop('Subevent', axis = 1)

  # Load Monte-Carlo data
  MC_file = './data/MC_EXT_flattened.pkl'
  MC_data = pd.read_pickle(MC_file)
  MC_data = MC_data.drop('Subevent', axis = 1)

  # Random forest training to improve cuts

  # Apply selection cuts
  data = Selections(data, False)
  MC_data = Selections(MC_data, False)

  # Oscillate Monte-Carlo data and calculate chi squared against real data
  # Assume a 2-flavour oscillation first.
  # We will scale the theta value after to simulate 4-flavour

  DIMS = (100,100)

  # Grid for chi square calculation
  delta_m2 = np.logspace(-2, 2, DIMS[0])
  sin2theta = np.logspace(-3, 0, DIMS[1])
  xv, yv = np.meshgrid(delta_m2, sin2theta)

  chi_squared_values = np.zeros(DIMS)
  MC_data["dis_prob"] = 1.0

  min_chi_squared = 1000.0
  min_loc = None

  for i,x in enumerate(sin2theta):
    for j,y in enumerate(delta_m2):
      # Disappearance probability

      dis_prob = MC_data.loc[MC_data["category"] == 21, "dis_prob"]
      true_E = np.array(MC_data.loc[MC_data["category"] == 21, "true_E"])

      dis_prob = 1.0 - oscillation_probability(x, y, 0.475, true_E)

      chi_squared_values[i,j] = get_chi_squared(MC_data, data)

      if chi_squared_values[i,j] < min_chi_squared:
        min_chi_squared = chi_squared_values[i,j]
        min_loc = (x,y)

  print(f"Minimum (theta, m): {min_loc}")
  print(f"Minimum chi squared: {min_chi_squared}")

  # Scale the \sin^2{\theta_{12}} value into \sin^2{\theta_{14}} in the 4-flavour model
  scaled_theta = (1 - np.sqrt(1 - yv)) * (1 - np.sqrt(1 - 0.24))

  # fig, ax = plt.subplots()

  # plt.contourf(scaled_theta, xv, chi_squared_values)
  # plt.colorbar()

  # ax.set_xscale("log")
  # ax.set_yscale("log")

  # plt.ylabel(r"$\Delta m_{12}^2$")
  # plt.xlabel(r"$\sin^2{(2\theta_{12})}$")
  # plt.title(r"A contour plot of the $\chi^2$ value")
  # plt.show()

  # Load data
  LSND_data = pd.read_csv('./data/DataSet_LSND.csv').to_numpy()
  MiniBooNE_data = pd.read_csv('./data/DataSet_MiniBooNE.csv').to_numpy()

  # Plot data
  plt.plot(LSND_data[:,0],LSND_data[:,1],'o')
  plt.plot(MiniBooNE_data[:,0],MiniBooNE_data[:,1],'o')

  # Producing MiniBooNE/LSND legend
  LSND_path = mpatches.Patch(color='tab:blue', label = 'LSND')
  MINI_path = mpatches.Patch(color='tab:orange', label = 'MiniBooNE')
  first_legend = plt.legend(handles=[LSND_path, MINI_path], loc = 'lower left', fontsize = 12)
  plt.gca().add_artist(first_legend)

  del_chi2 = 4.61
  contour = plt.contour(scaled_theta, xv, chi_squared_values, 
                        levels=[min_chi_squared + del_chi2], colors=["red"])

  plt.xlabel(r'$sin^2(2\theta_{\mu e})=sin^2(\theta_{24})sin^2(2\theta_{14})$',fontsize=20)
  plt.ylabel(r'$\Delta$ $m_{14}^2$',fontsize=20)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.yscale('log')
  plt.xscale('log')
  plt.show()

  return 0

if __name__ == "__main__":
  main()