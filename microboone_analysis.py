# Comparing the data from MicroBooNE to MiniBooNE and LSND

import time, math, os
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

CACHING = False
FOREST_CACHE = "./cache/random_forest_cache.pkl"
BINS = 20

def ApplySelection(selection, frame, filtered_frame, purity_efficiency, totalCount):
  """
  Single selection on the dataset, calculating the purity and efficiency of the cut.
  """

  new_frame = frame[selection]

  signalCount = len(new_frame[(new_frame['category'] == 21) | (new_frame['category'] == 10)])

  purity = signalCount / len(new_frame)
  efficiency = len(new_frame) / totalCount

  if purity_efficiency != None:
    purity_efficiency.append((purity, efficiency))

  return new_frame

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
  
  
  selection_cuts = [
    (frame['trk_len_v'] > -1000.0) & (frame['trk_len_v'] < 1000.0), # Base cuts
    (frame['trk_energy_tot'] < 5000.0),  # Non-physical values
    (frame['trk_energy_tot'] < 2.0),
    (frame['topological_score'] > 0.4),
    (frame['trk_distance_v'] < 5.0),
    (frame['reco_nu_vtx_sce_x'] > 50.0),
    (frame['reco_nu_vtx_sce_x'] < 200.0),
  ]

  purity_efficiency = []  # Store the purity and efficiency for each consecutive cut
  total_count = len(frame)

  combined_cut = pd.Series(True, index=frame.index)
  filtered_frame = frame

  for cut in selection_cuts:
    combined_cut = combined_cut & cut
    filtered_frame = ApplySelection(combined_cut, frame, filtered_frame, purity_efficiency, total_count)

  print("Purity and efficiency:")
  for pe in purity_efficiency:
    print(pe)

  if not show_plots:
    return filtered_frame

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

def train_random_forest(MC_data, features):
  """
  """

  MC_data = MC_data[(MC_data.category != 21) & (MC_data.category != 10)].copy(deep=True) # Remove muon and electron events?
  
  # Setting up input parameters for random forest.
  X = MC_data[features]
  y = np.array(MC_data['category'])

  # Then split the data up into a "training set" and "test set" using train_test_split.
  # Keep the random_state=1 in your arguments
  x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1) # 80/20 training/test split

  # NOTE: Could improve the training data by equally splitting it between each event type
  # 7, 4, 5, 31
  # Combine EXT and cosmic - 7, 4 as they should be similar

  count = {}

  for index, id in enumerate(y_test):
      if id == 4:
          y_test[index] = 7

  # Get event type with minimum data points
  for index, id in enumerate(y_train):
      if id == 4:
          y_train[index] = 7
          id = 7

      if id not in count:
          count[id] = 0
      count[id] += 1

  min_count = -1
  for id in count:
      if count[id] < min_count or min_count == -1:
          min_count = count[id]
      count[id] = 0

  # Remove extra samples
  removeIndices = []
  for index, id in enumerate(y_train):
      if count[id] >= min_count:
          removeIndices.append(index)
      count[id] += 1


  x_train = np.delete(x_train, removeIndices, 0)
  y_train = np.delete(y_train, removeIndices)

  # Random forest training
  start_time = time.time()
  print("Training Random Forest... ", end="")
  rf = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=8, criterion="gini")
  rf.fit(x_train, y_train)

  time_taken = time.time() - start_time
  print(f"Finished in {time_taken:.2f} seconds")

  # Produce predictions for the classification of your training dataset using your model:
  y_pred = rf.predict(x_train)

  # Show the accuracies of said predictions
  print("Accuracy on training dataset: ",metrics.accuracy_score(y_train, y_pred))
  rf_acc_train = metrics.accuracy_score(y_train, y_pred)
  y_pred = rf.predict(x_test)

  print("Accuracy on testing dataset: ",metrics.accuracy_score(y_test, y_pred))
  rf_acc_test = metrics.accuracy_score(y_test, y_pred)

  return rf

def categorise_data(MC_data, data):
  """
  Trains a ML algorithm to categorise the type of event based on the track data. The training input 
  is the Monte-Carlo data.
  """

  features = ['_closestNuCosmicDist', 'trk_len_v', 'trk_distance_v', 'topological_score', 'trk_sce_end_z_v', 'trk_sce_end_y_v', 'trk_sce_end_x_v', 'trk_score_v', 'trk_llr_pid_score_v', 'trk_sce_start_z_v', 'trk_sce_start_y_v', 'trk_sce_start_x_v', 'reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z', 'trk_energy_tot']
  output = ['category']

  # Try to load the cached Random Forest
  rf = None

  if CACHING and os.path.exists(FOREST_CACHE):
    with open(FOREST_CACHE, 'rb') as file:
      print("Found a cached random forest model. Loading into memory...")
      rf = pickle.load(file)
  else:
    rf = train_random_forest(MC_data, features) 

    if CACHING:
      if not os.path.exists("./cache"):
        os.mkdir("./cache")

      with open(FOREST_CACHE, 'wb') as file:
        print("Caching the model for future use")
        pickle.dump(rf, file)

  # Now categorise the real data
  X = data[features]
  category_pred = rf.predict(X)

  data["category"] = category_pred

  return data

def plot_histograms(data):
  """
  Plot histograms for how the counts are layed out in each track variable.
  """

  HISTOGRAM_BINS = 30

  plot_vars = list(data.columns)
  n_plots = len(plot_vars)
  
  # Determine layout
  cols = min(math.floor(math.sqrt(n_plots)), n_plots) # Square layout
  rows = math.ceil(n_plots / cols)
  
  figsize = (10 * cols, 10 * rows)

  # Create figure
  fig, axes = plt.subplots(rows, cols, figsize=figsize)
  
  # Flatten axes array for easy iteration
  if n_plots > 1:
    axes = axes.flatten()
  else:
    axes = [axes]
  
  # Plot each histogram
  for i, var in enumerate(plot_vars):
    if i < n_plots:
      current_weights = data['weight'] if 'weight' in data else [1.0 for i in range(len(data))] 

      use_legend = False  # (i == cols - 1)
      
      # Plot on the current subplot
      ax = sns.histplot(
        data=data, 
        x=var, 
        multiple="stack", 
        hue="category", 
        palette='deep', 
        weights=current_weights, 
        bins=HISTOGRAM_BINS, 
        legend=use_legend,
        ax=axes[i]
      )
      
      # Set labels
      ax.set(xlabel=var, ylabel="Events")
      
      # Set xlim to min and max of the data
      ax.set_xlim([np.min(data[var]), np.max(data[var])])
      
      if use_legend:
        ax.legend(
          title='Category',
          fontsize=12, 
          loc='upper right',
          labels=[r"$\nu$ NC", r"$\nu_{\mu}$ CC", r"$\nu_e$ CC", r"EXT", r"Out. fid. vol.", r"mis ID"]
        )

      pos = ax.get_position()
      ax.set_position([pos.x0, pos.y0, pos.width, pos.width * 0.7])
  
  # Hide any unused subplots
  for i in range(n_plots, rows * cols):
    axes[i].set_visible(False)

  handles, labels = axes[0].get_legend_handles_labels()

  fig.legend(handles, 
            labels=[r"$\nu$ NC", r"$\nu_{\mu}$ CC", r"$\nu_e$ CC", r"EXT", r"Out. fid. vol.", r"mis ID"],
            title='Category',
            fontsize=12,
            loc='center right',  # Position on the right
            bbox_to_anchor=(0.85, 0.5))  # Adjust as needed - (x, y) position
    
  # Add overall title
  plt.suptitle("Data from all variables", fontsize=16, y=0.98)
  
  # Adjust the padding to prevent cutting off axis labels and ticks
  plt.tight_layout(pad=1.0, h_pad=15.0, w_pad=10.0, rect=[0, 0, 1, 0.95])

  plt.show()

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
  data["category"] = [21 for i in range(len(data))]
  data = categorise_data(MC_data, data)

  # Apply selection cuts
  data = Selections(data, False)
  MC_data = Selections(MC_data, False)

  # Only include muon events now
  MC_data = MC_data[MC_data["category"] == 21]
  data = data[data["category"] == 21]

  print(len(data))

  # Plot the data after the selection cuts
  # plot_histograms(data)

  # Oscillate Monte-Carlo data and calculate chi squared against real data
  # Assume a 2-flavour oscillation first.
  # We will scale the theta value after to simulate 4-flavour

  DIMS = (50,50)

  # Grid for chi square calculation
  delta_m2 = np.logspace(-2, 2, DIMS[0])
  sin2theta = np.logspace(-3, 0, DIMS[1])
  xv, yv = np.meshgrid(delta_m2, sin2theta)

  chi_squared_values = np.zeros(DIMS)
  MC_data["dis_prob"] = 1.0

  min_chi_squared = 1000.0
  min_loc = None

  print("Populating chi squared values...")
  percent = 0

  for i,x in enumerate(sin2theta):
    for j,y in enumerate(delta_m2):
      
      # Disappearance probability
      true_E = np.array(MC_data.loc[MC_data["category"] == 21, "true_E"])
      MC_data.loc[MC_data["category"] == 21, "dis_prob"] = 1.0 - oscillation_probability(x, y, 0.475, true_E)

      chi_squared_values[i,j] = get_chi_squared(MC_data, data)

      if chi_squared_values[i,j] < min_chi_squared:
        min_chi_squared = chi_squared_values[i,j]
        min_loc = (x,y)

    current_percent = int((i * 100) / DIMS[1])
    if current_percent > percent:
      percent = current_percent
      print(f"{percent}%")

  print(f"Minimum (theta, m): {min_loc}")
  print(f"Minimum chi squared: {min_chi_squared}")

  # Scale the \sin^2{\theta_{12}} value into \sin^2{\theta_{14}} in the 4-flavour model
  scaled_theta = (1 - np.sqrt(1 - yv)) * (1 - np.sqrt(1 - 0.24))

  fig, ax = plt.subplots()

  plt.contourf(scaled_theta, xv, chi_squared_values)
  plt.colorbar()

  ax.set_xscale("log")
  ax.set_yscale("log")

  plt.ylabel(r"$\Delta m_{12}^2$")
  plt.xlabel(r"$\sin^2{(2\theta_{12})}$")
  plt.title(r"A contour plot of the $\chi^2$ value")
  plt.show()

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

  del_chi2 = 4.61 # 90% confidence level
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