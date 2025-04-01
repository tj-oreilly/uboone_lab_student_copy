import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

DIMS = (100, 100)

def plot_exclusion_line(chi_squared_file, colour, label, legend_proxies):

  cache_chi_squareds = None
  with open(chi_squared_file, "rb") as file:
    cache_chi_squareds = pickle.load(file)

  cache_min_chi = np.min(cache_chi_squareds)

  delta_m2 = np.logspace(-2, 2, DIMS[0])
  sin2theta = np.logspace(-3, 0, DIMS[1])
  xv, yv = np.meshgrid(delta_m2, sin2theta)

  # 4-flavour scaling
  scaled_theta = (1 - np.sqrt(1 - yv)) * (1 - np.sqrt(1 - 0.24))

  plt.contour(
    scaled_theta,
    xv,
    cache_chi_squareds,
    levels=[float(cache_min_chi) + 4.61],  # 90% confidence
    colors=[colour],
  )

  legend_proxies.append(plt.Line2D([0], [0], linestyle="solid", color=colour, label=label))


def main():

  # White font
  plt.rcParams["text.color"] = "white"
  plt.rcParams["axes.labelcolor"] = "white"
  plt.rcParams["xtick.color"] = "white"
  plt.rcParams["ytick.color"] = "white"
  plt.rcParams["axes.titlecolor"] = "white"
 
  # Create figure

  fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
  ax.set_facecolor("black")
  for spine in ax.spines.values():
     spine.set_color("white")

  LSND_data = pd.read_csv("./data/DataSet_LSND.csv").to_numpy()
  MiniBooNE_data = pd.read_csv("./data/DataSet_MiniBooNE.csv").to_numpy()

  # Plot data
  ax.plot(LSND_data[:, 0], LSND_data[:, 1], "o")
  ax.plot(MiniBooNE_data[:, 0], MiniBooNE_data[:, 1], "o")

  # Producing MiniBooNE/LSND legend
  LSND_path = mpatches.Patch(color="tab:blue", label="LSND 90%")
  MINI_path = mpatches.Patch(color="tab:orange", label="MiniBooNE 90%")

  del_chi2 = 4.61  # 90% confidence level

  # Plot cached contour
  proxy_lines = []

  plot_exclusion_line("./cache/chi_squared_rf.pkl", "red", "Random Forest", proxy_lines)
  plot_exclusion_line("./cache/chi_squared_nn.pkl", "lime", "Neural Network", proxy_lines)
  plot_exclusion_line("./cache/chi_squared_no_cut.pkl", "gray", "No ML", proxy_lines)

  first_legend = plt.legend(
      handles=[LSND_path, MINI_path] + proxy_lines, loc="upper right", fontsize=12, facecolor="black"
  )
  plt.gca().add_artist(first_legend)

  plt.xlabel(
      r"$\sin^2(2\theta_{\mu_e})$", fontsize=12
  )
  plt.ylabel(r"$\Delta$ $m_{14}^2$", fontsize=12)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.yscale("log")
  plt.xscale("log")
  plt.savefig("figure.svg", format="svg", transparent=True)
  plt.show()


if __name__ == "__main__":
  main()