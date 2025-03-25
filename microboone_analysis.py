# Comparing the data from MicroBooNE to NE and LSND

import time
import math
import os
import numpy as np
import pickle

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

CACHING = True
FOREST_CACHE = "./cache/random_forest_cache.pkl"
BINS = 20


def apply_selection(selection, frame, filtered_frame, purity_efficiency, total_count):
    """
    Single selection on the dataset, calculating the purity and efficiency of the cut.
    """

    new_frame = frame[selection]

    if "category" not in new_frame:
        return new_frame

    signal_count = len(
        new_frame[(new_frame["category"] == 21) | (new_frame["category"] == 10)]
    )

    purity = signal_count / len(new_frame)
    efficiency = len(new_frame) / total_count

    if purity_efficiency is not None:
        purity_efficiency.append((purity, efficiency))

    return new_frame


def Selections(frame, show_plots=True):
    selection_cuts = [
        (frame["trk_len_v"] > -1000.0) & (frame["trk_len_v"] < 1000.0),  # Base cuts
        (frame["trk_energy_tot"] < 5000.0),  # Non-physical values
        (frame["trk_energy_tot"] < 4.0),
        (frame["topological_score"] > 0.4),
        (frame["trk_distance_v"] < 100.0),
        (frame["trk_range_muon_mom_v"] < 2.0),
        (frame["trk_mcs_muon_mom_v"] < 2.0),
        (frame["trk_score_v"] > -0.1),
        (frame["_closestNuCosmicDist"] > 0.1) & (frame["_closestNuCosmicDist"] < 450.0),
        (frame["trk_sce_start_x_v"] > 10.0) & (frame["trk_sce_start_x_v"] < 240.0),
        (frame["trk_sce_start_y_v"] > -100.0) & (frame["trk_sce_start_y_v"] < 100.0),
        (frame["trk_sce_end_x_v"] > 0.0) & (frame["trk_sce_end_x_v"] < 250.0),
        (frame["trk_sce_end_y_v"] > -150.0) & (frame["trk_sce_end_y_v"] < 150.0),
    ]

    purity_efficiency = []  # Store the purity and efficiency for each consecutive cut
    total_count = len(frame)

    combined_cut = pd.Series(True, index=frame.index)
    filtered_frame = frame

    for cut in selection_cuts:
        combined_cut = combined_cut & cut
        filtered_frame = apply_selection(
            combined_cut, frame, filtered_frame, purity_efficiency, total_count
        )

    # print("Purity and efficiency:")
    # for pe in purity_efficiency:
    #   print(pe)

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
    sim_hist, _ = np.histogram(
        sim_data["trk_energy_tot"],
        BINS,
        weights=sim_data["weight"] * sim_data["dis_prob"],
    )

    real_hist, _ = np.histogram(real_data["trk_energy_tot"], BINS, weights=real_weights)

    # Calculate chi squared
    chi_squared = 0.0
    for i, _ in enumerate(sim_hist):
        chi_squared += ((sim_hist[i] - real_hist[i]) ** 2) / ((sim_hist[i] * 0.15) ** 2)

    return chi_squared


def train_random_forest(MC_data, features):
    """
    Trains a random forest based on the Monte Carlo dataset.
    """

    # Setting up input parameters for random forest.
    X = MC_data[features]
    y = MC_data["category"]

    # Combined cosmic and ext
    y[y == 4] = 7

    # Then split the data up into a "training set" and "test set" using train_test_split.
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # count = {}

    # # Get event type with minimum data points
    # for index, id in enumerate(y_train):
    #     if id == 4:
    #         y_train[index] = 7
    #         id = 7
    #
    #     if id not in count:
    #         count[id] = 0
    #     count[id] += 1
    #
    # min_count = -1
    # for id in count.keys():
    #     if count[id] < min_count or min_count == -1:
    #         min_count = count[id]
    #     count[id] = 0
    #
    # # Remove extra samples
    # removeIndices = []
    # for index, id in enumerate(y_train):
    #     if count[id] >= min_count:
    #         removeIndices.append(index)
    #     count[id] += 1
    #
    # x_train = np.delete(x_train, removeIndices, 0)
    # y_train = np.delete(y_train, removeIndices)

    # Random forest training
    start_time = time.time()
    print("Training Random Forest... ", end="")
    rf = RandomForestClassifier(
        random_state=1,
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        criterion="gini",
    )
    rf.fit(x_train, y_train)

    time_taken = time.time() - start_time
    print(f"Finished in {time_taken:.2f} seconds")

    # Produce predictions for the classification of your training dataset using your model:
    y_pred = rf.predict(x_train)

    # Show the accuracies of said predictions
    print("Accuracy on training dataset: ", metrics.accuracy_score(y_train, y_pred))

    y_pred = rf.predict(x_test)
    print("Accuracy on testing dataset: ", metrics.accuracy_score(y_test, y_pred))

    return rf


def categorise_data_forest(MC_data, data):
    """
    Trains a ML algorithm to categorise the type of event based on the track data. The training input
    is the Monte-Carlo data.
    """

    features = [
        "_closestNuCosmicDist",
        "trk_len_v",
        "trk_distance_v",
        "topological_score",
        "trk_sce_end_z_v",
        "trk_sce_end_y_v",
        "trk_sce_end_x_v",
        "trk_score_v",
        "trk_llr_pid_score_v",
        "trk_sce_start_z_v",
        "trk_sce_start_y_v",
        "trk_sce_start_x_v",
        "reco_nu_vtx_sce_x",
        "reco_nu_vtx_sce_y",
        "reco_nu_vtx_sce_z",
        "trk_energy_tot",
    ]

    # Try to load the cached Random Forest
    rf = None

    if CACHING and os.path.exists(FOREST_CACHE):
        with open(FOREST_CACHE, "rb") as file:
            print("Found a cached random forest model. Loading into memory...")
            rf = pickle.load(file)
    else:
        rf = train_random_forest(MC_data, features)

        if CACHING:
            if not os.path.exists("./cache"):
                os.mkdir("./cache")

            with open(FOREST_CACHE, "wb") as file:
                print("Caching the model for future use")
                pickle.dump(rf, file)

    # Now categorise the real data
    X = data[features]
    category_pred = rf.predict(X)

    data["background"] = category_pred != 21

    # Categorise MC data
    X = MC_data[features]
    category_pred = rf.predict(X)

    MC_data["background"] = category_pred != 21

    return data, MC_data


def plot_histograms(data):
    """
    Plot histograms for how the counts are layed out in each track variable.
    """

    HISTOGRAM_BINS = 30

    ignore_cols = ["category", "weight", "true_E", "true_L", "true_muon_mom"]
    plot_vars = list(data.columns)

    n_plots = len(plot_vars) - len(ignore_cols)

    # Determine layout as a square
    cols = min(math.floor(math.sqrt(n_plots)), n_plots)
    rows = math.ceil(n_plots / cols)

    figsize = (8 * cols, 8 * rows)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot each histogram
    plot_index = 0
    for _, var in enumerate(plot_vars):
        if var in ignore_cols:
            continue

        current_weights = (
            data["weight"] if "weight" in data else [1.0 for i in range(len(data))]
        )

        # Plot on the current subplot
        ax = sns.histplot(
            data=data,
            x=var,
            multiple="stack",
            hue="category",
            palette="deep",
            weights=current_weights,
            bins=HISTOGRAM_BINS,
            legend=False,
            ax=axes[plot_index],
        )

        # Set labels
        ax.set(xlabel=var, ylabel="Events")

        # Set xlim to min and max of the data
        ax.set_xlim([np.min(data[var]), np.max(data[var])])

        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.width * 0.7])

        plot_index += 1

    # Hide any unused subplots
    for i in range(n_plots, rows * cols):
        axes[i].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        labels=[
            r"$\nu$ NC",
            r"$\nu_{\mu}$ CC",
            r"$\nu_e$ CC",
            r"EXT",
            r"Out. fid. vol.",
            r"mis ID",
        ],
        title="Category",
        fontsize=12,
        loc="upper right",  # Position on the right
        bbox_to_anchor=(1.0, 1.0),
    )

    plt.suptitle("Data from all Monte Carlo variables", fontsize=16, y=0.98)
    plt.tight_layout(pad=1.0, h_pad=15.0, w_pad=10.0, rect=[0, 0.1, 0.9, 0.95])
    plt.show()


def categorise_data(MC_data, data):
    import neural_network

    MC_data = MC_data.copy(deep=True)
    data = data.copy(deep=True)

    nn = neural_network.NeuralNetwork(MC_data, data)
    nn.prepare_data()  # Restructure columns of data

    # Define features and target
    X = nn.MC_data.drop(
        columns=["category", "weight"]
    )  # Don't use category or weight for neural net
    y = nn.MC_data["category"].copy()
    mc_weights = nn.MC_data["weight"].copy()

    feature_names = X.columns.tolist()
    nn.train_evaluate_model(X, y, mc_weights, feature_names)

    nn.categorise_data()

    return nn.data, nn.MC_data


def main():
    # Load MicroBooNE data
    data_file = "./data/data_flattened.pkl"
    data = pd.read_pickle(data_file)
    data = data.drop("Subevent", axis=1)

    # Load Monte-Carlo data
    MC_file = "./data/MC_EXT_flattened.pkl"
    MC_data = pd.read_pickle(MC_file)
    MC_data = MC_data.drop("Subevent", axis=1)

    # Apply selection cuts
    initial_MC_len = len(MC_data)

    data = Selections(data, False)
    MC_data = Selections(MC_data, False)

    print(f"Efficiency of inital cuts: {len(MC_data) / float(initial_MC_len)}")
    print(
        f"Purity of initial cuts: {float(len(MC_data[MC_data['category'] == 21])) / len(MC_data)}"
    )

    # plot_histograms(MC_data)

    MC_copy = MC_data.copy(deep=True)

    # Neural network training to improve cuts
    data, MC_data = categorise_data_forest(MC_data, data)

    # Copy back data removed for nn training
    MC_data["true_E"] = MC_copy["true_E"]
    MC_data["weight"] = MC_copy["weight"]
    MC_data["category"] = MC_copy["category"]

    # Only include muon events after cuts
    initial_MC_len = len(MC_data)

    MC_data = MC_data[~MC_data["background"]]
    data = data[~data["background"]]

    print(f"Efficiency of nn cuts: {(len(MC_data) / float(initial_MC_len)):.2f}")
    print(
        f"Purity of nn cuts: {(len(MC_data[MC_data['category'] == 21]) / len(MC_data))}"
    )

    # Plot the data after the selection cuts
    # plot_histograms(data)

    # Oscillate Monte-Carlo data and calculate chi squared against real data
    # Assume a 2-flavour oscillation first.
    # We will scale the theta value after to simulate 4-flavour

    DIMS = (50, 50)

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

    for i, x in enumerate(sin2theta):
        for j, y in enumerate(delta_m2):
            # Disappearance probability
            true_E = np.array(MC_data["true_E"])
            MC_data["dis_prob"] = 1.0 - oscillation_probability(x, y, 0.475, true_E)

            chi_squared_values[i, j] = get_chi_squared(MC_data, data)

            if chi_squared_values[i, j] < min_chi_squared:
                min_chi_squared = chi_squared_values[i, j]
                min_loc = (x, y)

        current_percent = int((i * 100) / DIMS[1])
        if current_percent > percent:
            percent = current_percent
            # print(f"{percent}%")

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
    LSND_data = pd.read_csv("./data/DataSet_LSND.csv").to_numpy()
    MiniBooNE_data = pd.read_csv("./data/DataSet_MiniBooNE.csv").to_numpy()

    # Plot data
    plt.plot(LSND_data[:, 0], LSND_data[:, 1], "o")
    plt.plot(MiniBooNE_data[:, 0], MiniBooNE_data[:, 1], "o")

    # Producing MiniBooNE/LSND legend
    LSND_path = mpatches.Patch(color="tab:blue", label="LSND")
    MINI_path = mpatches.Patch(color="tab:orange", label="MiniBooNE")
    first_legend = plt.legend(
        handles=[LSND_path, MINI_path], loc="lower left", fontsize=12
    )
    plt.gca().add_artist(first_legend)

    del_chi2 = 4.61  # 90% confidence level
    plt.contour(
        scaled_theta,
        xv,
        chi_squared_values,
        levels=[min_chi_squared + del_chi2],
        colors=["red"],
    )

    plt.xlabel(
        r"$sin^2(2\theta_{\mu e})=sin^2(\theta_{24})sin^2(2\theta_{14})$", fontsize=20
    )
    plt.ylabel(r"$\Delta$ $m_{14}^2$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
