# from general_MicroBooNe import functions as fct
# from general_MicroBooNe import setup
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import shap  # For SHAP value calculations
import pickle


class NeuralNetwork:
    def __init__(self, MC_data, data):
        np.random.seed(42)
        tf.random.set_seed(42)

        self.model = None
        self.scaler = None

        self.MC_data = MC_data
        self.data = data

    def try_load_model(self):
        """Tries to load a cached model"""

        if os.path.exists("./cache/neural_net_cache.keras"):
            print("Found a cached neural network. Loading into memory...")

            self.model = tf.keras.models.load_model(
                "./cache/neural_net_cache.keras",
                custom_objects={"focal_loss_fn": self.focal_loss()},
            )
            with open("./cache/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            return True

        return False

    def save_model(self):
        if not os.path.exists("./cache"):
            os.mkdir("./cache")

        self.model.save("./cache/neural_net_cache.keras")

        with open("./cache/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepares the data for the neural network by applying selection cuts, combining categories,
        computing new columns, unifying units, and optionally removing vertex columns.
        Parameters:
        -----------
        plot_cuts : bool, optional
            If True, plots the selection cuts. Default is False.
        plot_category : bool, optional
            If True, plots the category selection cuts. Default is False.
        remove_vtx : bool, optional
            If True, removes the vertex columns from the data. Default is True.
        Returns:
        --------
        pd.DataFrame
            The processed MC_EXT and data DataFrames.
        """

        # Define the selection cuts
        # selection_cuts = {
        #             "cut_length":        lambda d: d['trk_len_v'] < 1000,
        #             "cut_distance":      lambda d: d['trk_distance_v'] < 100,
        #             "cut_energy":        lambda d: d['trk_energy_tot'] < 4.0,
        #             "cut_muon_mom":      lambda d: d['trk_range_muon_mom_v'] < 2,
        #             "cut_mcs_muon_mom":  lambda d: d['trk_mcs_muon_mom_v'] < 2,
        #             "cut_track_score":   lambda d: d['trk_score_v'] > -0.1,
        #             "cut_cosmic":        lambda d: (d['_closestNuCosmicDist'] > 0.1) & (d['_closestNuCosmicDist'] < 450),
        #             #"cut_reco_x":        lambda d: (d['reco_nu_vtx_sce_x'] > 30) & (d['reco_nu_vtx_sce_x'] < 220),
        #             #"cut_reco_y":        lambda d: (d['reco_nu_vtx_sce_y'] > -100) & (d['reco_nu_vtx_sce_y'] < 100),
        #             #"cut_reco_z":        lambda d: (d['reco_nu_vtx_sce_z'] > 75) & (d['reco_nu_vtx_sce_z'] < 900),
        #             "cut_trk_start_x_v": lambda d: (d['trk_sce_start_x_v'] > -5) & (d['trk_sce_start_x_v'] < 270),
        #             "cut_trk_start_y_v": lambda d: (d['trk_sce_start_y_v'] > -150) & (d['trk_sce_start_y_v'] < 150),
        #             #"cut_trk_start_z_v": lambda d: (d['trk_sce_start_z_v'] > 75) & (d['trk_sce_start_z_v'] < 900),
        #             "cut_trk_end_x_v":   lambda d: (d['trk_sce_end_x_v'] > -5) & (d['trk_sce_end_x_v'] < 270),
        #             "cut_trk_end_y_v":   lambda d: (d['trk_sce_end_y_v'] > -150) & (d['trk_sce_end_y_v'] < 150),
        #             #"cut_trk_end_z_v":   lambda d: (d['trk_sce_end_z_v'] > 75) & (d['trk_sce_end_z_v'] < 900),
        #         }

        # # load the data
        # MC_EXT, data = setup.setup()
        # MC_EXT = fct.clean_up_data(MC_EXT, re_add_nan=False, supress=True)
        # data = fct.clean_up_data(data, re_add_nan=False, supress=True)

        # # ------------------------------------------------------------------------------------
        # # --------------------------- Step 1: Remove Poisson Tails ---------------------------
        # # ------------------------------------------------------------------------------------

        # # For MC_EXT
        # MC_EXT = fct.Selections(MC_EXT, selection_cuts=selection_cuts, plot=plot_cuts)
        # if plot_cuts: fct.plot_selection_cut(MC_EXT, removed_columns=['true_L', 'true_muon_mom', 'true_E'])
        # MC_EXT = MC_EXT[MC_EXT['flag'] == 1]
        # MC_EXT = MC_EXT.drop(columns=['flag'])

        # # For data
        # data = fct.Selections(data, selection_cuts=selection_cuts, plot=plot_cuts)
        # if plot_cuts : fct.plot_selection_cut(data)
        # data = data[data['flag'] == 1]
        # data = data.drop(columns=['flag'])

        # ------------------------------------------------------------------------------------
        # -------------------------- Step 2: Combine categories ------------------------------
        # ------------------------------------------------------------------------------------

        self.MC_data["category"] = (self.MC_data["category"] == 21).astype(int)
        # self.data['category'] = (self.data['category'] == 21).astype(int) -- no category??

        print(r"Category mapping (old to new): $\nu_\mu$ CC -> 1, background -> 0")

        # ------------------------------------------------------------------------------------
        # -------------------------- Step 3: Compute collumns --------------------------------
        # ------------------------------------------------------------------------------------

        # Comnpute for MC_EXT

        # Compute the differences between end and start coordinates
        dx = (
            self.MC_data["trk_sce_end_x_v"] - self.MC_data["trk_sce_start_x_v"]
        ).astype(np.float64)
        dy = (
            self.MC_data["trk_sce_end_y_v"] - self.MC_data["trk_sce_start_y_v"]
        ).astype(np.float64)
        dz = (
            self.MC_data["trk_sce_end_z_v"] - self.MC_data["trk_sce_start_z_v"]
        ).astype(np.float64)

        # Compute the track length (Euclidean distance)
        self.MC_data["trk_sce_reco_len"] = np.sqrt(dx**2 + dy**2 + dz**2)

        # Compute the angle relative to the +z-axis which is the direction of the beam
        self.MC_data["trk_sce_reco_theta"] = np.where(
            self.MC_data["trk_sce_reco_len"] != 0,
            np.arccos(np.clip(dz / self.MC_data["trk_sce_reco_len"], -1, 1)),
            np.nan,
        )

        # Compute phi: the angle in the xy-plane relative to the x-axis.
        self.MC_data["trk_sce_reco_phi"] = np.arctan2(dy, dx)

        # drop the old columns:
        self.MC_data = self.MC_data.drop(
            columns=[
                "trk_sce_end_x_v",
                "trk_sce_end_y_v",
                "trk_sce_end_z_v",
                "trk_sce_start_x_v",
                "trk_sce_start_y_v",
                "trk_sce_start_z_v",
                "trk_sce_reco_len",
            ]
        )

        # Compute for data

        # Compute the differences between end and start coordinates
        dx = (self.data["trk_sce_end_x_v"] - self.data["trk_sce_start_x_v"]).astype(
            np.float64
        )
        dy = (self.data["trk_sce_end_y_v"] - self.data["trk_sce_start_y_v"]).astype(
            np.float64
        )
        dz = (self.data["trk_sce_end_z_v"] - self.data["trk_sce_start_z_v"]).astype(
            np.float64
        )

        # Compute the track length (Euclidean distance)
        self.data["trk_sce_reco_len"] = np.sqrt(dx**2 + dy**2 + dz**2)

        # Compute the angle relative to the +z-axis which is the direction of the beam
        self.data["trk_sce_reco_theta"] = np.where(
            self.data["trk_sce_reco_len"] != 0,
            np.arccos(np.clip(dz / self.data["trk_sce_reco_len"], -1, 1)),
            np.nan,
        )

        # Compute phi: the angle in the xy-plane relative to the x-axis.
        self.data["trk_sce_reco_phi"] = np.arctan2(dy, dx)

        # drop the old columns:
        self.data = self.data.drop(
            columns=[
                "trk_sce_end_x_v",
                "trk_sce_end_y_v",
                "trk_sce_end_z_v",
                "trk_sce_start_x_v",
                "trk_sce_start_y_v",
                "trk_sce_start_z_v",
                "trk_sce_reco_len",
            ]
        )

        # ------------------------------------------------------------------------------------
        # -------------------------- Step 4: Unifie units ------------------------------------
        # ------------------------------------------------------------------------------------

        # For MC_EXT
        # Convert cm to m to help the model converge faster
        self.MC_data["trk_len_v"] = self.MC_data["trk_len_v"] / 100
        self.MC_data["trk_distance_v"] = self.MC_data["trk_distance_v"] / 100
        self.MC_data["reco_nu_vtx_sce_x"] = self.MC_data["reco_nu_vtx_sce_x"] / 100
        self.MC_data["reco_nu_vtx_sce_y"] = self.MC_data["reco_nu_vtx_sce_y"] / 100
        self.MC_data["reco_nu_vtx_sce_z"] = self.MC_data["reco_nu_vtx_sce_z"] / 100
        self.MC_data["_closestNuCosmicDist"] = (
            self.MC_data["_closestNuCosmicDist"] / 100
        )

        # Plot the result
        # if plot_category: fct.plot_selection_cut_category(MC_data, removed_columns=['true_L', 'true_muon_mom', 'true_E', 'reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z'])

        # For data
        # Convert cm to m to help the model converge faster
        self.data["trk_len_v"] = self.data["trk_len_v"] / 100
        self.data["trk_distance_v"] = self.data["trk_distance_v"] / 100
        self.data["reco_nu_vtx_sce_x"] = self.data["reco_nu_vtx_sce_x"] / 100
        self.data["reco_nu_vtx_sce_y"] = self.data["reco_nu_vtx_sce_y"] / 100
        self.data["reco_nu_vtx_sce_z"] = self.data["reco_nu_vtx_sce_z"] / 100
        self.data["_closestNuCosmicDist"] = self.data["_closestNuCosmicDist"] / 100

        # Plot the result
        # if plot_category: fct.plot_selection_cut_category(data, removed_columns=['reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z'])

        # -------------------------------------------------------------------------------------
        # ---------------------------- Step 5: extra columns ----------------------------------
        # -------------------------------------------------------------------------------------

        self.MC_data = self.MC_data.drop(columns=["true_L", "true_muon_mom", "true_E"])

        self.MC_data = self.MC_data.drop(
            columns=["reco_nu_vtx_sce_x", "reco_nu_vtx_sce_y", "reco_nu_vtx_sce_z"]
        )
        self.data = self.data.drop(
            columns=["reco_nu_vtx_sce_x", "reco_nu_vtx_sce_y", "reco_nu_vtx_sce_z"]
        )

    def focal_loss(self, gamma=1.0, alpha=0.25):
        """
        Focal Loss function for binary classification

        Args:
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            alpha: Weighting factor for the rare class

        Returns:
            A loss function that can be used in model compilation
        """

        def focal_loss_fn(y_true, y_pred):
            # Clip prediction values to avoid log(0) errors
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            # Calculate cross entropy
            cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
                1 - y_pred
            )

            # Calculate focal term
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            focal_term = tf.pow(1 - p_t, gamma)

            # Apply alpha weighting
            alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

            # Calculate final focal loss
            focal_cross_entropy = alpha_factor * focal_term * cross_entropy

            return tf.reduce_mean(focal_cross_entropy)

        return focal_loss_fn

    def train_evaluate_model(self, X, y, mc_weights, feature_names):
        """
        Train and evaluate neural network model with feature importance analysis
        """

        if self.try_load_model():
            return

        # Split the data with stratification for imbalanced dataset
        X_train, X_test, y_train, y_test, weights_train, weights_test = (
            train_test_split(
                X, y, mc_weights, test_size=0.2, stratify=y, random_state=42
            )
        )

        # Use additional validation split for more reliable evaluation
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X_train,
            y_train,
            weights_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42,
        )

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Create model with Leaky ReLU and Focal Loss
        def create_model(
            input_dim, use_focal_loss=True, learning_rate=0.001, leaky_alpha=0.01
        ):
            # Use functional API for more flexibility
            inputs = Input(shape=(input_dim,))

            # First layer with Leaky ReLU
            x = Dense(256)(inputs)
            x = LeakyReLU(negative_slope=leaky_alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            # Second layer with Leaky ReLU
            x = Dense(128)(x)
            x = LeakyReLU(negative_slope=leaky_alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            # Third layer with Leaky ReLU
            # Note: Fixed the connection - was previously connected to inputs
            x = Dense(64)(x)
            x = LeakyReLU(negative_slope=leaky_alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            # Output layer
            outputs = Dense(1, activation="sigmoid")(x)

            model = Model(inputs=inputs, outputs=outputs)

            # Choose loss function based on parameter
            if use_focal_loss:
                # Alpha is adjusted to match class imbalance: ~25% for class 0, ~75% for class 1
                loss_fn = self.focal_loss(gamma=1.0, alpha=0.25)
                print("Using Focal Loss with gamma=1.0, alpha=0.25")
            else:
                loss_fn = "binary_crossentropy"
                print("Using Binary Cross-Entropy Loss")

            # Compile with loss function
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=loss_fn,
                metrics=[
                    "accuracy",
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )

            return model

        # Calculate class weights to handle imbalanced data
        n_neg, n_pos = np.bincount(y_train)
        total = n_neg + n_pos
        class_weight = {0: (total / n_neg) / 2, 1: (total / n_pos) / 2}

        # Print class distribution and weights information
        print(f"Class distribution in training set: Class 0: {n_neg}, Class 1: {n_pos}")
        print(f"Class weights: {class_weight}")

        # Create and train the model
        input_dim = X_train.shape[1]
        model = create_model(input_dim, use_focal_loss=True)

        # Print model summary
        model.summary()

        # Improved callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-5),
        ]

        print("Training model...")
        history = model.fit(
            X_train_scaled,
            y_train,
            sample_weight=weights_train,
            epochs=100,
            batch_size=128,  # Smaller batch size for better generalization
            validation_data=(X_val_scaled, y_val, weights_val),
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate model
        confidence_values = model.predict(X_test_scaled)
        binary_predictions = (confidence_values > 0.5).astype(int)

        print("\nNeural Network Evaluation Metrics:")
        print("Accuracy:", accuracy_score(y_test, binary_predictions))
        print(
            "\nClassification Report:\n",
            classification_report(y_test, binary_predictions),
        )
        print("\nROC AUC Score:", roc_auc_score(y_test, confidence_values))

        # Plot training history
        # self.plot_training_history(history)

        # Perform feature importance analysis
        # self.feature_importance_analysis(model, X_train_scaled, y_train, X_test_scaled, y_test, feature_names)

        # Evaluate model with different thresholds
        # self.optimize_threshold_for_purity(model, X_test_scaled, y_test, target_purity=0.9)

        self.model = model
        self.scaler = scaler

        self.save_model()

    def categorise_data(self):
        THRESHOLD = 0.4

        X = self.MC_data.drop(
            columns=["category", "weight"]
        )  # Don't use category or weight for neural net
        # x_scaled = self.scaler.transform(X)
        # MC_confidence = self.model.predict(x_scaled)

        # count = 0
        # for i, (index, row) in enumerate(X.iterrows()):
        #     if MC_confidence[i] < THRESHOLD:
        #         count += 1

        # print(f"Purity: {(count / float(len(self.MC_data))):.2f}")

        y = self.MC_data["category"].copy()

        # self.optimize_threshold_for_purity(self.model, X, y)

        if self.model is None or self.scaler is None:
            return

        self.data = self.data.drop(
            columns=["category"]
        )  # Seems to have picked up a rogue category col

        data_scaled = self.scaler.transform(self.data)
        confidence_values = self.model.predict(data_scaled)

        for i, (index, row) in enumerate(self.data.iterrows()):
            if confidence_values[i] > THRESHOLD:
                self.data.at[index, "background"] = False
            else:
                self.data.at[index, "background"] = True

        # MC data

        categories = self.MC_data["category"].copy()

        self.MC_data = self.MC_data.drop(columns=["category", "weight"])

        data_scaled = self.scaler.transform(self.MC_data)
        confidence_values = self.model.predict(data_scaled)

        for i, (index, row) in enumerate(self.MC_data.iterrows()):
            if confidence_values[i] > THRESHOLD:
                self.MC_data.at[index, "background"] = False
            else:
                self.MC_data.at[index, "background"] = True
