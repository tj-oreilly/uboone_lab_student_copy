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
    accuracy_score, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, precision_score, recall_score
)
import matplotlib.pyplot as plt
import shap  # For SHAP value calculations
import pickle

class NeuralNetwork():
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
        
            self.model = tf.keras.models.load_model('./cache/neural_net_cache.keras', 
                                                    custom_objects={'focal_loss_fn': self.focal_loss()})
            with open('./cache/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            return True

        return False

    def save_model(self):

        if not os.path.exists("./cache"):
            os.mkdir("./cache")

        self.model.save('./cache/neural_net_cache.keras')

        with open('./cache/scaler.pkl', 'wb') as f:
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

        self.MC_data['category'] = (self.MC_data['category'] == 21).astype(int)
        # self.data['category'] = (self.data['category'] == 21).astype(int) -- no category??

        print(r"Category mapping (old to new): $\nu_\mu$ CC -> 1, background -> 0")

        # ------------------------------------------------------------------------------------
        # -------------------------- Step 3: Compute collumns --------------------------------
        # ------------------------------------------------------------------------------------

        # Comnpute for MC_EXT

        # Compute the differences between end and start coordinates
        dx = (self.MC_data['trk_sce_end_x_v'] - self.MC_data['trk_sce_start_x_v']).astype(np.float64)
        dy = (self.MC_data['trk_sce_end_y_v'] - self.MC_data['trk_sce_start_y_v']).astype(np.float64)
        dz = (self.MC_data['trk_sce_end_z_v'] - self.MC_data['trk_sce_start_z_v']).astype(np.float64)

        # Compute the track length (Euclidean distance)
        self.MC_data['trk_sce_reco_len'] = np.sqrt(dx**2 + dy**2 + dz**2)

        # Compute the angle relative to the +z-axis which is the direction of the beam
        self.MC_data['trk_sce_reco_theta'] = np.where(
            self.MC_data['trk_sce_reco_len'] != 0,
            np.arccos(np.clip(dz / self.MC_data['trk_sce_reco_len'], -1, 1)),
            np.nan
        )

        # Compute phi: the angle in the xy-plane relative to the x-axis.
        self.MC_data['trk_sce_reco_phi'] = np.arctan2(dy, dx)

        # drop the old columns:
        self.MC_data = self.MC_data.drop(columns=[
            'trk_sce_end_x_v', 'trk_sce_end_y_v', 'trk_sce_end_z_v',
            'trk_sce_start_x_v', 'trk_sce_start_y_v', 'trk_sce_start_z_v',
            'trk_sce_reco_len'
        ])

        # Compute for data

        # Compute the differences between end and start coordinates
        dx = (self.data['trk_sce_end_x_v'] - self.data['trk_sce_start_x_v']).astype(np.float64)
        dy = (self.data['trk_sce_end_y_v'] - self.data['trk_sce_start_y_v']).astype(np.float64)
        dz = (self.data['trk_sce_end_z_v'] - self.data['trk_sce_start_z_v']).astype(np.float64)

        # Compute the track length (Euclidean distance)
        self.data['trk_sce_reco_len'] = np.sqrt(dx**2 + dy**2 + dz**2)

        # Compute the angle relative to the +z-axis which is the direction of the beam
        self.data['trk_sce_reco_theta'] = np.where(
            self.data['trk_sce_reco_len'] != 0,
            np.arccos(np.clip(dz / self.data['trk_sce_reco_len'], -1, 1)),
            np.nan
        )

        # Compute phi: the angle in the xy-plane relative to the x-axis.
        self.data['trk_sce_reco_phi'] = np.arctan2(dy, dx)

        # drop the old columns:
        self.data = self.data.drop(columns=[
            'trk_sce_end_x_v', 'trk_sce_end_y_v', 'trk_sce_end_z_v',
            'trk_sce_start_x_v', 'trk_sce_start_y_v', 'trk_sce_start_z_v',
            'trk_sce_reco_len'
        ])

        # ------------------------------------------------------------------------------------
        # -------------------------- Step 4: Unifie units ------------------------------------
        # ------------------------------------------------------------------------------------

        # For MC_EXT
        # Convert cm to m to help the model converge faster
        self.MC_data['trk_len_v'] = self.MC_data['trk_len_v'] / 100
        self.MC_data['trk_distance_v'] = self.MC_data['trk_distance_v'] / 100
        self.MC_data['reco_nu_vtx_sce_x'] = self.MC_data['reco_nu_vtx_sce_x'] / 100
        self.MC_data['reco_nu_vtx_sce_y'] = self.MC_data['reco_nu_vtx_sce_y'] / 100
        self.MC_data['reco_nu_vtx_sce_z'] = self.MC_data['reco_nu_vtx_sce_z'] / 100
        self.MC_data['_closestNuCosmicDist'] = self.MC_data['_closestNuCosmicDist'] / 100

        # Plot the result
        # if plot_category: fct.plot_selection_cut_category(MC_data, removed_columns=['true_L', 'true_muon_mom', 'true_E', 'reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z'])

        # For data
        # Convert cm to m to help the model converge faster
        self.data['trk_len_v'] = self.data['trk_len_v'] / 100
        self.data['trk_distance_v'] = self.data['trk_distance_v'] / 100
        self.data['reco_nu_vtx_sce_x'] = self.data['reco_nu_vtx_sce_x'] / 100
        self.data['reco_nu_vtx_sce_y'] = self.data['reco_nu_vtx_sce_y'] / 100
        self.data['reco_nu_vtx_sce_z'] = self.data['reco_nu_vtx_sce_z'] / 100
        self.data['_closestNuCosmicDist'] = self.data['_closestNuCosmicDist'] / 100

        # Plot the result
        # if plot_category: fct.plot_selection_cut_category(data, removed_columns=['reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z'])
        
        # -------------------------------------------------------------------------------------
        # ---------------------------- Step 5: extra columns ----------------------------------
        # -------------------------------------------------------------------------------------

        self.MC_data = self.MC_data.drop(columns=['true_L', 'true_muon_mom', 'true_E'])
        
        self.MC_data = self.MC_data.drop(columns=['reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z'])
        self.data = self.data.drop(columns=['reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z'])

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
            cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
            
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
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, mc_weights, test_size=0.2, stratify=y, random_state=42
        )
        
        # Use additional validation split for more reliable evaluation
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X_train, y_train, weights_train, test_size=0.2, stratify=y_train, random_state=42
        )

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Create model with Leaky ReLU and Focal Loss
        def create_model(input_dim, use_focal_loss=True, learning_rate=0.001, leaky_alpha=0.01):
            # Use functional API for more flexibility
            inputs = Input(shape=(input_dim,))
            
            # First layer with Leaky ReLU
            x = Dense(120)(inputs)
            x = LeakyReLU(negative_slope=leaky_alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            # Second layer with Leaky ReLU
            x = Dense(140)(x)
            x = LeakyReLU(negative_slope=leaky_alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            # Third layer with Leaky ReLU
            # Note: Fixed the connection - was previously connected to inputs
            x = Dense(40)(x)
            x = LeakyReLU(negative_slope=leaky_alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Output layer
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Choose loss function based on parameter
            if use_focal_loss:
                # Alpha is adjusted to match class imbalance: ~25% for class 0, ~75% for class 1
                loss_fn = self.focal_loss(gamma=1.0, alpha=0.25)
                print("Using Focal Loss with gamma=1.0, alpha=0.25")
            else:
                loss_fn = 'binary_crossentropy'
                print("Using Binary Cross-Entropy Loss")
            
            # Compile with loss function 
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=loss_fn,
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            return model

        # Calculate class weights to handle imbalanced data
        n_neg, n_pos = np.bincount(y_train)
        total = n_neg + n_pos
        class_weight = {0: (total/n_neg)/2, 1: (total/n_pos)/2}
        
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
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
        ]
        
        print("Training model...")
        history = model.fit(
            X_train_scaled, y_train,
            sample_weight=weights_train,
            epochs=150,
            batch_size=128,  # Smaller batch size for better generalization
            validation_data=(X_val_scaled, y_val, weights_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        confidence_values = model.predict(X_test_scaled)
        binary_predictions = (confidence_values > 0.5).astype(int)
        
        print("\nNeural Network Evaluation Metrics:")
        print("Accuracy:", accuracy_score(y_test, binary_predictions))
        print("\nClassification Report:\n", classification_report(y_test, binary_predictions))
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

        THRESHOLD = 0.6

        # Beginning purity
        startPurity = len(self.MC_data[self.MC_data["category"] == 1]) / float(len(self.MC_data))
        print(f"Start purity: {startPurity}")
        
        # X = self.MC_data.drop(columns=['category', 'weight']) # Don't use category or weight for neural net
        # x_scaled = self.scaler.transform(X)
        # MC_confidence = self.model.predict(x_scaled)

        # count = 0
        # for i, (index, row) in enumerate(X.iterrows()):
        #     if MC_confidence[i] < THRESHOLD:
        #         count += 1

        # print(f"Purity: {(count / float(len(self.MC_data))):.2f}")

        # y = self.MC_data["category"].copy()

        # THRESHOLD = self.optimize_threshold_for_purity(self.model, X, y)
        
        if self.model == None or self.scaler == None:
            return
        
        self.data = self.data.drop(columns=['category']) # Seems to have picked up a rogue category col
        
        data_scaled = self.scaler.transform(self.data)
        confidence_values = self.model.predict(data_scaled)

        for i, (index, row) in enumerate(self.data.iterrows()):
            if confidence_values[i] > THRESHOLD:
                self.data.at[index, 'background'] = False
            else:
                self.data.at[index, 'background'] = True

        # MC data

        categories = self.MC_data["category"].copy()

        self.MC_data = self.MC_data.drop(columns=['category','weight'])

        data_scaled = self.scaler.transform(self.MC_data)
        confidence_values = self.model.predict(data_scaled)

        for i, (index, row) in enumerate(self.MC_data.iterrows()):
            if confidence_values[i] > THRESHOLD:
                self.MC_data.at[index, 'background'] = False
            else:
                self.MC_data.at[index, 'background'] = True

        # Calculate purity

        self.MC_data["category"] = categories

        count = 0
        for index, row in self.MC_data.iterrows():
            if not self.MC_data.at[index, 'background'] and self.MC_data.at[index, 'category'] == 1:
                count += 1

        print(f"Purity: {count / float(len(self.MC_data)):.2f}")

    def optimize_threshold_for_purity(self, model, X_test, y_test, target_purity=0.89):
        """
        Find a threshold that optimizes for purity while considering selection efficiency.
        
        Efficiency = Fraction of total data remaining after selection (selected/total)
        Purity = Fraction of selected data that is truly class 1 (true_class1/selected)
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            target_purity: Minimum desired purity for class 1
        """
        print("\n--- Purity Optimization with Corrected Definitions ---")
        print(f"Finding threshold that achieves at least {target_purity:.2f} purity while maximizing selection efficiency")
        
        # Get prediction probabilities
        y_pred_prob = model.predict(X_test).flatten()
        
        # Calculate total number of samples
        total_samples = len(y_test)
        total_class1 = np.sum(y_test)
        total_class0 = total_samples - total_class1
        
        # Try a range of thresholds with finer granularity
        thresholds = np.linspace(0.1, 0.99, 40)  # Detailed threshold search
        results = []
        
        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Calculate metrics with your definitions
            selected_samples = tp + fp  # Total samples classified as class 1
            
            # Calculate purity: fraction of selected data that is truly class 1
            purity = tp / selected_samples if selected_samples > 0 else 0
            
            # Calculate efficiency: fraction of total data that remains after selection
            selection_efficiency = selected_samples / total_samples
            
            # Calculate signal efficiency (traditional recall): fraction of class 1 correctly identified
            signal_efficiency = tp / total_class1 if total_class1 > 0 else 0
            
            # Calculate background rejection: fraction of class 0 correctly rejected
            background_rejection = tn / total_class0 if total_class0 > 0 else 0
            
            # Calculate a custom score that prioritizes thresholds meeting target purity
            if purity >= target_purity:
                # If we meet the target purity, the score is the selection efficiency
                custom_score = selection_efficiency
            else:
                # If we don't meet target purity, penalize proportional to how far we are
                purity_shortfall = target_purity - purity
                custom_score = selection_efficiency - (purity_shortfall * 5)  # Penalty for not meeting purity
            
            results.append({
                'threshold': threshold,
                'purity': purity,
                'selection_efficiency': selection_efficiency,  # New definition: fraction of total data kept
                'signal_efficiency': signal_efficiency,        # Traditional recall: fraction of class 1 captured
                'background_rejection': background_rejection,  # Fraction of class 0 correctly rejected
                'selected_samples': selected_samples,          # Absolute number of samples selected
                'custom_score': custom_score
            })
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Find all thresholds that meet the target purity
        meets_target = results_df[results_df['purity'] >= target_purity]
        
        if meets_target.empty:
            # If no threshold meets the target, find the one with highest purity
            print(f"No threshold achieves the target purity of {target_purity:.2f}")
            best_idx = results_df['purity'].idxmax()
        else:
            # Find the threshold with highest selection efficiency among those meeting target purity
            best_idx = meets_target['selection_efficiency'].idxmax()
        
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_purity = results_df.loc[best_idx, 'purity']
        best_efficiency = results_df.loc[best_idx, 'selection_efficiency']
        
        # Sort results by purity for display
        print(f"\nTop threshold options by purity:")
        print(results_df.sort_values('purity', ascending=False).head(5)[['threshold', 'purity', 'selection_efficiency', 'selected_samples']].to_string(index=False))
        
        # Sort results by selection efficiency for display
        print(f"\nTop threshold options by selection efficiency (fraction of data kept):")
        print(results_df.sort_values('selection_efficiency', ascending=False).head(5)[['threshold', 'purity', 'selection_efficiency', 'selected_samples']].to_string(index=False))
        
        # Show options that meet target purity
        if not meets_target.empty:
            print(f"\nThreshold options meeting target purity of {target_purity:.2f}:")
            print(meets_target.sort_values('selection_efficiency', ascending=False).head(5)[['threshold', 'purity', 'selection_efficiency', 'selected_samples']].to_string(index=False))
        
        # Evaluate with the best threshold
        y_pred_optimal = (y_pred_prob >= best_threshold).astype(int)
        selected_samples = np.sum(y_pred_optimal)
        
        print(f"\nOptimal threshold for purity >= {target_purity:.2f}: {best_threshold:.4f}")
        print(f"Achieved purity: {best_purity:.4f}, Selection efficiency: {best_efficiency:.4f}")
        print(f"Number of samples selected: {selected_samples} out of {total_samples} ({selected_samples/total_samples*100:.2f}%)")
        print(f"Performance at optimal threshold:")
        print(classification_report(y_test, y_pred_optimal))
        
        # Plot threshold vs metrics
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['threshold'], results_df['purity'], 'r-', label='Purity')
        plt.plot(results_df['threshold'], results_df['selection_efficiency'], 'b-', label='Efficiency')
        #plt.plot(results_df['threshold'], results_df['signal_efficiency'], 'g-', label='Signal Efficiency (Recall)')
        #plt.plot(results_df['threshold'], results_df['background_rejection'], 'm-', label='Background Rejection')
        
        # Add horizontal line for target purity
        #plt.axhline(y=target_purity, color='k', linestyle='--', label=f'Target Purity: {target_purity:.2f}')
        
        # Add vertical line for best threshold
        #plt.axvline(x=best_threshold, color='y', linestyle='--', label=f'Optimal Threshold: {best_threshold:.3f}')
        
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Threshold vs. Purity and Efficiency Metrics')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
        plt.savefig('revised_purity_optimization.png')
        plt.close()
        print("Revised purity optimization plot saved to 'revised_purity_optimization.png'")
        
        return best_threshold

    def plot_training_history(self, history):
        """Plot the training and validation metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved to 'training_history.png'")

    def feature_importance_analysis(self, model, X_train, y_train, X_test, y_test, feature_names):
        """
        Perform multiple feature importance analyses using different methods
        """
        print("\n--- Feature Importance Analysis ---")
        
        # Method 1: Simple permutation importance using manual implementation
        # since scikit-learn's implementation is having compatibility issues
        def get_model_score(X, y):
            y_pred = model.predict(X).flatten()
            return roc_auc_score(y, y_pred)
        
        # Get baseline score
        baseline_score = get_model_score(X_test, y_test)
        print(f"Baseline ROC AUC score: {baseline_score:.4f}")
        
        # Calculate feature importance through permutation
        importances = []
        importances_std = []
        
        for i in range(len(feature_names)):
            feature_scores = []
            for _ in range(5):  # Run 5 iterations for each feature
                # Create a copy of the test data
                X_permuted = X_test.copy()
                # Permute one feature
                np.random.shuffle(X_permuted[:, i])
                # Calculate new score
                permuted_score = get_model_score(X_permuted, y_test)
                # Store importance as decrease in performance
                feature_scores.append(baseline_score - permuted_score)
            
            importances.append(np.mean(feature_scores))
            importances_std.append(np.std(feature_scores))
        
        # Create DataFrame for easier reading
        perm_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'StdDev': importances_std
        })
        perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)
        
        print("\nPermutation Feature Importance:")
        print(perm_importance_df)
        
        # Plot permutation importance
        plt.figure(figsize=(12, 6))
        plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance'])
        plt.xlabel('Importance (Decrease in ROC AUC)')
        plt.title('Feature Importance by Permutation')
        plt.tight_layout()
        plt.savefig('permutation_importance.png')
        plt.close()
        print("Permutation importance plot saved to 'permutation_importance.png'")
        
        # Method 2: SHAP values for more detailed analysis
        try:
            # Create explainer for the model
            explainer = shap.DeepExplainer(model, X_train[:100])  # Use a subset for efficiency
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test[:100])
            
            # Plot summary
            plt.figure()
            shap.summary_plot(shap_values[0], X_test[:100], feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            plt.close()
            print("SHAP summary plot saved to 'shap_summary.png'")
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            print("Continuing with other methods...")
        
        return perm_importance_df