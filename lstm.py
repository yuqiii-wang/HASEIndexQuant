import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target

# For reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class _LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_seq_len, num_target_features, dropout_prob):
        super().__init__()
        self.input_size = input_size # num features in input
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len # num timesteps to predict
        self.num_target_features = num_target_features # num features to predict per timestep

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # Fully connected layer to map LSTM output to the desired output shape
        # Output will be (batch_size, output_seq_len * num_target_features)
        self.fc = nn.Linear(hidden_size, output_seq_len * num_target_features)

    def forward(self, x):
        # x shape: (batch_size, history_seq_len, input_size)
        
        # h0 and c0 are initialized to zeros by default if not provided
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We use the output from the last time step of the LSTM
        # lstm_out shape: (batch_size, history_seq_len, hidden_size)
        # last_time_step_out shape: (batch_size, hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        
        out = self.fc(last_time_step_out)
        
        # Reshape to (batch_size, output_seq_len, num_target_features)
        out = out.view(-1, self.output_seq_len, self.num_target_features)
        return out

class SklearnLSTM(BaseEstimator, RegressorMixin):
    def __init__(self, history_seq_len=10, output_seq_len=1,
                 hidden_size=50, num_layers=1, dropout_prob=0.0,
                 epochs=100, batch_size=32, learning_rate=1e-3,
                 random_state=None, verbose=0):
        
        self.history_seq_len = history_seq_len
        self.output_seq_len = output_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.scaler_ = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size_ = None
        self.num_target_features_ = None
        
        if self.random_state is not None:
            seed_everything(self.random_state)

    def _create_sequences(self, data, history_len, output_len):
        X_list, y_list = [], []
        # data shape: (num_samples, num_features)
        for i in range(len(data) - history_len - output_len + 1):
            X_list.append(data[i : i + history_len, :]) # History
            # Target: all features for the output_len period
            y_list.append(data[i + history_len : i + history_len + output_len, :]) 
        
        if not X_list: # Not enough data to create any sequence
            raise ValueError(
                f"Not enough data to create sequences. "
                f"Need at least {history_len + output_len} samples, got {len(data)}."
            )
            
        return np.array(X_list), np.array(y_list)

    def fit(self, X, y=None): # y is ignored, targets are derived from X
        # Input validation and processing
        if isinstance(X, pd.DataFrame):
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("If X is a DataFrame, its index must be a DatetimeIndex.")
            X_sorted = X.sort_index()
            self.feature_names_in_ = X_sorted.columns.to_list()
            data_values = X_sorted.values
        elif isinstance(X, np.ndarray):
            # Assuming X is already sorted if it's a NumPy array of time series data
            data_values = X
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            raise ValueError("X must be a pandas DataFrame with DatetimeIndex or a NumPy array.")

        if data_values.ndim == 1:
            data_values = data_values.reshape(-1, 1)
        
        self.n_features_in_ = data_values.shape[1]

        # Scaling
        self.scaler_ = StandardScaler()
        scaled_data = self.scaler_.fit_transform(data_values)

        # Create sequences
        # X_seq shape: (num_sequences, history_seq_len, num_features)
        # y_seq shape: (num_sequences, output_seq_len, num_features)
        X_seq, y_seq = self._create_sequences(scaled_data, self.history_seq_len, self.output_seq_len)

        self.input_size_ = X_seq.shape[2] # num features in input
        self.num_target_features_ = y_seq.shape[2] # num features in output (usually same)

        # Initialize PyTorch model, optimizer, criterion
        self.model_ = _LSTMModel(
            input_size=self.input_size_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_seq_len=self.output_seq_len,
            num_target_features=self.num_target_features_,
            dropout_prob=self.dropout_prob
        ).to(self.device_)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        # Create DataLoader
        X_tensor = torch.from_numpy(X_seq).float().to(self.device_)
        y_tensor = torch.from_numpy(y_seq).float().to(self.device_)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if self.verbose > 0 and (epoch + 1) % (self.epochs // 10 if self.epochs >=10 else 1) == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(dataloader):.6f}")
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self) # Check if fit has been called

        # Input X for predict should be the last `history_seq_len` observations
        if isinstance(X, pd.DataFrame):
            if not all(col in X.columns for col in self.feature_names_in_):
                 raise ValueError("Input DataFrame for predict is missing some training columns.")
            # Ensure columns are in the same order as during training
            X_input_df = X[self.feature_names_in_]
            if isinstance(X_input_df.index, pd.DatetimeIndex):
                 X_input_df = X_input_df.sort_index() # Sort if datetime index provided

            # Take the last history_seq_len rows if more are provided
            if len(X_input_df) > self.history_seq_len:
                if self.verbose > 0:
                    print(f"Warning: Input X for predict has {len(X_input_df)} rows, "
                          f"using the last {self.history_seq_len} rows.")
                input_values = X_input_df.iloc[-self.history_seq_len:].values
            elif len(X_input_df) < self.history_seq_len:
                raise ValueError(f"Input X for predict must have at least {self.history_seq_len} rows, "
                                 f"got {len(X_input_df)}.")
            else: # len(X_input_df) == self.history_seq_len
                input_values = X_input_df.values

        elif isinstance(X, np.ndarray):
            input_values = X
            if input_values.ndim == 1:
                input_values = input_values.reshape(-1, 1) # Reshape if 1D array
            if input_values.shape[0] != self.history_seq_len:
                 raise ValueError(f"Input X (NumPy array) for predict must have {self.history_seq_len} rows (time steps), "
                                  f"got {input_values.shape[0]}.")
            if input_values.shape[1] != self.n_features_in_:
                raise ValueError(f"Input X (NumPy array) for predict must have {self.n_features_in_} features, "
                                 f"got {input_values.shape[1]}.")
        else:
            raise ValueError("X for predict must be a pandas DataFrame or a NumPy array.")


        # Scale the input
        scaled_input = self.scaler_.transform(input_values)
        
        # Convert to tensor and add batch dimension
        # Expected shape: (1, history_seq_len, num_features)
        input_tensor = torch.from_numpy(scaled_input).float().unsqueeze(0).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            # predicted_scaled shape: (1, output_seq_len, num_target_features)
            predicted_scaled = self.model_(input_tensor)
        
        # Remove batch dimension and move to CPU
        # predicted_scaled_np shape: (output_seq_len, num_target_features)
        predicted_scaled_np = predicted_scaled.squeeze(0).cpu().numpy()
        
        # Inverse transform
        # scaler_.inverse_transform expects (num_samples, num_features)
        # Our predicted_scaled_np is (output_seq_len, num_target_features)
        # This is fine as each time step in output_seq_len is treated as a "sample" by scaler
        predicted_values = self.scaler_.inverse_transform(predicted_scaled_np)
        
        return predicted_values # Shape: (output_seq_len, num_target_features)

    # For sklearn compatibility (e.g. GridSearchCV)
    def get_params(self, deep=True):
        return {
            "history_seq_len": self.history_seq_len,
            "output_seq_len": self.output_seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout_prob": self.dropout_prob,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "verbose": self.verbose
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        
        # Re-apply random seed if it's set
        if self.random_state is not None:
            seed_everything(self.random_state)
        return self

if __name__ == '__main__':
    # --- Model Parameters for 55 min input, 5 min output ---
    # Assuming data is sampled at 1-minute intervals
    HISTORY_LEN = 55  # 55 minutes of history
    OUTPUT_LEN = 5    # 5 minutes to predict
    
    from utils import load_hsi_data, add_features

    hsi_df = load_hsi_data()
    hsi_feat_df = add_features(hsi_df)
    df = hsi_feat_df
    print(f"Generated data shape: {df.shape}")

    # --- Split Data (e.g., 80% train, 20% test) ---
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    
    # --- Initialize and Train Model ---
    lstm_regressor = SklearnLSTM(
        history_seq_len=HISTORY_LEN,   # 55
        output_seq_len=OUTPUT_LEN,     # 5
        hidden_size=64,
        num_layers=2,
        dropout_prob=0.1,
        epochs=30, # Keep epochs moderate for faster demo; increase for real tasks
        batch_size=32,
        learning_rate=0.003,
        random_state=42,
        verbose=1 
    )

    print(f"\n--- Fitting model with history_seq_len={HISTORY_LEN}, output_seq_len={OUTPUT_LEN} ---")
    # Fit the model using the training DataFrame
    lstm_regressor.fit(df_train) # y is not needed as it's derived from X

    print("\n--- Making a prediction ---")
    # To make a prediction, we need the last `HISTORY_LEN` (55) known data points.
    # Let's take them from the end of the training set to predict the start of the test set.
    
    input_for_prediction = df_train.iloc[-HISTORY_LEN:]
    print(f"Shape of input for prediction: {input_for_prediction.shape}") # Should be (55, num_features)
    
    predicted_sequence = lstm_regressor.predict(input_for_prediction)
    print(f"Predicted sequence shape: {predicted_sequence.shape}") # Should be (5, num_features)
    print(f"Predicted sequence (next {OUTPUT_LEN} minutes for {df.shape[1]} features):")
    print(predicted_sequence)

    # --- Compare with actual values from the test set ---
    # The predicted sequence corresponds to the first OUTPUT_LEN steps of df_test
    if len(df_test) >= OUTPUT_LEN:
        actual_sequence = df_test.iloc[:OUTPUT_LEN].values
        print("\nActual sequence from test set (first 5 minutes):")
        print(actual_sequence)
        
        mse = np.mean((predicted_sequence - actual_sequence)**2)
        print(f"\nMSE for this specific 5-minute prediction: {mse:.4f}")
    else:
        print(f"\nNot enough data in test set (need {OUTPUT_LEN}) to compare the full predicted sequence.")

    # --- Example: Sliding window prediction over the test set ---
    print("\n--- Evaluating on test set using a sliding window ---")
    all_predictions = []
    all_actuals = []

    # Ensure test set is large enough for at least one full evaluation window
    min_test_len = HISTORY_LEN + OUTPUT_LEN
    if len(df_test) >= min_test_len:
        num_test_predictions = len(df_test) - HISTORY_LEN - OUTPUT_LEN + 1
        print(f"Will make {num_test_predictions} sliding window predictions on the test set.")

        for i in range(num_test_predictions):
            current_history_window = df_test.iloc[i : i + HISTORY_LEN]
            current_actual_output = df_test.iloc[i + HISTORY_LEN : i + HISTORY_LEN + OUTPUT_LEN].values
            
            prediction = lstm_regressor.predict(current_history_window)
            
            all_predictions.append(prediction)
            all_actuals.append(current_actual_output)
        
        if all_predictions:
            all_predictions_np = np.array(all_predictions) # (num_windows, output_len, num_features)
            all_actuals_np = np.array(all_actuals)         # (num_windows, output_len, num_features)
            
            print(f"\nShape of all predictions: {all_predictions_np.shape}")
            print(f"Shape of all actuals: {all_actuals_np.shape}")

            # Calculate overall MSE for each of the 5 predicted steps
            for k in range(OUTPUT_LEN):
                mse_step_k = np.mean((all_predictions_np[:, k, :] - all_actuals_np[:, k, :])**2)
                print(f"Overall MSE for {k+1}-minute ahead prediction on test set: {mse_step_k:.4f}")
        else:
            print("No predictions made (this shouldn't happen if len(df_test) >= min_test_len).")
    else:
        print(f"Test set too small (length {len(df_test)}) for sliding window evaluation. "
              f"Need at least {min_test_len} data points.")