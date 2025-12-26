
import pandas as pd
import numpy as np

# --- Configuration ---
N_SAMPLES = 50000
FAILURE_RATE = 0.01 # 1% of events are failures
NOISE_LEVEL = 0.1

# --- Base sensor characteristics ---
BASE_TEMP = 25.0
BASE_VIBRATION = 0.1
BASE_ROTATION = 1500 # RPM

# --- Generate clean data ---
timestamps = pd.to_datetime(np.arange(N_SAMPLES), unit='m', origin=pd.Timestamp('2025-01-01'))
data = {
    'timestamp': timestamps,
    'temperature': np.random.normal(loc=BASE_TEMP, scale=2, size=N_SAMPLES),
    'vibration': np.random.normal(loc=BASE_VIBRATION, scale=0.05, size=N_SAMPLES),
    'rotation_speed': np.random.normal(loc=BASE_ROTATION, scale=50, size=N_SAMPLES),
    'failure': 0
}
df = pd.DataFrame(data)

# --- Introduce anomalies leading to failures ---
n_failures = int(N_SAMPLES * FAILURE_RATE)
failure_indices = np.random.choice(df.index, n_failures, replace=False)

for idx in failure_indices:
    # Mark the failure event
    if idx < N_SAMPLES -1:
        df.loc[idx, 'failure'] = 1
    
    # Create anomalous data in the 10-50 steps leading up to the failure
    pre_failure_window = np.random.randint(10, 51)
    start_anomaly = max(0, idx - pre_failure_window)
    
    # Temperature spike
    df.loc[start_anomaly:idx, 'temperature'] += np.linspace(0, np.random.uniform(15, 30), num=(idx - start_anomaly + 1))
    # Vibration increase
    df.loc[start_anomaly:idx, 'vibration'] += np.linspace(0, np.random.uniform(0.2, 0.5), num=(idx - start_anomaly + 1))
    # Rotation speed fluctuation
    df.loc[start_anomaly:idx, 'rotation_speed'] += np.random.normal(0, 200, size=(idx - start_anomaly + 1))


# --- Add some random noise to the whole dataset ---
df['temperature'] += np.random.normal(0, NOISE_LEVEL * 2, N_SAMPLES)
df['vibration'] += np.random.normal(0, NOISE_LEVEL * 0.1, N_SAMPLES)
df['rotation_speed'] += np.random.normal(0, NOISE_LEVEL * 10, N_SAMPLES)

# --- Save to CSV ---
output_filename = 'simulated_iot_device_data.csv'
df.to_csv(output_filename, index=False)

print(f"Successfully generated simulated IoT data and saved it to '{output_filename}'")
print("\n--- Data Head ---")
print(df.head())
print("\n--- Failure Distribution ---")
print(df['failure'].value_counts())
