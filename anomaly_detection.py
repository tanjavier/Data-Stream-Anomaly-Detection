import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
import logging
import json
from datetime import datetime
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class DataStreamSimulator:
    """Simulates a real-time data stream with complex patterns and anomalies."""
    
    def __init__(self, seasonal_periods=[100, 50], noise_level=0.2, anomaly_rate=0.05, trend_factor=0.001):
        self.seasonal_periods = seasonal_periods  # Multiple seasonal periods
        self.noise_level = noise_level
        self.anomaly_rate = anomaly_rate
        self.trend_factor = trend_factor
        self.t = 0
        
    def get_next_value(self):
        """Generate next value with multiple seasonal patterns, trend, and possible anomaly."""
        # Multiple seasonal patterns
        seasonal = sum(np.sin(2 * np.pi * self.t / period) for period in self.seasonal_periods)
        
        # Add trend
        trend = self.trend_factor * self.t
        
        # Add random noise
        noise = np.random.normal(0, self.noise_level)
        
        # Complex anomaly patterns
        if np.random.random() < self.anomaly_rate:
            anomaly_types = ['spike', 'level_shift', 'trend_change']
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'spike':
                anomaly = np.random.choice([-1, 1]) * np.random.uniform(2, 4)
            elif anomaly_type == 'level_shift':
                anomaly = np.random.choice([-1, 1]) * 2
                self.trend_factor += anomaly * 0.0001
            else:  # trend_change
                self.trend_factor *= np.random.choice([0.5, 1.5])
                anomaly = 0
        else:
            anomaly = 0
            
        self.t += 1
        return seasonal + trend + noise + anomaly

class AnomalyDetector:
    """Enhanced anomaly detection using multiple algorithms."""
    
    def __init__(self, window_size=50, threshold_factor=3, contamination=0.1):
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.values = deque(maxlen=window_size)
        self.mean = 0
        self.std = 1
        
        # Initialize Isolation Forest
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
        # Initialize EWMA parameters
        self.ewma = 0
        self.alpha = 0.1
        self.ewma_std = 1
        
    def update_statistics(self):
        """Update all statistical measures."""
        if len(self.values) >= 2:
            values_array = np.array(self.values).reshape(-1, 1)
            
            # Traditional statistics
            self.mean = np.mean(self.values)
            self.std = np.std(self.values)
            
            # EWMA update
            if len(self.values) == 1:
                self.ewma = self.values[0]
            else:
                self.ewma = self.alpha * self.values[-1] + (1 - self.alpha) * self.ewma
                self.ewma_std = np.sqrt(self.alpha * (1 - (1 - self.alpha)**(2 * len(self.values))) / (2 - self.alpha))
            
            # Retrain Isolation Forest periodically
            if len(self.values) == self.window_size:
                scaled_values = self.scaler.fit_transform(values_array)
                self.isolation_forest.fit(scaled_values)
    
    def is_anomaly(self, value):
        """Detect anomalies using multiple methods."""
        self.values.append(value)
        self.update_statistics()
        
        if len(self.values) < self.window_size:
            return False, {}
        
        # Z-score method
        z_score = abs(value - self.mean) / (self.std + 1e-10)
        z_score_anomaly = z_score > self.threshold_factor
        
        # EWMA method
        ewma_score = abs(value - self.ewma) / (self.ewma_std + 1e-10)
        ewma_anomaly = ewma_score > self.threshold_factor
        
        # Isolation Forest method
        scaled_value = self.scaler.transform([[value]])
        isolation_forest_pred = self.isolation_forest.predict(scaled_value)[0] == -1
        
        # Combine results
        is_anomaly = any([z_score_anomaly, ewma_anomaly, isolation_forest_pred])
        
        return is_anomaly, {
            'z_score': z_score,
            'ewma_score': ewma_score,
            'isolation_forest': isolation_forest_pred,
            'current_mean': self.mean,
            'current_std': self.std,
            'ewma': self.ewma
        }

class DataLogger:
    """Handles data persistence and logging with proper NumPy type conversion."""
    
    def __init__(self, log_file='anomaly_detection.log', data_file='stream_data.json'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.data_file = data_file
        self.buffer = []
        self.buffer_size = 100
        
    def _convert_to_serializable(self, obj):
        """Convert NumPy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
        
    def log_datapoint(self, timestamp, value, anomaly_info):
        """Log a single datapoint and its anomaly status with type conversion."""
        record = {
            'timestamp': timestamp,
            'value': float(value),  # Convert to native Python float
            'anomaly_info': self._convert_to_serializable(anomaly_info),
            'datetime': datetime.now().isoformat()
        }
        
        self.buffer.append(record)
        
        if anomaly_info.get('is_anomaly', False):
            logging.info(f"Anomaly detected: {record}")
        
        if len(self.buffer) >= self.buffer_size:
            self.flush_buffer()
            
    def flush_buffer(self):
        """Write buffered data to file."""
        if not self.buffer:
            return
            
        try:
            mode = 'a' if os.path.exists(self.data_file) else 'w'
            with open(self.data_file, mode) as f:
                for record in self.buffer:
                    f.write(json.dumps(record) + '\n')
            
            self.buffer = []
        except Exception as e:
            logging.error(f"Error writing to file: {str(e)}")
            # Keep the buffer in case of write failure
            print(f"Error writing to file: {str(e)}")

class RealTimeVisualizer:
    """Enhanced real-time visualization with proper window closing handling."""
    
    def __init__(self, max_points=200):
        self.max_points = max_points
        self.timestamps = deque(maxlen=max_points)
        self.values = deque(maxlen=max_points)
        self.anomalies_x = deque(maxlen=max_points)
        self.anomalies_y = deque(maxlen=max_points)
        self.metrics = {
            'z_scores': deque(maxlen=max_points),
            'ewma': deque(maxlen=max_points),
            'means': deque(maxlen=max_points),
            'stds': deque(maxlen=max_points)
        }
        
        # Setup plots with smaller figure size
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        
        # Add window close event handler
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Flag to track if window is closed
        self.running = True
        
        self.setup_subplots()
        
    def on_close(self, event):
        """Handle window close event."""
        self.running = False
        plt.close('all')  # Close all matplotlib windows
        
    def is_running(self):
        """Check if the visualization window is still open."""
        return self.running
        
    def setup_subplots(self):
        """Initialize multiple subplots for different metrics."""
        # Main data stream plot
        self.ax1 = self.fig.add_subplot(311)
        self.line_data, = self.ax1.plot([], [], 'b-', label='Data Stream')
        self.line_ewma, = self.ax1.plot([], [], 'g-', label='EWMA', alpha=0.5)
        self.anomaly_scatter = self.ax1.scatter([], [], color='red', marker='o', label='Anomalies')
        self.ax1.set_title('Real-time Data Stream')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Z-scores plot
        self.ax2 = self.fig.add_subplot(312)
        self.line_zscore, = self.ax2.plot([], [], 'r-', label='Z-Score')
        self.ax2.axhline(y=3, color='k', linestyle='--', alpha=0.5)
        self.ax2.set_title('Z-Scores')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Standard deviation plot
        self.ax3 = self.fig.add_subplot(313)
        self.line_std, = self.ax3.plot([], [], 'm-', label='Rolling StdDev')
        self.ax3.set_title('Rolling Standard Deviation')
        self.ax3.legend()
        self.ax3.grid(True)
        
        plt.tight_layout()
        
    def update(self, timestamp, value, anomaly_data):
        """Update all visualizations with new data."""
        # Update data collections
        self.timestamps.append(timestamp)
        self.values.append(value)
        
        if anomaly_data.get('is_anomaly', False):
            self.anomalies_x.append(timestamp)
            self.anomalies_y.append(value)
            
        self.metrics['z_scores'].append(anomaly_data.get('z_score', 0))
        self.metrics['ewma'].append(anomaly_data.get('ewma', value))
        self.metrics['means'].append(anomaly_data.get('current_mean', value))
        self.metrics['stds'].append(anomaly_data.get('current_std', 0))
        
        # Update main plot
        self.line_data.set_data(list(self.timestamps), list(self.values))
        self.line_ewma.set_data(list(self.timestamps), list(self.metrics['ewma']))
        self.anomaly_scatter.set_offsets(np.c_[list(self.anomalies_x), list(self.anomalies_y)])
        
        # Update z-score plot
        self.line_zscore.set_data(list(self.timestamps), list(self.metrics['z_scores']))
        
        # Update std plot
        self.line_std.set_data(list(self.timestamps), list(self.metrics['stds']))
        
        # Adjust plot limits
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()
            if len(self.timestamps) > 0:
                ax.set_xlim(max(0, self.timestamps[-1] - self.max_points), self.timestamps[-1] + 5)
        
        plt.draw()
        plt.pause(0.01)

def main():
    """Enhanced main function with proper window closing handling."""
    # Initialize components
    simulator = DataStreamSimulator(
        seasonal_periods=[100, 50, 25],
        noise_level=0.2,
        anomaly_rate=0.05,
        trend_factor=0.001
    )
    detector = AnomalyDetector(window_size=50, threshold_factor=3)
    visualizer = RealTimeVisualizer(max_points=200)
    logger = DataLogger()
    
    try:
        timestamp = 0
        while visualizer.is_running():  # Check if visualization window is still open
            # Get next value from stream
            value = simulator.get_next_value()
            
            # Check for anomaly
            is_anomaly, anomaly_info = detector.is_anomaly(value)
            anomaly_info['is_anomaly'] = is_anomaly
            
            # Update visualization if window is still open
            if visualizer.is_running():
                visualizer.update(timestamp, value, anomaly_info)
            else:
                break
            
            # Log data
            logger.log_datapoint(timestamp, value, anomaly_info)
            
            # Print detection results
            if is_anomaly:
                print(f"Anomaly detected at timestamp {timestamp}: {value:.2f}")
                print(f"Detection details: {anomaly_info}")
            
            timestamp += 1
            time.sleep(0.05)  # Control stream speed
            
    except KeyboardInterrupt:
        print("\nStopping data stream...")
    finally:
        logger.flush_buffer()  # Ensure remaining data is written
        plt.close('all')  # Ensure all matplotlib windows are closed
        print("Program terminated.")

if __name__ == "__main__":
    main()
    