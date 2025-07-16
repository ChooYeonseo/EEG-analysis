"""
Data visualization window for EEG Analysis GUI application.

This module contains the plotting window component with time navigation
and matplotlib integration for EEG data visualization.
"""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QGroupBox, QSplitter, QTextEdit,
                            QMessageBox, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from pathlib import Path

# Add utils directory to Python path
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
sys.path.insert(0, str(utils_dir))

try:
    import read_intan
    import plot_chuncked_data
except ImportError as e:
    print(f"Error importing utilities: {e}")


class PlottingWindow(QWidget):
    """Window for visualizing EEG data with time navigation."""
    
    def __init__(self, data_results):
        super().__init__()
        self.data_results = data_results
        self.current_data = None
        self.current_start_time = 0.0
        self.current_duration = 10.0
        self.sample_rate = None
        self.channel_data = {}
        
        self.setup_ui()
        self.process_data()
        
    def setup_ui(self):
        """Set up the plotting window UI."""
        self.setWindowTitle("EEG Data Visualization")
        self.setGeometry(150, 150, 1200, 800)
        
        # Apply Tokyo Night theme to this window
        from tokyo_night_theme import TOKYO_NIGHT_STYLES, TOKYO_NIGHT_COLORS
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Control panel for time navigation
        self.setup_control_panel(main_layout)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(TOKYO_NIGHT_STYLES['splitter'])
        
        # Plotting area (left side, larger)
        self.setup_plotting_area(splitter)
        
        # Info panel (right side, smaller)
        self.setup_info_panel(splitter)
        
        # Set splitter proportions (plot gets more space)
        splitter.setSizes([800, 300])
        main_layout.addWidget(splitter)
        
    def setup_control_panel(self, main_layout):
        """Set up the time navigation control panel."""
        control_group = QGroupBox("Time Navigation")
        control_layout = QHBoxLayout(control_group)
        
        # Navigation buttons
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.go_previous)
        self.prev_button.setEnabled(False)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.go_next)
        self.next_button.setEnabled(False)
        
        # Time input fields
        form_layout = QFormLayout()
        
        self.start_time_input = QLineEdit("0.0")
        self.start_time_input.setValidator(QDoubleValidator(0.0, 999999.0, 2))
        self.start_time_input.textChanged.connect(self.on_time_changed)
        
        self.duration_input = QLineEdit("10.0")
        self.duration_input.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        self.duration_input.textChanged.connect(self.on_time_changed)
        
        form_layout.addRow("Start Time (s):", self.start_time_input)
        form_layout.addRow("Duration (s):", self.duration_input)
        
        # Update button
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)
        self.update_button.setEnabled(False)
        
        # Layout arrangement
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.next_button)
        control_layout.addLayout(form_layout)
        control_layout.addWidget(self.update_button)
        control_layout.addStretch()
        
        main_layout.addWidget(control_group)
        
    def setup_plotting_area(self, splitter):
        """Set up the matplotlib plotting area."""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Create matplotlib figure and canvas with normal matplotlib styling
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Only style the widget container, not the matplotlib content
        from tokyo_night_theme import TOKYO_NIGHT_COLORS
        plot_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
                border: 1px solid {TOKYO_NIGHT_COLORS['border']};
                border-radius: 8px;
                padding: 5px;
            }}
        """)
        
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)
        
    def setup_info_panel(self, splitter):
        """Set up the data information panel."""
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumWidth(300)
        info_layout.addWidget(self.info_text)
        
        splitter.addWidget(info_group)
        
    def process_data(self):
        """Process the loaded data for visualization."""
        try:
            if not self.data_results:
                self.show_message("No data available", "No EEG data loaded.")
                return
                
            # Find the first file with data
            first_data_file = None
            for filename, result, data_present in self.data_results:
                if data_present:
                    first_data_file = (filename, result)
                    break
                    
            if not first_data_file:
                self.show_message("No data", "No files contain EEG data.")
                return
                
            filename, result = first_data_file
            self.current_data = result
            
            # Extract sample rate
            self.sample_rate = read_intan.get_sample_rate(result)
            if not self.sample_rate:
                self.show_message("Error", "Could not determine sample rate.")
                return
                
            # Extract channel data
            if 'amplifier_data' not in result:
                self.show_message("Error", "No amplifier data found.")
                return
                
            amplifier_data = result['amplifier_data']
            amplifier_channels = result.get('amplifier_channels', [])
            
            # Ensure amplifier_data is a numpy array
            if not isinstance(amplifier_data, np.ndarray):
                amplifier_data = np.array(amplifier_data)
            
            # Create time array
            num_samples = amplifier_data.shape[1]
            time_array = np.arange(num_samples, dtype=np.float64) / self.sample_rate
            
            # Organize data by channels
            self.channel_data = {'time': time_array}
            
            for i, channel_info in enumerate(amplifier_channels):
                if i < amplifier_data.shape[0]:
                    channel_name = (channel_info.get('custom_channel_name') or 
                                  channel_info.get('native_channel_name') or 
                                  f'Channel_{i}')
                    # Ensure channel data is float64 for consistency
                    channel_data = amplifier_data[i, :].astype(np.float64)
                    self.channel_data[channel_name] = channel_data
                    
            # Convert to DataFrame for easier handling
            self.df_data = pd.DataFrame(self.channel_data)
            
            # Verify data integrity
            if self.df_data.empty:
                self.show_message("Error", "No valid data could be processed.")
                return
                
            # Check for any NaN or infinite values
            if self.df_data.isnull().any().any():
                print("Warning: Found NaN values in data, filling with 0")
                self.df_data = self.df_data.fillna(0)
                
            if np.isinf(self.df_data.select_dtypes(include=[np.number]).values).any():
                print("Warning: Found infinite values in data, replacing with finite values")
                self.df_data = self.df_data.replace([np.inf, -np.inf], 0)
            
            # Update info panel
            self.update_info_display()
            
            # Enable controls
            self.update_button.setEnabled(True)
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            
            # Initial plot
            self.update_plot()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Data processing error details: {error_details}")
            self.show_message("Error", f"Error processing data: {str(e)}\n\nSee console for details.")
            
    def update_info_display(self):
        """Update the information display panel."""
        try:
            if not hasattr(self, 'df_data') or self.df_data is None or self.df_data.empty:
                self.info_text.setPlainText("No data loaded.")
                return
                
            max_time = float(self.df_data['time'].max())
            num_channels = len(self.df_data.columns) - 1  # Exclude time column
            num_samples = len(self.df_data)
            
            info_text = f"""
Data Information:
================

Total Duration: {max_time:.2f} seconds
Sample Rate: {self.sample_rate} Hz
Total Samples: {num_samples:,}
Number of Channels: {num_channels}

Current View:
------------
Start Time: {self.current_start_time:.2f}s
Duration: {self.current_duration:.2f}s
End Time: {self.current_start_time + self.current_duration:.2f}s

Channels:
---------
"""
            
            # List all channels except time
            channels = [col for col in self.df_data.columns if col != 'time']
            for i, channel in enumerate(channels):
                info_text += f"{i+1}. {channel}\n"
                
            self.info_text.setPlainText(info_text)
            
        except Exception as e:
            print(f"Error updating info display: {e}")
            self.info_text.setPlainText("Error displaying information.")
        
    def on_time_changed(self):
        """Handle changes in time input fields."""
        try:
            self.current_start_time = float(self.start_time_input.text() or "0.0")
            self.current_duration = float(self.duration_input.text() or "10.0")
            self.update_info_display()
        except ValueError:
            pass  # Invalid input, ignore
            
    def go_previous(self):
        """Navigate to previous time segment."""
        new_start = self.current_start_time - self.current_duration
        if new_start < 0:
            new_start = 0
        self.current_start_time = new_start
        self.start_time_input.setText(f"{new_start:.2f}")
        self.update_plot()
        
    def go_next(self):
        """Navigate to next time segment."""
        if not hasattr(self, 'df_data'):
            return
            
        max_time = self.df_data['time'].max()
        new_start = self.current_start_time + self.current_duration
        
        if new_start + self.current_duration > max_time:
            new_start = max(0, max_time - self.current_duration)
            
        self.current_start_time = new_start
        self.start_time_input.setText(f"{new_start:.2f}")
        self.update_plot()
        
    def update_plot(self):
        """Update the matplotlib plot with current time range."""
        try:
            if not hasattr(self, 'df_data') or self.df_data is None:
                return
                
            # Get current time parameters
            self.on_time_changed()
            
            start_time = self.current_start_time
            end_time = start_time + self.current_duration
            
            # Clear the figure (use default matplotlib styling)
            self.figure.clear()
            
            # Filter data for time range - fix the boolean ambiguity issue
            time_array = self.df_data['time'].values
            time_mask = (time_array >= start_time) & (time_array <= end_time)
            filtered_data = self.df_data[time_mask].copy()
            
            if len(filtered_data) == 0:
                self.figure.text(0.5, 0.5, 'No data in selected time range', 
                               ha='center', va='center', fontsize=14)
                self.canvas.draw()
                return
                
            # Get channel columns (exclude time)
            channels = [col for col in self.df_data.columns if col != 'time']
            n_channels = len(channels)
            
            if n_channels == 0:
                self.figure.text(0.5, 0.5, 'No channel data available', 
                               ha='center', va='center', fontsize=14)
                self.canvas.draw()
                return
                
            # Create subplots with normal matplotlib styling
            axes = self.figure.subplots(n_channels, 1, sharex=True)
            if n_channels == 1:
                axes = [axes]
                
            # Calculate global y-axis range
            all_values = []
            for channel in channels:
                channel_data = filtered_data[channel].values
                if len(channel_data) > 0:
                    all_values.extend(channel_data)
                
            if len(all_values) > 0:
                all_values = np.array(all_values)
                global_min = np.nanmin(all_values)
                global_max = np.nanmax(all_values)
                
                # Add margin and handle edge cases
                if global_min == global_max:
                    y_margin = abs(global_min) * 0.1 or 1.0
                    y_min = global_min - y_margin
                    y_max = global_max + y_margin
                else:
                    y_margin = (global_max - global_min) * 0.1
                    y_min = global_min - y_margin
                    y_max = global_max + y_margin
            else:
                y_min, y_max = -100, 100
                
            # Plot each channel with normal matplotlib colors
            for i, channel in enumerate(channels):
                ax = axes[i]
                
                # Get channel data safely
                time_data = filtered_data['time'].values
                channel_data = filtered_data[channel].values
                
                if len(time_data) > 0 and len(channel_data) > 0:
                    # Use default matplotlib color cycle
                    ax.plot(time_data, channel_data, 
                           linewidth=0.8, color=f'C{i % 10}')
                
                # Use normal matplotlib styling
                ax.set_ylabel(f'{channel}\n(µV)', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(start_time, end_time)
                
                # Add channel title with default styling
                ax.text(0.02, 0.95, channel, transform=ax.transAxes, 
                       fontweight='bold', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
            # Set x-axis label on bottom subplot
            if len(axes) > 0:
                axes[-1].set_xlabel('Time (s)', fontsize=10)
                
            # Set overall title with normal styling
            self.figure.suptitle(f'EEG Signals ({start_time:.2f}-{end_time:.2f} seconds)', 
                               fontsize=12, fontweight='bold')
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update info display
            self.update_info_display()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Plotting error details: {error_details}")
            self.show_message("Plotting Error", f"Error updating plot: {str(e)}\n\nSee console for details.")
            
    def show_message(self, title, message):
        """Show a message box."""
        QMessageBox.information(self, title, message)
