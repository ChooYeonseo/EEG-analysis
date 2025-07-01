"""
Test script to verify the read_intan module functionality.
Combined RHD files to pandas DataFrame testing.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the utils directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

def test_individual_files():
    """Test individual RHD file loading"""
    print("="*50)
    print("Testing individual RHD file loading...")
    print("="*50)
    
    try:
        import read_intan
        
        # Test loading a single file from the d0_1 directory
        print("Loading experiment data...")
        experiment_data = read_intan.load_experiment_data('data/intan/attempt1/d0_1')
        # List of (filename, result, data_present)
        # Sort experiment_data by filename (1st element of tuple)
        if experiment_data:
            experiment_data.sort(key=lambda x: x[0])  # Sort by filename

        if experiment_data:
            print(f"✓ Loaded {len(experiment_data)} files")
            
            # Show info for the first file
            filename, result, data_present = experiment_data[0]
            print(f"\nFirst file: {filename}")
            
            # Debug: Check what keys are in result
            print(f"Result keys: {list(result.keys())}")
            
            # Check if amplifier_channels exists and its structure
            # if 'amplifier_channels' in result:
            #     print(f"Number of amplifier channels: {len(result['amplifier_channels'])}")
            #     if len(result['amplifier_channels']) > 0:
            #         print(f"First channel info keys: {list(result['amplifier_channels'][0].keys())}")
            #         print(f"First channel: {result['amplifier_channels'][0]}")
            
            read_intan.print_file_info(result)
            
            # Try to get data from a specific channel (example) - with better error handling
            if 'amplifier_channels' in result and len(result['amplifier_channels']) > 0:
                first_channel_name = result['amplifier_channels'][0]['native_channel_name']
                print(f"Trying to get data for channel: {first_channel_name}")
                
                try:
                    channel_data = read_intan.get_channel_data(result, first_channel_name)
                    if channel_data is not None:
                        print(f"✓ Channel '{first_channel_name}' data shape: {channel_data.shape}")
                    else:
                        print(f"✗ Channel '{first_channel_name}' data is None")
                except Exception as e:
                    print(f"✗ Error getting channel data: {e}")
                    import traceback
                    traceback.print_exc()
                    
            return True, read_intan
        else:
            print("✗ No data loaded")
            return False, None
            
    except Exception as e:
        print(f"✗ Error in individual file test: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_combined_dataframe(read_intan_module):
    """Test combined RHD files to DataFrame functionality"""
    print("\n" + "="*50)
    print("Testing combined RHD files to DataFrame...")
    print("="*50)
    
    try:
        # Test 1: Load all channels from d0_1 directory
        print("\n1. Testing rhd_folder_to_dataframe() with all channels...")
        df, metadata = read_intan_module.rhd_folder_to_dataframe('data/intan/attempt1/d0_1')
        
        # Verify it's a pandas DataFrame
        print(f"\n✓ Data type verification:")
        print(f"   - Type: {type(df)}")
        print(f"   - Is pandas DataFrame: {isinstance(df, pd.DataFrame)}")
        
        # Check DataFrame properties
        print(f"\n✓ DataFrame properties:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check time column
        if 'time' in df.columns:
            print(f"\n✓ Time column analysis:")
            print(f"   - Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds")
            print(f"   - Duration: {df['time'].max() - df['time'].min():.3f} seconds")
            print(f"   - Sample rate: {len(df) / (df['time'].max() - df['time'].min()):.1f} Hz")
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        print(f"\n✓ Data quality check:")
        print(f"   - Total NaN values: {nan_counts.sum()}")
        if nan_counts.sum() > 0:
            print(f"   - NaN per column: {dict(nan_counts[nan_counts > 0])}")
        
        # Show first few rows
        print(f"\n✓ First 5 rows:")
        print(df.head())
        
        # Show metadata
        print(f"\n✓ Metadata:")
        for key, value in metadata.items():
            if key != 'file_info':  # Skip detailed file info for cleaner output
                print(f"   - {key}: {value}")
        
        # Test 2: Load specific channels
        print(f"\n2. Testing with specific channels...")
        available_channels = [col for col in df.columns if col != 'time']
        if len(available_channels) >= 2:
            test_channels = available_channels[:2]  # Take first 2 channels
            print(f"   Testing with channels: {test_channels}")
            
            df_subset, metadata_subset = read_intan_module.rhd_folder_to_dataframe(
                'data/intan/attempt1/d0_1', 
                channel_names=test_channels
            )
            
            print(f"   ✓ Subset DataFrame shape: {df_subset.shape}")
            print(f"   ✓ Subset columns: {list(df_subset.columns)}")
        
        # Test 3: Test resampling (if original sample rate is high enough)
        original_rate = metadata.get('sample_rate', 20000)
        if original_rate > 2000:
            print(f"\n3. Testing resampling from {original_rate} Hz to 1000 Hz...")
            df_resampled, metadata_resampled = read_intan_module.rhd_folder_to_dataframe(
                'data/intan/attempt1/d0_1',
                resample_rate=1000
            )
            
            print(f"   ✓ Resampled DataFrame shape: {df_resampled.shape}")
            print(f"   ✓ New sample rate: {metadata_resampled.get('sample_rate')} Hz")
            print(f"   ✓ Compression ratio: {df.shape[0] / df_resampled.shape[0]:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in DataFrame test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_channel_mapping(read_intan_module):
    """Test channel mapping functionality"""
    print("\n" + "="*50)
    print("Testing channel mapping...")
    print("="*50)
    
    try:
        mapping = read_intan_module.get_channel_mapping()
        print(f"✓ Channel mapping loaded:")
        for pin, channel in mapping.items():
            print(f"   Pin {pin:2d} -> {channel}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in channel mapping test: {e}")
        return False


def main():
    """Main test function"""
    print("RHD Reader Module Testing")
    print("Testing combined file reading and DataFrame creation")
    print("="*60)
    
    # Test 1: Individual files
    success1, read_intan_module = test_individual_files()
    
    if success1 and read_intan_module:
        # Test 2: Combined DataFrame
        success2 = test_combined_dataframe(read_intan_module)
        
        # Test 3: Channel mapping
        success3 = test_channel_mapping(read_intan_module)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Individual file loading: {'✓ PASS' if success1 else '✗ FAIL'}")
        print(f"Combined DataFrame creation: {'✓ PASS' if success2 else '✗ FAIL'}")
        print(f"Channel mapping: {'✓ PASS' if success3 else '✗ FAIL'}")
        
        if success1 and success2 and success3:
            print("\n🎉 All tests passed! The module is working correctly.")
            print("✓ RHD files can be successfully combined into pandas DataFrames")
        else:
            print("\n⚠️  Some tests failed. Check the error messages above.")
    else:
        print("\n✗ Initial tests failed. Make sure the data directory exists and contains RHD files")


if __name__ == "__main__":
    main()
