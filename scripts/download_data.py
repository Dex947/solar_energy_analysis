"""
Solar Energy Data Downloader
Downloads and prepares solar power generation data from Kaggle.

Author: Bryant M.
Date: July 2025
Project: Solar Energy Performance Analysis
"""

import os
import sys
import zipfile
import pandas as pd
import kaggle
from pathlib import Path

class SolarDataDownloader:
    """Handles downloading and initial setup of solar power generation data."""
    
    def __init__(self, project_root=None):
        """Initialize the data downloader.
        
        Args:
            project_root (str): Root directory of the project
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.processed_data_dir = self.project_root / "data" / "processed"
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def check_kaggle_setup(self):
        """Verify Kaggle API credentials are configured."""
        try:
            kaggle.api.authenticate()
            print("✅ Kaggle API authenticated successfully")
            return True
        except Exception as e:
            print(f"❌ Kaggle authentication failed: {e}")
            print("Please ensure ~/.kaggle/kaggle.json is configured with your API credentials")
            print("Instructions: https://github.com/Kaggle/kaggle-api#api-credentials")
            return False
    
    def download_solar_dataset(self):
        """Download solar power generation dataset from Kaggle."""
        if not self.check_kaggle_setup():
            return False
        
        try:
            print("📥 Downloading solar power generation dataset...")
            
            # Download the dataset
            kaggle.api.dataset_download_files(
                'anikannal/solar-power-generation-data',
                path=self.raw_data_dir,
                unzip=True
            )
            
            print(f"✅ Dataset downloaded to: {self.raw_data_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def verify_data_files(self):
        """Verify all expected data files are present."""
        expected_files = [
            "Plant_1_Generation_Data.csv",
            "Plant_2_Generation_Data.csv", 
            "Plant_1_Weather_Sensor_Data.csv",
            "Plant_2_Weather_Sensor_Data.csv"
        ]
        
        missing_files = []
        present_files = []
        
        for file in expected_files:
            file_path = self.raw_data_dir / file
            if file_path.exists():
                present_files.append(file)
                print(f"✅ Found: {file}")
            else:
                missing_files.append(file)
                print(f"❌ Missing: {file}")
        
        if missing_files:
            print(f"\n⚠️  Missing {len(missing_files)} expected files")
            print("Available files in data/raw/:")
            for file in self.raw_data_dir.glob("*.csv"):
                print(f"   📄 {file.name}")
        else:
            print(f"\n✅ All {len(expected_files)} data files found!")
        
        return len(missing_files) == 0
    
    def preview_data_structure(self):
        """Display basic information about the downloaded datasets."""
        print("\n📊 DATASET PREVIEW")
        print("=" * 50)
        
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        
        for file_path in csv_files:
            print(f"\n📄 {file_path.name}")
            print("-" * 30)
            
            try:
                df = pd.read_csv(file_path)
                print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                print(f"Columns: {', '.join(df.columns.tolist())}")
                print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                # Check for missing values
                missing = df.isnull().sum().sum()
                if missing > 0:
                    print(f"Missing values: {missing:,}")
                else:
                    print("Missing values: None")
                    
            except Exception as e:
                print(f"Error reading file: {e}")
    
    def create_sample_data(self):
        """Create sample data if Kaggle download isn't available."""
        print("🔧 Creating sample solar power data for testing...")
        
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate 30 days of sample data
        start_date = datetime(2020, 5, 15)
        dates = [start_date + timedelta(days=d, minutes=m*15) 
                for d in range(30) for m in range(96)]  # 15-min intervals
        
        # Simulate solar generation patterns
        np.random.seed(42)
        n_records = len(dates)
        
        # Plant 1 Generation Data
        plant1_gen = pd.DataFrame({
            'DATE_TIME': dates,
            'SOURCE_KEY': ['1BY6WEcLGh8j5v7' for _ in range(n_records)],
            'DC_POWER': np.random.normal(3000, 1200, n_records).clip(0, None),
            'AC_POWER': np.random.normal(2850, 1150, n_records).clip(0, None),
            'DAILY_YIELD': np.random.normal(55, 20, n_records).clip(0, None),
            'TOTAL_YIELD': np.cumsum(np.random.normal(0.5, 0.2, n_records).clip(0, None))
        })
        
        # Plant 2 Generation Data  
        plant2_gen = pd.DataFrame({
            'DATE_TIME': dates,
            'SOURCE_KEY': ['3by6WCclGe8k7p9' for _ in range(n_records)],
            'DC_POWER': np.random.normal(3200, 1300, n_records).clip(0, None),
            'AC_POWER': np.random.normal(3050, 1250, n_records).clip(0, None),
            'DAILY_YIELD': np.random.normal(58, 22, n_records).clip(0, None),
            'TOTAL_YIELD': np.cumsum(np.random.normal(0.55, 0.25, n_records).clip(0, None))
        })
        
        # Weather sensor data
        weather_data = {
            'DATE_TIME': dates,
            'PLANT_ID': 4135001,
            'SOURCE_KEY': ['1BY6WEcLGh8j5v7' for _ in range(n_records)],
            'AMBIENT_TEMPERATURE': np.random.normal(25, 8, n_records).clip(-5, 45),
            'MODULE_TEMPERATURE': np.random.normal(35, 12, n_records).clip(0, 65),
            'IRRADIATION': np.random.normal(0.5, 0.3, n_records).clip(0, 1.2)
        }
        
        plant1_weather = pd.DataFrame(weather_data)
        plant2_weather = pd.DataFrame(weather_data)
        plant2_weather['SOURCE_KEY'] = '3by6WCclGe8k7p9'
        plant2_weather['PLANT_ID'] = 4135002
        
        # Save sample data
        plant1_gen.to_csv(self.raw_data_dir / "Plant_1_Generation_Data.csv", index=False)
        plant2_gen.to_csv(self.raw_data_dir / "Plant_2_Generation_Data.csv", index=False)
        plant1_weather.to_csv(self.raw_data_dir / "Plant_1_Weather_Sensor_Data.csv", index=False)
        plant2_weather.to_csv(self.raw_data_dir / "Plant_2_Weather_Sensor_Data.csv", index=False)
        
        print("✅ Sample data created successfully!")

def main():
    """Main execution function."""
    print("🌞 SOLAR ENERGY DATA DOWNLOADER")
    print("=" * 40)
    
    downloader = SolarDataDownloader()
    
    # Try to download from Kaggle first
    if downloader.download_solar_dataset():
        print("\n✅ Kaggle download successful!")
    else:
        print("\n⚠️  Kaggle download failed. Creating sample data...")
        downloader.create_sample_data()
    
    # Verify files exist
    print("\n🔍 VERIFYING DATA FILES")
    print("-" * 25)
    if downloader.verify_data_files():
        downloader.preview_data_structure()
        print("\n🎉 Data setup complete! Ready for analysis.")
    else:
        print("\n❌ Data setup incomplete. Please check the files manually.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())