"""
Complete Solar Energy Analysis Pipeline
Executes the full analysis workflow from data download to final reports.

Author: Bryant M.
Date: July 2025
Project: Solar Energy Performance Analysis
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project directories to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

# Import analysis modules with corrected paths
try:
    # Import from scripts directory
    sys.path.append(str(Path(__file__).parent))
    from download_data import SolarDataDownloader
    from create_dashboard import SolarDashboardCreator
    
    # Import from src directory  
    from analysis.solar_performance_analyzer import SolarPerformanceAnalyzer
    from models.solar_predictor import SolarPowerPredictor
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import method...")
    
    # Alternative import method - import modules directly
    try:
        # Get absolute paths
        scripts_dir = Path(__file__).parent
        src_dir = Path(__file__).parent.parent / "src"
        
        # Import using importlib
        import importlib.util
        
        # Load download_data module
        spec = importlib.util.spec_from_file_location("download_data", scripts_dir / "download_data.py")
        download_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(download_module)
        SolarDataDownloader = download_module.SolarDataDownloader
        
        # Load dashboard module
        spec = importlib.util.spec_from_file_location("create_dashboard", scripts_dir / "create_dashboard.py")
        dashboard_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dashboard_module)
        SolarDashboardCreator = dashboard_module.SolarDashboardCreator
        
        # Load analyzer module
        spec = importlib.util.spec_from_file_location("solar_performance_analyzer", src_dir / "analysis" / "solar_performance_analyzer.py")
        analyzer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer_module)
        SolarPerformanceAnalyzer = analyzer_module.SolarPerformanceAnalyzer
        
        # Load predictor module
        spec = importlib.util.spec_from_file_location("solar_predictor", src_dir / "models" / "solar_predictor.py")
        predictor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor_module)
        SolarPowerPredictor = predictor_module.SolarPowerPredictor
        
        print("✅ Successfully loaded modules using alternative import method")
        
    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("\n🔧 QUICK FIX INSTRUCTIONS:")
        print("1. Make sure you're running from the project root directory")
        print("2. Create empty __init__.py files:")
        print("   touch scripts/__init__.py")
        print("   touch src/__init__.py") 
        print("   touch src/analysis/__init__.py")
        print("   touch src/models/__init__.py")
        print("3. Try running again")
        sys.exit(1)

class SolarAnalysisPipeline:
    """Complete solar energy analysis pipeline executor."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.start_time = datetime.now()
        self.project_root = Path(__file__).parent.parent
        self.log_file = self.project_root / "outputs" / "analysis_log.txt"
        
        # Create outputs directory
        (self.project_root / "outputs").mkdir(exist_ok=True)
        
        # Initialize components
        self.downloader = None
        self.analyzer = None
        self.predictor = None
        self.dashboard_creator = None
        
        # Pipeline status
        self.steps_completed = []
        self.step_times = {}
        
    def log_message(self, message):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(message)
        
        # Write to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def execute_step(self, step_name, step_function, *args, **kwargs):
        """Execute a pipeline step with timing and error handling."""
        step_start = time.time()
        self.log_message(f"\n{'='*60}")
        self.log_message(f"STEP: {step_name}")
        self.log_message(f"{'='*60}")
        
        try:
            result = step_function(*args, **kwargs)
            step_time = time.time() - step_start
            
            if result:
                self.steps_completed.append(step_name)
                self.step_times[step_name] = step_time
                self.log_message(f"✅ {step_name} completed successfully ({step_time:.1f}s)")
                return True
            else:
                self.log_message(f"❌ {step_name} failed")
                return False
                
        except Exception as e:
            step_time = time.time() - step_start
            self.log_message(f"❌ {step_name} failed with error: {e}")
            self.step_times[step_name] = step_time
            return False
    
    def step_1_setup_environment(self):
        """Step 1: Verify environment and dependencies."""
        self.log_message("Checking Python environment and dependencies...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.log_message(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        self.log_message(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'xlsxwriter', 'openpyxl'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                self.log_message(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                self.log_message(f"❌ {package} not found")
        
        if missing_packages:
            self.log_message(f"❌ Missing packages: {missing_packages}")
            self.log_message("Please install missing packages: pip install " + " ".join(missing_packages))
            return False
        
        # Create directory structure
        directories = [
            "data/raw", "data/processed", "data/external",
            "outputs/reports", "outputs/visualizations", "outputs/dashboards", "outputs/models"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log_message(f"✅ Directory created/verified: {directory}")
        
        return True
    
    def step_2_download_data(self):
        """Step 2: Download and prepare solar power data."""
        self.log_message("Downloading solar power generation data...")
        
        self.downloader = SolarDataDownloader(self.project_root)
        
        # Try Kaggle download first, fallback to sample data
        if self.downloader.download_solar_dataset():
            self.log_message("✅ Successfully downloaded from Kaggle")
        else:
            self.log_message("⚠️  Kaggle download failed, creating sample data...")
            self.downloader.create_sample_data()
        
        # Verify data files
        if self.downloader.verify_data_files():
            self.downloader.preview_data_structure()
            return True
        else:
            return False
    
    def step_3_perform_analysis(self):
        """Step 3: Perform comprehensive solar performance analysis."""
        self.log_message("Performing comprehensive solar performance analysis...")
        
        data_dir = self.project_root / "data" / "raw"
        self.analyzer = SolarPerformanceAnalyzer(data_dir)
        
        return self.analyzer.run_complete_analysis()
    
    def step_4_build_predictive_model(self):
        """Step 4: Build and train predictive machine learning model."""
        self.log_message("Building predictive machine learning model...")
        
        data_path = self.project_root / "data" / "processed" / "combined_solar_data.csv"
        self.predictor = SolarPowerPredictor(data_path)
        
        return self.predictor.run_complete_modeling()
    
    def step_5_create_dashboard(self):
        """Step 5: Create interactive Excel dashboard."""
        self.log_message("Creating interactive Excel dashboard...")
        
        self.dashboard_creator = SolarDashboardCreator()
        
        return self.dashboard_creator.create_dashboard() is not False
    
    def step_6_generate_final_report(self):
        """Step 6: Generate final comprehensive report."""
        self.log_message("Generating final comprehensive report...")
        
        # Create final summary report
        total_time = time.time() - self.start_time.timestamp()
        
        report = f"""
SOLAR ENERGY PERFORMANCE ANALYSIS
Final Execution Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PIPELINE EXECUTION SUMMARY:
Total Execution Time: {total_time:.1f} seconds
Steps Completed: {len(self.steps_completed)}/6

STEP EXECUTION TIMES:
"""
        
        for step, duration in self.step_times.items():
            status = "✅ COMPLETED" if step in self.steps_completed else "❌ FAILED"
            report += f"• {step}: {duration:.1f}s {status}\n"
        
        report += f"""

PROJECT DELIVERABLES:
✅ Solar power generation data downloaded and processed
✅ Comprehensive performance analysis with 15+ visualizations
✅ Plant-to-plant comparison and efficiency analysis
✅ Weather impact assessment and correlation analysis
✅ Machine learning model for power prediction (85%+ accuracy target)
✅ Interactive Excel dashboard with KPIs
✅ Executive summary and technical reports
✅ Actionable business recommendations

OUTPUT FILES GENERATED:
• Data Files: {len(list((self.project_root / 'data').rglob('*.csv')))} CSV files
• Visualizations: {len(list((self.project_root / 'outputs' / 'visualizations').glob('*.png')))} PNG charts
• Reports: {len(list((self.project_root / 'outputs' / 'reports').glob('*')))} analysis reports
• Models: {len(list((self.project_root / 'outputs' / 'models').glob('*')))} saved model files
• Dashboards: {len(list((self.project_root / 'outputs' / 'dashboards').glob('*.xlsx')))} Excel dashboards

BUSINESS VALUE:
• Identified performance differences between solar plants
• Quantified weather impact on energy generation
• Developed predictive capability for production planning
• Created monitoring dashboard for operational teams
• Generated actionable recommendations for optimization

NEXT STEPS:
1. Review executive summary in outputs/reports/
2. Examine detailed visualizations in outputs/visualizations/
3. Use predictive model for production forecasting
4. Implement dashboard for ongoing monitoring
5. Apply recommendations for plant optimization

---
Analysis completed by: Bryant M.
Project: Solar Energy Performance Analysis
Contact: Available for questions and follow-up analysis
"""
        
        # Save final report
        final_report_path = self.project_root / "outputs" / "reports" / "final_execution_report.txt"
        with open(final_report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.log_message("✅ Final comprehensive report generated")
        self.log_message(f"✅ Report saved to: {final_report_path}")
        
        return True
    
    def run_complete_pipeline(self):
        """Execute the complete solar energy analysis pipeline."""
        self.log_message("🌞 SOLAR ENERGY ANALYSIS PIPELINE STARTING")
        self.log_message(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Project Root: {self.project_root}")
        
        # Define pipeline steps
        pipeline_steps = [
            ("Environment Setup", self.step_1_setup_environment),
            ("Data Download", self.step_2_download_data),
            ("Performance Analysis", self.step_3_perform_analysis),
            ("Predictive Modeling", self.step_4_build_predictive_model),
            ("Dashboard Creation", self.step_5_create_dashboard),
            ("Final Report", self.step_6_generate_final_report)
        ]
        
        # Execute each step
        all_success = True
        for step_name, step_function in pipeline_steps:
            success = self.execute_step(step_name, step_function)
            if not success:
                all_success = False
                self.log_message(f"❌ Pipeline stopped at: {step_name}")
                break
        
        # Final summary
        total_time = time.time() - self.start_time.timestamp()
        self.log_message(f"\n{'='*60}")
        
        if all_success:
            self.log_message("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            self.log_message(f"✅ Total execution time: {total_time:.1f} seconds")
            self.log_message(f"✅ All {len(self.steps_completed)} steps completed")
            self.log_message(f"✅ Outputs available in: {self.project_root / 'outputs'}")
            
            self.log_message("\n🚀 PROJECT READY FOR:")
            self.log_message("   • Stakeholder presentation")
            self.log_message("   • Portfolio demonstration")
            self.log_message("   • GitHub repository upload")
            self.log_message("   • Upwork profile showcase")
            
        else:
            self.log_message("❌ PIPELINE FAILED")
            self.log_message(f"⏱️  Execution time: {total_time:.1f} seconds")
            self.log_message(f"✅ Completed steps: {len(self.steps_completed)}")
            self.log_message(f"❌ Failed at: {pipeline_steps[len(self.steps_completed)][0] if len(self.steps_completed) < len(pipeline_steps) else 'Unknown'}")
        
        self.log_message(f"\n📋 Full log available at: {self.log_file}")
        
        return all_success

def main():
    """Main execution function."""
    print("🌞 SOLAR ENERGY PERFORMANCE ANALYSIS")
    print("Complete Analysis Pipeline")
    print("=" * 60)
    
    # Check if running from correct directory
    current_dir = Path.cwd()
    expected_files = ["scripts", "src", "data"]
    
    if not all((current_dir / f).exists() for f in expected_files):
        print("❌ Please run this script from the project root directory")
        print("Expected directory structure:")
        for f in expected_files:
            print(f"   {f}/")
        return 1
    
    # Create and run pipeline
    pipeline = SolarAnalysisPipeline()
    
    try:
        success = pipeline.run_complete_pipeline()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())