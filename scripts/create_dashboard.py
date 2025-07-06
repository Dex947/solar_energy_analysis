"""
Solar Dashboard Creator
Creates interactive Excel dashboard with KPIs and visualizations.

Author: Bryant M.
Date: July 2025
Project: Solar Energy Performance Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xlsxwriter
from datetime import datetime
import json

class SolarDashboardCreator:
    """Creates comprehensive Excel dashboard for solar performance analysis."""
    
    def __init__(self):
        """Initialize the dashboard creator."""
        self.data_dir = Path("data/processed")
        self.output_dir = Path("outputs/dashboards")
        self.reports_dir = Path("outputs/reports")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.combined_data = None
        self.performance_metrics = None
        
        # Dashboard filename
        self.dashboard_file = self.output_dir / f"solar_performance_dashboard_{datetime.now().strftime('%Y%m%d')}.xlsx"
    
    def load_data(self):
        """Load processed data and performance metrics."""
        print("📥 Loading data for dashboard creation...")
        
        try:
            # Load combined data
            self.combined_data = pd.read_csv(self.data_dir / "combined_solar_data.csv")
            self.combined_data['DATE_TIME'] = pd.to_datetime(self.combined_data['DATE_TIME'])
            
            # Load performance metrics
            with open(self.reports_dir / "performance_metrics.json", 'r') as f:
                self.performance_metrics = json.load(f)
            
            print(f"✅ Loaded {len(self.combined_data):,} records")
            print("✅ Loaded performance metrics")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def prepare_dashboard_data(self):
        """Prepare summary data for dashboard."""
        print("🔧 Preparing dashboard data...")
        
        # Summary statistics
        self.summary_stats = {}
        
        for plant in ['Plant_1', 'Plant_2']:
            plant_data = self.combined_data[self.combined_data['PLANT'] == plant]
            
            if len(plant_data) == 0:
                continue
            
            self.summary_stats[plant] = {
                'total_records': len(plant_data),
                'avg_power': plant_data['AC_POWER'].mean(),
                'max_power': plant_data['AC_POWER'].max(),
                'min_power': plant_data['AC_POWER'].min(),
                'total_yield': plant_data['TOTAL_YIELD'].max(),
                'avg_efficiency': plant_data['EFFICIENCY'].mean(),
                'avg_temp': plant_data['AMBIENT_TEMPERATURE'].mean(),
                'avg_irradiation': plant_data['IRRADIATION'].mean(),
                'uptime_pct': (plant_data['AC_POWER'] > 0).mean() * 100
            }
        
        # Hourly performance
        self.hourly_performance = self.combined_data.groupby(['HOUR', 'PLANT']).agg({
            'AC_POWER': 'mean',
            'EFFICIENCY': 'mean',
            'AMBIENT_TEMPERATURE': 'mean',
            'IRRADIATION': 'mean'
        }).reset_index()
        
        # Daily performance
        self.daily_performance = self.combined_data.groupby([
            self.combined_data['DATE_TIME'].dt.date, 'PLANT'
        ]).agg({
            'AC_POWER': ['sum', 'mean', 'max'],
            'DAILY_YIELD': 'max',
            'EFFICIENCY': 'mean'
        }).reset_index()
        
        # Flatten column names
        self.daily_performance.columns = ['DATE', 'PLANT', 'TOTAL_POWER', 'AVG_POWER', 
                                        'MAX_POWER', 'DAILY_YIELD', 'AVG_EFFICIENCY']
        
        # Weather correlations
        self.weather_correlations = {}
        for plant in ['Plant_1', 'Plant_2']:
            plant_data = self.combined_data[self.combined_data['PLANT'] == plant]
            if len(plant_data) > 0:
                corr_matrix = plant_data[['AC_POWER', 'AMBIENT_TEMPERATURE', 
                                        'MODULE_TEMPERATURE', 'IRRADIATION']].corr()
                self.weather_correlations[plant] = corr_matrix.to_dict()
        
        print("✅ Dashboard data prepared")
    
    def create_excel_dashboard(self):
        """Create the main Excel dashboard."""
        print("📊 Creating Excel dashboard...")
        
        # Create workbook
        workbook = xlsxwriter.Workbook(self.dashboard_file)
        
        # Define formats
        formats = self._create_formats(workbook)
        
        # Create worksheets
        self._create_summary_sheet(workbook, formats)
        self._create_performance_sheet(workbook, formats)
        self._create_comparison_sheet(workbook, formats)
        self._create_trends_sheet(workbook, formats)
        self._create_raw_data_sheet(workbook, formats)
        
        # Close workbook
        workbook.close()
        
        print(f"✅ Excel dashboard created: {self.dashboard_file}")
        return self.dashboard_file
    
    def _create_formats(self, workbook):
        """Create formatting styles for the workbook."""
        return {
            'title': workbook.add_format({
                'bold': True,
                'font_size': 16,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#1f497d',
                'font_color': 'white'
            }),
            'header': workbook.add_format({
                'bold': True,
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#4f81bd',
                'font_color': 'white',
                'border': 1
            }),
            'kpi_value': workbook.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'center',
                'num_format': '#,##0.0',
                'bg_color': '#dce6f1',
                'border': 1
            }),
            'kpi_label': workbook.add_format({
                'bold': True,
                'align': 'center',
                'bg_color': '#f2f2f2',
                'border': 1
            }),
            'data': workbook.add_format({
                'align': 'center',
                'num_format': '#,##0.0',
                'border': 1
            }),
            'percent': workbook.add_format({
                'align': 'center',
                'num_format': '0.0%',
                'border': 1
            }),
            'date': workbook.add_format({
                'align': 'center',
                'num_format': 'yyyy-mm-dd',
                'border': 1
            })
        }
    
    def _create_summary_sheet(self, workbook, formats):
        """Create executive summary dashboard sheet."""
        worksheet = workbook.add_worksheet('Executive Summary')
        
        # Set column widths
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:E', 15)
        
        # Title
        worksheet.merge_range('A1:E1', 'SOLAR ENERGY PERFORMANCE DASHBOARD', formats['title'])
        worksheet.merge_range('A2:E2', f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', formats['header'])
        
        # Key Performance Indicators
        row = 4
        worksheet.write(row, 0, 'KEY PERFORMANCE INDICATORS', formats['header'])
        row += 2
        
        # Plant comparison KPIs
        kpi_headers = ['Metric', 'Plant 1', 'Plant 2', 'Difference', 'Better']
        for col, header in enumerate(kpi_headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1
        
        # KPI data
        if 'Plant_1' in self.summary_stats and 'Plant_2' in self.summary_stats:
            p1 = self.summary_stats['Plant_1']
            p2 = self.summary_stats['Plant_2']
            
            kpis = [
                ('Average Power (kW)', p1['avg_power'], p2['avg_power']),
                ('Maximum Power (kW)', p1['max_power'], p2['max_power']),
                ('Total Yield (kWh)', p1['total_yield'], p2['total_yield']),
                ('Average Efficiency (%)', p1['avg_efficiency'], p2['avg_efficiency']),
                ('Uptime (%)', p1['uptime_pct'], p2['uptime_pct']),
                ('Avg Temperature (°C)', p1['avg_temp'], p2['avg_temp']),
                ('Avg Irradiation (W/m²)', p1['avg_irradiation'], p2['avg_irradiation'])
            ]
            
            for metric, val1, val2 in kpis:
                diff = val2 - val1
                better = 'Plant 2' if val2 > val1 else 'Plant 1'
                
                worksheet.write(row, 0, metric, formats['kpi_label'])
                worksheet.write(row, 1, val1, formats['kpi_value'])
                worksheet.write(row, 2, val2, formats['kpi_value'])
                worksheet.write(row, 3, diff, formats['kpi_value'])
                worksheet.write(row, 4, better, formats['data'])
                row += 1
        
        # Performance insights
        row += 2
        worksheet.write(row, 0, 'PERFORMANCE INSIGHTS', formats['header'])
        row += 2
        
        # Add insights if available
        try:
            with open(self.reports_dir / "insights_and_recommendations.txt", 'r') as f:
                content = f.read()
                insights_section = content.split('KEY INSIGHTS:')[1].split('RECOMMENDATIONS:')[0]
                insights = [line.strip() for line in insights_section.split('\n') if line.strip() and not line.startswith('-')]
                
                for i, insight in enumerate(insights[:5], 1):
                    worksheet.write(row, 0, f"{i}. {insight}", formats['data'])
                    row += 1
        except:
            worksheet.write(row, 0, "Insights available in separate report", formats['data'])
        
        return worksheet
    
    def _create_performance_sheet(self, workbook, formats):
        """Create detailed performance analysis sheet."""
        worksheet = workbook.add_worksheet('Performance Analysis')
        
        # Set column widths
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:Z', 12)
        
        # Title
        worksheet.merge_range('A1:H1', 'DETAILED PERFORMANCE ANALYSIS', formats['title'])
        
        # Hourly performance data
        row = 3
        worksheet.write(row, 0, 'HOURLY PERFORMANCE AVERAGES', formats['header'])
        row += 2
        
        # Headers
        hourly_headers = ['Hour', 'Plant 1 Power (kW)', 'Plant 2 Power (kW)', 
                         'Plant 1 Efficiency (%)', 'Plant 2 Efficiency (%)']
        for col, header in enumerate(hourly_headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1
        
        # Hourly data
        for hour in range(24):
            p1_data = self.hourly_performance[
                (self.hourly_performance['HOUR'] == hour) & 
                (self.hourly_performance['PLANT'] == 'Plant_1')
            ]
            p2_data = self.hourly_performance[
                (self.hourly_performance['HOUR'] == hour) & 
                (self.hourly_performance['PLANT'] == 'Plant_2')
            ]
            
            p1_power = p1_data['AC_POWER'].iloc[0] if len(p1_data) > 0 else 0
            p2_power = p2_data['AC_POWER'].iloc[0] if len(p2_data) > 0 else 0
            p1_eff = p1_data['EFFICIENCY'].iloc[0] if len(p1_data) > 0 else 0
            p2_eff = p2_data['EFFICIENCY'].iloc[0] if len(p2_data) > 0 else 0
            
            worksheet.write(row, 0, hour, formats['data'])
            worksheet.write(row, 1, p1_power, formats['data'])
            worksheet.write(row, 2, p2_power, formats['data'])
            worksheet.write(row, 3, p1_eff, formats['data'])
            worksheet.write(row, 4, p2_eff, formats['data'])
            row += 1
        
        # Weather correlations
        row += 2
        worksheet.write(row, 0, 'WEATHER CORRELATIONS', formats['header'])
        row += 2
        
        if self.weather_correlations:
            corr_headers = ['Plant', 'Temp Correlation', 'Module Temp Correlation', 'Irradiation Correlation']
            for col, header in enumerate(corr_headers):
                worksheet.write(row, col, header, formats['header'])
            row += 1
            
            for plant in ['Plant_1', 'Plant_2']:
                if plant in self.weather_correlations:
                    corr_data = self.weather_correlations[plant]['AC_POWER']
                    worksheet.write(row, 0, plant, formats['data'])
                    worksheet.write(row, 1, corr_data.get('AMBIENT_TEMPERATURE', 0), formats['data'])
                    worksheet.write(row, 2, corr_data.get('MODULE_TEMPERATURE', 0), formats['data'])
                    worksheet.write(row, 3, corr_data.get('IRRADIATION', 0), formats['data'])
                    row += 1
        
        return worksheet
    
    def _create_comparison_sheet(self, workbook, formats):
        """Create plant comparison sheet."""
        worksheet = workbook.add_worksheet('Plant Comparison')
        
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:E', 15)
        
        # Title
        worksheet.merge_range('A1:E1', 'PLANT-TO-PLANT COMPARISON', formats['title'])
        
        # Statistical comparison
        row = 3
        worksheet.write(row, 0, 'STATISTICAL COMPARISON', formats['header'])
        row += 2
        
        if 'Plant_1' in self.summary_stats and 'Plant_2' in self.summary_stats:
            p1 = self.summary_stats['Plant_1']
            p2 = self.summary_stats['Plant_2']
            
            # Comparison table
            comp_headers = ['Metric', 'Plant 1', 'Plant 2', 'Difference (%)', 'Winner']
            for col, header in enumerate(comp_headers):
                worksheet.write(row, col, header, formats['header'])
            row += 1
            
            comparisons = [
                ('Records Count', p1['total_records'], p2['total_records']),
                ('Average Power', p1['avg_power'], p2['avg_power']),
                ('Maximum Power', p1['max_power'], p2['max_power']),
                ('Total Yield', p1['total_yield'], p2['total_yield']),
                ('Average Efficiency', p1['avg_efficiency'], p2['avg_efficiency']),
                ('Uptime Percentage', p1['uptime_pct'], p2['uptime_pct'])
            ]
            
            for metric, val1, val2 in comparisons:
                diff_pct = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                winner = 'Plant 2' if val2 > val1 else 'Plant 1'
                
                worksheet.write(row, 0, metric, formats['kpi_label'])
                worksheet.write(row, 1, val1, formats['data'])
                worksheet.write(row, 2, val2, formats['data'])
                worksheet.write(row, 3, diff_pct, formats['data'])
                worksheet.write(row, 4, winner, formats['data'])
                row += 1
        
        return worksheet
    
    def _create_trends_sheet(self, workbook, formats):
        """Create trends analysis sheet."""
        worksheet = workbook.add_worksheet('Trends Analysis')
        
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:H', 12)
        
        # Title
        worksheet.merge_range('A1:H1', 'DAILY TRENDS ANALYSIS', formats['title'])
        
        # Daily performance trends
        row = 3
        worksheet.write(row, 0, 'DAILY PERFORMANCE DATA', formats['header'])
        row += 2
        
        # Headers
        trend_headers = ['Date', 'Plant', 'Total Power (kW)', 'Avg Power (kW)', 
                        'Max Power (kW)', 'Daily Yield (kWh)', 'Avg Efficiency (%)']
        for col, header in enumerate(trend_headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1
        
        # Daily data (last 30 days for readability)
        recent_data = self.daily_performance.tail(60)  # Last 60 rows (30 days * 2 plants)
        
        for _, daily_row in recent_data.iterrows():
            worksheet.write(row, 0, daily_row['DATE'], formats['date'])
            worksheet.write(row, 1, daily_row['PLANT'], formats['data'])
            worksheet.write(row, 2, daily_row['TOTAL_POWER'], formats['data'])
            worksheet.write(row, 3, daily_row['AVG_POWER'], formats['data'])
            worksheet.write(row, 4, daily_row['MAX_POWER'], formats['data'])
            worksheet.write(row, 5, daily_row['DAILY_YIELD'], formats['data'])
            worksheet.write(row, 6, daily_row['AVG_EFFICIENCY'], formats['data'])
            row += 1
        
        return worksheet
    
    def _create_raw_data_sheet(self, workbook, formats):
        """Create raw data sheet with sample data."""
        worksheet = workbook.add_worksheet('Sample Raw Data')
        
        # Set column widths
        worksheet.set_column('A:A', 18)
        worksheet.set_column('B:Z', 12)
        
        # Title
        worksheet.merge_range('A1:L1', 'SAMPLE RAW DATA (Last 1000 Records)', formats['title'])
        
        # Get sample data
        sample_data = self.combined_data.tail(1000)
        
        # Headers
        row = 3
        headers = ['DateTime', 'Plant', 'DC Power', 'AC Power', 'Daily Yield', 
                  'Total Yield', 'Efficiency', 'Ambient Temp', 'Module Temp', 'Irradiation']
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1
        
        # Data
        for _, data_row in sample_data.iterrows():
            worksheet.write(row, 0, data_row['DATE_TIME'].strftime('%Y-%m-%d %H:%M'), formats['data'])
            worksheet.write(row, 1, data_row['PLANT'], formats['data'])
            worksheet.write(row, 2, data_row.get('DC_POWER', 0), formats['data'])
            worksheet.write(row, 3, data_row.get('AC_POWER', 0), formats['data'])
            worksheet.write(row, 4, data_row.get('DAILY_YIELD', 0), formats['data'])
            worksheet.write(row, 5, data_row.get('TOTAL_YIELD', 0), formats['data'])
            worksheet.write(row, 6, data_row.get('EFFICIENCY', 0), formats['data'])
            worksheet.write(row, 7, data_row.get('AMBIENT_TEMPERATURE', 0), formats['data'])
            worksheet.write(row, 8, data_row.get('MODULE_TEMPERATURE', 0), formats['data'])
            worksheet.write(row, 9, data_row.get('IRRADIATION', 0), formats['data'])
            row += 1
        
        return worksheet
    
    def create_dashboard(self):
        """Main method to create the complete dashboard."""
        print("📊 CREATING SOLAR PERFORMANCE DASHBOARD")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data")
            return False
        
        # Prepare dashboard data
        self.prepare_dashboard_data()
        
        # Create Excel dashboard
        dashboard_file = self.create_excel_dashboard()
        
        print(f"\n🎉 DASHBOARD CREATION COMPLETE!")
        print(f"✅ Excel dashboard: {dashboard_file}")
        print(f"✅ File size: {dashboard_file.stat().st_size / 1024:.1f} KB")
        print("\n📋 Dashboard contains:")
        print("   • Executive Summary with KPIs")
        print("   • Detailed Performance Analysis")
        print("   • Plant-to-Plant Comparison")
        print("   • Daily Trends Analysis")
        print("   • Sample Raw Data")
        
        return dashboard_file

def main():
    """Main execution function."""
    processed_data = Path("data/processed/combined_solar_data.csv")
    metrics_file = Path("outputs/reports/performance_metrics.json")
    
    if not processed_data.exists():
        print(f"❌ Processed data not found: {processed_data}")
        print("Please run the main analysis script first.")
        return 1
    
    if not metrics_file.exists():
        print(f"❌ Performance metrics not found: {metrics_file}")
        print("Please run the main analysis script first.")
        return 1
    
    creator = SolarDashboardCreator()
    
    if creator.create_dashboard():
        print("\n🚀 Dashboard ready for stakeholder presentation!")
        return 0
    else:
        print("❌ Dashboard creation failed.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())