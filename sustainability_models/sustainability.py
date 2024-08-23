from bigq_extract import query_db
from PublicTransUtilizationPrediction import PublicTransUtilizationPrediction
from CarbonEmissionPrediction import CarbonEmissionPrediction
import matplotlib.pyplot as plt
from datetime import datetime

class sustainability:
    def __init__(self):
        self.query_data = query_db()
        self.query1 = "SELECT * FROM ``"
        self.query2 = "SELECT * FROM ``"
        self.df1 = self.query_data.run_query(self.query1)
        self.df2 = self.query_data.run_query(self.query2)
        self.df1_f = None
        self.df2_f = None

    def plot_dataframes(self, final_year, show_legend, point_style='line'):
        current_year = datetime.now().year

        public_trans = PublicTransUtilizationPrediction(input_df=self.df1[self.df1['year'] < current_year].sort_values(by="year").reset_index(drop=True), input_column='util_peak_period', arima_order=(5,1,1), year=final_year)
        self.df2_f = public_trans.predict().sort_values(by="year")
        
        carbon_emission = CarbonEmissionPrediction(
            input_df=self.df2[self.df2['year'] < current_year].sort_values(by="year").reset_index(drop=True), input_column='car_population', arima_order=(5,1,1), year=final_year, average_yearly_emission=10500
        )
        self.df1_f = carbon_emission.predict().sort_values(by="year")

        dataframes = [self.df1_f, self.df2_f[self.df2_f['year']>=2013]]
                
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        titles = ["Carbon Emissions & Car Volume", "Peak Period Public Transport Utilitzation Rate"]
        plt_format = ({"cross": "X", "line": "-", "circle": "o--"})[point_style]
        for i, (df, title) in enumerate(zip((dataframes), titles)):
            ax = axs[i]
            for col in df.columns[1:]:
                mask = df['year'] <= current_year - 1
                mask_forecast = df['year'] >= current_year - 1
                x_point = str(current_year-1)
                if col == "carbon_emission":
                    ax.plot(df[mask]['year'].astype(str), df[mask][col]/1000000, plt_format, label="Carbon Emissions", color='blue')
                    ax.plot(df[mask_forecast]['year'].astype(str), df[mask_forecast][col]/1000000, plt_format, label="Carbon Emissions (Forecasted)", 
                            color='blue', linestyle='dotted')
                    ax.plot(df['year'].astype(str), [6000]*len(df), plt_format, label="Carbon Emissions Target", color='red', linestyle='-')
                    
                    y_point = df[df['year'] == current_year-1]['carbon_emission'].values[0]/1000000
                    ax.annotate(f'({x_point}, {y_point:.2f})', xy=(x_point, y_point), xytext=(10, 0), textcoords='offset points', color='blue')
                    ax.plot(x_point, y_point, marker='o', markersize=5, color='blue')
                    
                elif col == "car_population":
                    ax.plot(df[mask]['year'].astype(str), df[mask][col]/100, plt_format, label="Car Volume", color='orange')
                    ax.plot(df[mask_forecast]['year'].astype(str), df[mask_forecast][col]/100, plt_format, label="Car Volume (Forecasted)", 
                            color='orange', linestyle='dotted')
        
                    y_point = df[df['year'] == current_year-1]['car_population'].values[0]/100
                    ax.annotate(f'({x_point}, {y_point:.2f})', xy=(x_point, y_point), xytext=(-20, 5), textcoords='offset points', color='orange')
                    ax.plot(x_point, y_point, marker='o', markersize=5, color='orange')
                else:
                    ax.plot(df[mask]['year'].astype(str), df[mask][col], plt_format, label="Utilization Rate", color='purple')
                    ax.plot(df[mask_forecast]['year'].astype(str), df[mask_forecast][col], plt_format, label="Utilization Rate (Forecasted)", 
                            color='purple', linestyle='dotted')
                    ax.plot(df['year'].astype(str), [0.75]*len(df), plt_format, label="Utilization Rate Target", color='red', linestyle='-')
        
                    y_point = df[df['year'] == current_year-1]['util_peak_period'].values[0]
                    ax.annotate(f'({x_point}, {y_point:.2f})', xy=(x_point, y_point), xytext=(10, 0), textcoords='offset points', color='purple')
                    ax.plot(x_point, y_point, marker='o', markersize=5, color='purple')
                        
            ax.set_xlabel('Year')
            if title == "Peak Period Public Transport Utilitzation Rate":
                ax.set_ylabel('Utilization Rate (%)')
            else:
                ax.set_ylabel('Carbon Emissions (kg CO2) in Millions')
                secax = ax.secondary_yaxis('right')
                secax.set_ylabel('Car Volume (In Hundreds)')
            ax.set_title(title)
            if show_legend: 
                ax.legend()
            ax.tick_params(axis='x', rotation=45)
            ax.set_xticks(ax.get_xticks()[::1])
            
        plt.tight_layout()
        return fig

