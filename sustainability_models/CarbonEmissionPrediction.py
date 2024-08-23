import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

class CarbonEmissionPrediction:
    def __init__(self, input_df, input_column, arima_order, year, average_yearly_emission):
        self.input_column = input_column
        self.arima_order = arima_order
        self.year = year
        self.average_yearly_emission = average_yearly_emission
        self.df = input_df
        self.model = ARIMA(self.df[self.input_column], order=self.arima_order)
        self.model_fit = self.model.fit()
        self.year_range_predict = [x for x in range(self.df['year'][len(self.df)-1] + 1, self.year + 1)]
        
    def predict(self):
        output_dict = {'year': self.year_range_predict, self.input_column:[]}
        output = self.model_fit.forecast(len(self.year_range_predict))
        output_dict[self.input_column] = output.values.tolist()
        output_df = pd.DataFrame(output_dict)
        final_df = pd.concat([self.df, output_df]).reset_index(drop=True)
        final_df['carbon_emission'] = final_df[self.input_column] * self.average_yearly_emission
        return final_df
