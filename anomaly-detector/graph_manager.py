import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
from plotly.graph_objs import Layout, Margin
from plotly.subplots import make_subplots

from io import BytesIO
import base64
import pandas as pd
class GraphManager:
    def __init__(self):
        pass


    
    def plot_data(df_data, anomaly_times):
        """
        fig_height = 4
        fig_width = 10
        plt.figure(figsize=(fig_width, fig_height))
        plt.title('Predictions Showing Alarm Points in Red')
        df_data.plot(c="black", figsize=(fig_width, fig_height))
        plt.show()"""
        cols = list(df_data.columns.values)
        ncols = len(cols)

        # subplot setup

        traces = []
        for col in cols:
            traces.append(go.Scatter(x=df_data[col].index,
                                     y=df_data[col].values,
                                     mode='lines',
                                     line_color='#dde3ed'))

        for time in anomaly_times:
            traces.append(go.Scatter(x=[time, time],
                                     y=[0, 0.5],
                                     mode='lines',
                                     line_color='#e04031',
                                     marker_line_width=5
                                     ))
        layout = Layout(width=750,
                        height=250,
                        showlegend=False,
                        title={'text': 'Anomaly Detection',
                               'y': 0.99,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'
                               },
                        yaxis={'visible': False},
                        margin=Margin(l=5, r=5, b=20,t=40, pad=0))
        div_buffer = plotly.offline.plot({"data": traces, "layout": layout},
                                         include_plotlyjs='cdn',
                                         output_type='div',
                                         config=dict(displayModeBar=False))

        return div_buffer






