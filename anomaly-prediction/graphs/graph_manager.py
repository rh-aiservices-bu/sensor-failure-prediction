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

    @staticmethod
    def plot_history(history):
        fig_height = 4
        fig_width = 10
        plt.figure(figsize=(fig_width, fig_height))
        plt.title('Cross-Entropy Loss and Binary-Accuracy')
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['binary_accuracy'], label='Accuracy')

        plt.legend()
        plt.xlabel("Epoch")

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        buff_contents = buffer.getvalue()
        buff_encoded = base64.b64encode(buff_contents)
        buff_str = buff_encoded.decode('ascii')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        plt.close()
        return encoded

    @staticmethod
    def plot_test(df_test_window, alarm_times):
        """
        fig_height = 4
        fig_width = 10
        plt.figure(figsize=(fig_width, fig_height))
        plt.title('Predictions Showing Alarm Points in Red')
        df_test_window.plot(c="black", figsize=(fig_width, fig_height))
        plt.show()"""
        cols = list(df_test_window.columns.values)
        ncols = len(cols)

        # subplot setup

        traces = []
        for col in cols:
            traces.append(go.Scatter(x=df_test_window[col].index,
                                     y=df_test_window[col].values,
                                     mode='lines',
                                     line_color='#dde3ed'))

        for alarm_time in alarm_times:
            traces.append(go.Scatter(x=[alarm_time, alarm_time],
                                     y=[0, 0.5],
                                     mode='lines',
                                     line_color='#e04031',
                                     marker_line_width=5
                                     ))
        layout = Layout(width=750,
                        height=250,
                        showlegend=False,
                        title={'text': 'Failure Prediction',
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





    @staticmethod
    def df_plot_failure(df_prediction, feature_names):
        df_prediction[feature_names].plot(c='black', figsize=(10,2))
        plt.show()
