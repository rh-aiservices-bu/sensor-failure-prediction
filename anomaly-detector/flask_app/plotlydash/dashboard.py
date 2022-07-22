import dash
from dash import html
from data.data_prep import load_sensor
from dash import dcc
from graph_manager import GraphManager as GM

df, anomalies = load_sensor()
traces,layout = GM.prepare_data(df, anomalies)

def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets=[
            '/static/main.css',
        ]
    )

    # Create Dash Layout
        # Create Layout
    dash_app.layout = html.Div(
        children=[
            dcc.Graph(
                id="time-series-graph",
                figure= {"data": traces, "layout": layout}
            ),
        ],
        id="dash-container",
    )
    

    return dash_app.server

