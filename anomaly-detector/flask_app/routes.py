"""Routes for parent Flask app."""
from flask import render_template
from flask import current_app as app


@app.route('/')
def main():
    return render_template('main.html', col_list=['sensor_25', 'sensor_11', 'sensor_36', 'sensor_34'])

@app.route('/generate-graph', methods=['GET', 'POST'])
def generate_graph():
    chosen_sensor = request.form or '{}'
    print(chosen_sensor)
    #json.loads(chosen_sensor).get('sensor')
    df, anomalies = load_sensor('sensor_25')
    buffer = GM.plot_data(df, anomalies)
    return buffer
