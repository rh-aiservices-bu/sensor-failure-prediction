var graph = document.getElementById("predGraph");
let data = [{ // 0 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'gray', size: 3},
        xaxis:{type: 'date'}
    },
    { // 1 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'gray', size: 3},
        xaxis:{type: 'date'}
    },
    { // 2 Trace for alarm
        x: [],
        y: [],
        yaxis: 'y2',
        mode: 'lines',
        marker: {color: 'red', size: 1},
        xaxis:{type: 'date'}
    }];


// This array of empty traces is used whenever we need to restart a plot after it has been stopped.
// Since the array is empty, the restarted plot will start with no data.
let initData = [{ // 0 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'gray', size: 3},
        xaxis:{type: 'date'}
    },
    { // 1 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'gray', size: 3},
        xaxis:{type: 'date'}
    },
    { // 2 Trace for alarm
        x: [],
        y: [],
        yaxis: 'y2',
        mode: 'lines',
        marker: {color: 'red', size: 1},
        xaxis:{type: 'date'}
    }];
    let layout = {
        title: {text: 'Failure Prediction',
                font: {size: 20},
                xanchor: 'center',
                yanchor: 'top'},
        margin: {t:50},
        xaxis: {type: 'date'},
        yaxis: {range: [-1, 1],
                title: 'Scaled Sensors',
                side: 'left'},
        yaxis2: {title: 'Failure Prediction',
                 titlefont: {color: 'red'},
                 overlaying: 'y',
                 side: 'right',
                 range: [0, 1],
                 zerolinecolor: 'red'
        },
        showlegend: false

    };

// First do a deep clone of the data array of traces.  The clone uses values from the empty array, initData
// Then call Plotly.newPlot() using the cloned array of empty traces to start a new plot.
function initPlot(){
    data[0].x = Array.from(initData[0].x);
    data[0].y = Array.from(initData[0].y);
    data[1].x = Array.from(initData[1].x);
    data[1].y = Array.from(initData[1].y);
    data[2].x = Array.from(initData[2].x);
    data[2].y = Array.from(initData[2].y);

    Plotly.newPlot('predGraph', data, layout);
}

var msgCounter = 0; // Another way of shifting. Not used in this code.

function updatePlot(jsonData){
   // console.log("plot.js updatePlot()  " + jsonData);
    //console.log("msgCounter: " + msgCounter++);
    let max = 200;
    let jsonObj = JSON.parse(jsonData);  // json obj is in form:  ['timestamp', 'sensorVal']
 //           plot_dict = {
//            'timestamp': one_row_df.index[0],
//            'pc1': one_row_df['pc1'].values[0],
//            'pc2': one_row_df['pc2'].values[0],
//            'alarm': alarm_value
//        }

    let timestamp = jsonObj.timestamp;
    let pc1 = jsonObj.pc1;
    let pc2 = jsonObj.pc2;
    let alarm =    jsonObj.alarm;



    // Note there are three traces.  The first two traces use the same layout named yaxis.
    // The first two traces plot the values of the two chosen pc's
    // The third trace uses the layout named yaxis2.  This naming convention follows that of plotly.js
    // The third trace plots vertical red lines showing the predictions
    Plotly.extendTraces('predGraph', {

        x: [[timestamp], [timestamp], [timestamp, timestamp] ],
        y: [[pc1], [pc2], [0, alarm]]

    }, [0, 1, 2], max);  // The array denotes to plot all three traces(0 based).  Keep only last max data points


}