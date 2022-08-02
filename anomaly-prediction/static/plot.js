var graph = document.getElementById("predGraph");
let data = [{ // 0 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },
    { // 1 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },
    { // 2 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },
    { // 3 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },

    {   // Trace for alarm
           x:[],
           y:[],
           yaxis: 'y2',

           // plotcolor: ['green'],
           mode: 'lines',
           marker: {color: 'red'},
           line: {width: 2},
           opacity: 0.6
    }];
// This array of empty traces is used whenever we need to restart a plot after it has been stopped.
// Since the array is empty, the restarted plot will start with no data.
let initData = [{ // 0 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },
    { // 1 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },
    { // 2 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },
    { // 3 Trace for Data points for sensor
        x: [],
        y: [],
        mode: 'markers',
        marker: {color: 'black', size: 3},
        xaxis:{type: 'date'}
    },

    {   // 4 Trace for alarm
           x:[],
           y:[],
           yaxis: 'y2',

           // plotcolor: ['green'],
           mode: 'lines',
           marker: {color: 'red'},
           line: {width: 2},
           opacity: 0.6
    }];
    let layout = {
        margin: {t:100},
        xaxis: {type: 'date'},
        yaxis: {range: [0, 1],
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
    data[3].x = Array.from(initData[3].x);
    data[3].y = Array.from(initData[3].y);
    data[4].x = Array.from(initData[3].x);
    data[4].y = Array.from(initData[3].y);
    Plotly.newPlot('predGraph', data, layout);
}

var msgCounter = 0; // Another way of shifting. Not used in this code.

function updatePlot(jsonData){
    console.log("plot.js updatePlot()  " + jsonData);
    //console.log("msgCounter: " + msgCounter++);
    let max = 200;
    let jsonObj = JSON.parse(jsonData);  // json obj is in form:  ['timestamp', 'sensorVal']
    //plot_dict = {
    //        'timestamp': [row[0].timestamp],
   //         'sensor0': [row[0].sensor25],
   //         'sensor1': [row[0].sensor11],
    //        'sensor2': [row[0].sensor36],
    //        'sensor3': [row[0].sensor34],
    //        'alarm': [0]
     //   }

    let timestamp = jsonObj.timestamp;
    let sensor25 = jsonObj.sensor0;
    let sensor11 = jsonObj.sensor1;
    let sensor36 = jsonObj.sensor2;
    let sensor34 = jsonObj.sensor3;
    let alarm =    jsonObj.alarm;



    // Add new data point as well as calculated data for regression line start and end points as well as the
    // y difference plot.  Note there are three traces.  The first two traces use the same layout named yaxis.
    // The third trace uses the layout named yaxis2.  This naming convention follows that of plotly.js
    Plotly.extendTraces('predGraph', {
        // NOTE:  The below 2 commented lines would be used if the dynamic line coloring were supported by plotly.js
        //x: [[timestamp], [x1, x2], [diff_x1, diff_x2]],
        //y: [[sensorValue], [y1,y2], [diff_y1, diff_y2]]
       // x: [[timestamp], [x1, x2], [diff_x2]],
       // y: [[sensorValue], [y1,y2], [diff_y2]]
        x: [[timestamp], [timestamp], [timestamp], [timestamp] ],
        y: [[sensor25], [sensor11], [sensor36], [sensor34]]

    }, [0, 1, 2,3], max);  // The array denotes to plot all three traces(0 based).  Keep only last max data points


    // NOTE:  This code below was an attempt to dynamically change the color of the Percent Diff plot to red whenever
    // the plot went beyond the acceptable range.  Plotly.js does not support such a feature.

    //if(plot_color != null){
    //    let updateStr = 'marker.color[' + row_counter + ']';
        //Plotly.restyle('graph', editObj, [2]);
    //    Plotly.react(graph, {[updateStr]: plot_color}, null, [2]);

    //}

}