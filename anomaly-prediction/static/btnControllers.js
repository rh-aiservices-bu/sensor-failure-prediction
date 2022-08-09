// Tab Buttons
let dataTabBtn = document.getElementById("dataPrepTabBtn");
let trainTabBtn = document.getElementById("trainTabBtn");
let testTabBtn = document.getElementById("testTabBtn");
let predTabBtn = document.getElementById("predictTabBtn");

// Content Buttons
let startDataPrepBtn = document.getElementById("startBtn");
let trainBtn = document.getElementById('trainingBtn');
let testBtn = document.getElementById('testingBtn');
let predictStartBtn = document.getElementById('startPredictBtn');
let predictStopBtn = document.getElementById('stopPredictBtn');



// Event Listeners for tab buttons
dataTabBtn.addEventListener("click", function(){
    openTab(this, 'tab1'); onloadDataCheck();
});
trainTabBtn.addEventListener("click", function(){
    openTab(this, 'tab2')
});
testTabBtn.addEventListener("click", function(){
    openTab(this, 'tab3')
});
predTabBtn.addEventListener("click", function(){
    openTab(this, 'tab4')
});
startDataPrepBtn.addEventListener('click', function(){
    startDataPrepControlButtons(this);
})

// Determine if both files exist on server.  Return true if both exist, false otherwise
// Return as a Promise a boolean that is true if both files exist, false if otherwise
async function doTwoFilesExist(fileName1, fileName2){
    // Do a fetch for each filename
    let response1 = await fetch(fileName1, {method: 'HEAD'});
    let response2 = await fetch(fileName2, {method: 'HEAD'});
    let retValBool = response1.status == 200 && response2.status == 200;
    return retValBool;


}
// When Data Prep tab is clicked, first determined if model has already been trained and scaler exists.
// If both exist, enable Prediction Tab
function onloadDataCheck(){
    let fileName1 = 'static/trained_model/saved_model.pb';
    let fileName2 = 'static/training_scaler.gz'
    let dataPresent = doTwoFilesExist(fileName1, fileName2); //This is a Promise that was returned from the
                                                             //async function, doTwoFilesExist()

    dataPresent.then(function(result){  // The variable, dataPresent is a Promise,
    if(result){
        predTabBtn.disabled = false;
    }else{
        predTabBtn.disabled = true;  // This could be redundant since when page loads, the btn is disabled.
    }
    });

}
// When the Start Data Prep btn is clicked, Disable train tab, test tab, and predict tab
function startDataPrepControlButtons(){
    trainTabBtn.disabled = true;
    testTabBtn.disabled = true;
    predTabBtn.disabled = true;
}




