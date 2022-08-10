let displayDataDiv = document.getElementById("displayDataDiv");
let dataPrepStartBtn = document.getElementById('startBtn');
window.onload = async function(){
    console.log("window.onload");
    let url = '/initData'
    postInitData(url)
        .then(jsonData =>  displayJson(jsonData))
        .catch(error => console.error(error))
    }; onloadDataCheck();

async function postInitData(url){
    return fetch(url,{
        method: 'POST',
        contentType: "application/json"
    })
        .then((response) => response.json());

}
function displayJson(jsonData){
   // Parse json, put values into html page elements
    displayDataDiv.innerHTML = jsonData.data;
    dataPrepStartBtn.disabled = false;
    //onloadDataCheck();

}
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
// If both exist, enable Train Tab, Test Tab and Prediction Tab
function onloadDataCheck(){
    let fileName1 = 'static/trained_model/saved_model.pb';
    let fileName2 = 'static/training_scaler.gz'
    let dataPresent = doTwoFilesExist(fileName1, fileName2); //This is a Promise that was returned from the
                                                             //async function, doTwoFilesExist()

    dataPresent.then(function(result){  // The variable, dataPresent is a Promise,
    if(result){
        predTabBtn.disabled = false;
        testTabBtn.disabled = false;
        trainTabBtn.disabled = false;
    }else{
        predTabBtn.disabled = true;  // This could be redundant since when page loads, the btn is disabled.
        testTabBtn.disabled = true;
        trainTabBtn.disabled = true;
    }
    });

}
