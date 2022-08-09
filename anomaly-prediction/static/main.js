let displayDataDiv = document.getElementById("displayDataDiv");
let dataPrepStartBtn = document.getElementById('startBtn');
window.onload = async function(){
    console.log("window.onload");
    let url = '/initData'
    postInitData(url)
        .then(jsonData =>  displayJson(jsonData))
        .catch(error => console.error(error))
    };

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

}
