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
    openTab(this, 'tab1'); // onloadDataCheck();
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



// When the Start Data Prep btn is clicked, Disable train tab, test tab, and predict tab
function startDataPrepControlButtons(){
    trainTabBtn.disabled = true;
    testTabBtn.disabled = true;
    predTabBtn.disabled = true;
}




