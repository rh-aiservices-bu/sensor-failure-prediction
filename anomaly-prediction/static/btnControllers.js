let dataTabBtn = document.getElementById("dataPrepBtn");
let trainBtn = document.getElementById("trainBtn");
let testBtn = document.getElementById("testBtn");
let predBtn = document.getElementById("predictBtn");


// Event Listeners for tab buttons
dataTabBtn.addEventListener("click", function(){
    openTab(this, 'tab1')
});
trainBtn.addEventListener("click", function(){
    openTab(this, 'tab2')
});

testBtn.addEventListener("click", function(){
    openTab(this, 'tab3')
});
predBtn.addEventListener("click", function(){
    openTab(this, 'tab4')
});


