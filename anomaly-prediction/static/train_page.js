function trainGraph(){
    //console.log("trainGraph()");
    graphDivObj.innerHTML = "<img src='static/workingMsg.png'/>";
    let url = '/train-model';
    let formObj = document.getElementById("trainForm")
    let formData = new FormData(formObj);
    postTrainRequest(url, formData)
        .then(serializedImage => displayTrainGraphHTML(serializedImage))
        .catch(error => console.error(error))
}
async function postTrainRequest(url, formData) {
    return fetch(url, {
        method: 'POST',
        body: formData
    })
        .then((response) => response.text());
}

function displayTrainGraphHTML(serializedImage){
    graphDivObj.style.display = 'block';
    let graphHTML =  "<img src='data:image/png;base64, " + serializedImage +
        "' class='imgSize'/>" ;
    graphDivObj.innerHTML = graphHTML;
    testTabBtn.disabled = false;
    predTabBtn.disabled = false;
}


const graphDivObj = document.getElementById("trainGraphDiv");
const makeGraphBtnObj = document.getElementById("trainingBtn");
makeGraphBtnObj.addEventListener("click", trainGraph);