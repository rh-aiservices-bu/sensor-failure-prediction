function displayGraph(){
    let url = '/generateData';
    let formObj = document.getElementById("postgresForm")
    let formData = new FormData(formObj);
    postTestRequest(url, formData)
        .then(graphHTML => displayGraphHTML(graphHTML))
        .catch(error => console.error(error))
}

function displayGraphFromCsv(){
    let url = '/generateDataFromCsv';
    let formObj = document.getElementById("csvForm")
    let formData = new FormData(formObj);
    postTestRequest(url, formData)
        .then(graphHTML => displayGraphHTML(graphHTML))
        .catch(error => console.error(error))
}

function displayGraphFromSynthesis(){
    let url = '/generateDataFromSynthesis';
    let formObj = document.getElementById("dataForm")
    let formData = new FormData(formObj);
    postTestRequest(url, formData)
        .then(graphHTML => displayGraphHTML(graphHTML))
        .catch(error => console.error(error))
}


async function postTestRequest(url, formData) {
    return fetch(url, {
        method: 'POST',
        body: formData
    })
        .then((response) => response.text());
}

function displayGraphHTML(graphHTML){
    graphIFrameObj.style.display = 'block';
   let iFrameDoc = graphIFrameObj.document;
   if(graphIFrameObj.contentDocument){
		iFrameDoc = graphIFrameObj.contentDocument;
	}else if(graphIFrameObj.contentWindow){
		iFrameDoc = graphIFrameObj.contentWindow.document;
	}
	if(iFrameDoc){
		iFrameDoc.open();
		iFrameDoc.writeln(graphHTML);
		iFrameDoc.close();
	}
}


const graphIFrameObj = document.getElementById("graph");

const graphBtnObj = document.getElementById("startPlotBtn");
const csvGraphBtnObj = document.getElementById("startCsvPlotBtn");
const dataGraphBtnObj = document.getElementById("startDataPlotBtn");

graphBtnObj.addEventListener("click", displayGraph);
csvGraphBtnObj.addEventListener("click", displayGraphFromCsv);
dataGraphBtnObj.addEventListener("click", displayGraphFromSynthesis);

