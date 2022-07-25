function displayGraph(){
    let url = '/generate-graph';
    let formObj = document.getElementById("displayForm")
    
    postTestRequest(url, formObj)
        .then(graphHTML => displayGraphHTML(graphHTML))
        .catch(error => console.error(error))
}
async function postTestRequest(url, formObj) {
    return fetch(url, {
        method: 'POST',
        body: formObj
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


const graphIFrameObj = document.getElementById("graphIFrame");
const graphBtnObj = document.getElementById("graphBtn");
graphBtnObj.addEventListener("click", displayGraph);
