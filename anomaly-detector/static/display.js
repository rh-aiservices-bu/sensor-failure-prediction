function displayGraph(){
    let url = '/generate-graph';
    let formObj = document.getElementById("displayForm")
    
    postTestRequest(url, formObj)
        .then(graphHTML => displayTestGraphHTML(graphHTML))
        .catch(error => console.error(error))
}
async function postTestRequest(url, formObj) {
    return fetch(url, {
        method: 'POST',
        body: formObj
    })
        .then((response) => response.text());
}

function displayTestGraphHTML(graphHTML){
    testGraphIFrameObj.style.display = 'block';
   let iFrameDoc = testGraphIFrameObj.document;
   if(testGraphIFrameObj.contentDocument){
		iFrameDoc = testGraphIFrameObj.contentDocument;
	}else if(testGraphIFrameObj.contentWindow){
		iFrameDoc = testGraphIFrameObj.contentWindow.document;
	}
	if(iFrameDoc){
		iFrameDoc.open();
		iFrameDoc.writeln(graphHTML);
		iFrameDoc.close();
	}
}


const testGraphIFrameObj = document.getElementById("graphIFrame");
const testGraphBtnObj = document.getElementById("graphBtn");
graphBtnObj.addEventListener("click", displayGraph);