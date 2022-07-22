function testGraph(){
    console.log("testGraph()");
    graphDivObj.innerHTML = "<img src='static/workingMsg.png'/>";
    let url = '/test-model';
    let formObj = document.getElementById("testForm")
    let formData = new FormData(formObj);
    postTestRequest(url, formData)
        .then(graphHTML => displayTestGraphHTML(graphHTML))
        .catch(error => console.error(error))
}
async function postTestRequest(url, formData) {
    return fetch(url, {
        method: 'POST',
        body: formData
    })
        .then((response) => response.text());
}

function displayTestGraphHTML(graphHTML){
    testGraphIFrameObj.style.display = 'block';
    //let graphHTML =  "<img src='data:image/png;base64, " + serializedImage +
    //    "' class='imgSize'/>" ;
    //testGraphDivObj.innerHTML = graphHTML;
   // testGraphDivObj.innerHTML = serializedImage;
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


const testGraphIFrameObj = document.getElementById("testGraphIFrame");
const testGraphBtnObj = document.getElementById("testingBtn");
testGraphBtnObj.addEventListener("click", testGraph);