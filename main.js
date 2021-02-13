function myFunction() {
	
	var request = new XMLHttpRequest();
	request.open("GET", "http://127.0.0.1:8000/", true);
	
	request.onreadystatechange = function() {
		if (request.readyState === 4 && request.status === 200) {
			var json = JSON.parse(request.responseText);
			document.getElementById("translation-result").textContent = request.responseText;
		}
	};
	
	request.send();
}

