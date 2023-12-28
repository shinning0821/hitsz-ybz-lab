var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) {
    alert("Please select a file to analyze!");
    return; // 停止执行
  }

  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var url = "http://127.0.0.1:5000/analyze"
  xhr.open("POST", url, true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      if (response.hasOwnProperty("image_url")) {
        // 获取返回的图像URL
        var imageUrl = response["image_url"];
        // 在名为 "result-label" 的DOM元素中显示图像
        el("result-label").innerHTML = `<img src="${imageUrl}" alt="Analyzed Image">`;
      } else if (response.hasOwnProperty("error")) {
        alert("Error: " + response["error"]);
      }
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}

// 辅助函数，简化获取DOM元素的操作
function el(id) {
  return document.getElementById(id);
}


