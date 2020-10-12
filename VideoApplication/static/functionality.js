function opencam() {
    var x = document.createElement("IMG");
    x.src = "/video_feed";
    x.id = "image"
    var element = document.getElementById("div1");
    element.appendChild(x);
}

function closecam() {
    var parent = document.getElementById("div1");
    var child = document.getElementById("image");
    parent.removeChild(child);
}