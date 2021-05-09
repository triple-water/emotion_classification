function delay_url() {
        var delay = document.getElementById("rest").innerHTML;
        var t = setTimeout("delay_url()", 5000);
        if (delay > 0) {
            delay--;
            document.getElementById("time").innerHTML = delay;
        } else {
            clearTimeout(t);
            window.location.href = "/video";
        }
    }