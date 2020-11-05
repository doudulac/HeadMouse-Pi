var runScript = function() {
    var socket = io();
    var maxbuf = 300;
    var ave_buf = [];
    var cur_buf = [];
    var pos_buf = [];
    var vel_buf = [];
    var acc_buf = [];
    var updateInterval = 30;   // ms
    var realtime       = 'on'; //If == to on then fetch data every x seconds. else stop fetching

    var datasets = {
        "average height": {
            xaxis: 1, yaxis: 1,
            label: "average height",
            data: ave_buf,
        },
        "current height": {
            xaxis: 1, yaxis: 1,
            label: "current height",
            data: cur_buf,
        },
        "kf position": {
            xaxis: 1, yaxis: 1,
            label: "kf position",
            data: pos_buf,
        },
        "kf velocity": {
            xaxis: 1, yaxis: 2,
            label: "kf velocity",
            data: vel_buf,
        },
        "kf acceleration": {
            xaxis: 1, yaxis: 2,
            label: "kf acceleration",
            data: acc_buf,
        },
    };

    var i = 0;
    $.each(datasets, function(key, val) {
        val.color = i;
        ++i;
    });

    // insert checkboxes
    var choiceContainer = $("#choices");
    $.each(datasets, function(key, val) {
        choiceContainer.append("<br/><input type='checkbox' name='" + key +
            "' checked='checked' id='id" + key + "'></input>" +
            "&nbsp;<label for='id" + key + "'>"
            + val.label + "</label>");
    });

    var eyebrow_plot = $.plot('#eyebrow', getData(),
    {
        legend: {
            show: true,
            position: "sw",
        },
        grid: {
            borderColor: '#f3f3f3',
            borderWidth: 1,
            tickColor: '#f3f3f3',
        },
        series: {
            color: '#3c8dbc',
            lines: {
                lineWidth: 2,
                show: true,
                fill: true,
            },
        },
        yaxes: [{min: 0, max: 50,}, {position: "right"}],
        xaxes: [{
            mode: "time",
            timeBase: "milliseconds",
            timeformat: "%I:%M:%S",
            timezone: "browser",
            show: true,
        }],
    });

    function getData() {
        var data = [];
        choiceContainer.find("input:checked").each(function () {
            var key = $(this).attr("name");
            if (key && datasets[key]) {
                data.push(datasets[key]);
            }
        });
        return data;
    }

    function update() {
        eyebrow_plot.setData(getData());
        eyebrow_plot.setupGrid(true);
        eyebrow_plot.draw();

        if (realtime === 'on') {
          setTimeout(update, 30);
        }
    }

    function initData(now, data) {
        for (var i = 0; i < maxbuf; i++) {
            data[i] = [now - updateInterval * (maxbuf-i), 0];
        }
    }

    var now = (new Date()).getTime();
    initData(now, ave_buf);
    initData(now, cur_buf);

    //INITIALIZE REALTIME DATA FETCHING
    if (realtime === 'on') {
        update();
    }
    //REALTIME TOGGLE
    $('#realtime .btn').click(function () {
        if ($(this).data('toggle') === 'on') {
            realtime = 'on';
        }
        else {
            realtime = 'off';
        }
        update();
    });

    socket.on('eyebrows', function(data) {
        var now = (new Date()).getTime();
        if (ave_buf.push([now, data[0]]) > maxbuf) { ave_buf.shift(); }
        if (cur_buf.push([now, data[1]]) > maxbuf) { cur_buf.shift(); }
        if (pos_buf.push([now, data[2]]) > maxbuf) { pos_buf.shift(); }
        if (vel_buf.push([now, data[3]]) > maxbuf) { vel_buf.shift(); }
        if (acc_buf.push([now, data[4]]) > maxbuf) { acc_buf.shift(); }
    });
};
