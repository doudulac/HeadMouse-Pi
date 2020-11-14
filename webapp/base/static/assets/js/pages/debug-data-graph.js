var runScript = function() {
    var socket = io();
    var maxbuf = 300;
    var ave_buf = [];
    var cur_buf = [];
    var pos_buf = [];
    var vel_buf = [];
    var acc_buf = [];
    var fcx_buf = [];
    var fcy_buf = [];
    var dif_buf = [];
    var up_buf = [];
    var updateInterval = 30;   // ms
    var realtime_brow       = 'on'; //If == to on then fetch data every x seconds. else stop fetching

    var xraw_buf = [];
    var yraw_buf = [];
    var xpos_buf = [];
    var xvel_buf = [];
    var ypos_buf = [];
    var yvel_buf = [];
    var xdelt_buf = [];
    var xacc_buf = [];
    var ydelt_buf = [];
    var yacc_buf = [];
    var realtime_nose       = 'on'; //If == to on then fetch data every x seconds. else stop fetching

    var brow_dataset = {
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
        "face xangle": {
            xaxis: 1, yaxis: 1,
            label: "face xangle",
            data: fcx_buf,
        },
        "face yangle": {
            xaxis: 1, yaxis: 1,
            label: "face yangle",
            data: fcy_buf,
        },
        "height delta": {
            xaxis: 1, yaxis: 2,
            label: "height delta",
            data: dif_buf,
        },
        "raised": {
            xaxis: 1, yaxis: 1,
            label: "raised",
            data: up_buf,
        },

    };

    var nose_dataset = {
        "x raw": {
            xaxis: 1, yaxis: 1,
            label: "x raw",
            lines: { fill: false },
            data: xraw_buf,
        },
        "y raw": {
            xaxis: 1, yaxis: 1,
            label: "y raw",
            lines: { fill: false },
            data: yraw_buf,
        },
        "x pos": {
            xaxis: 1, yaxis: 1,
            label: "x pos",
            lines: { fill: false },
            data: xpos_buf,
        },
        "x vel": {
            xaxis: 1, yaxis: 2,
            label: "x vel",
            data: xvel_buf,
        },
        "y pos": {
            xaxis: 1, yaxis: 1,
            label: "y pos",
            lines: { fill: false },
            data: ypos_buf,
        },
        "y vel": {
            xaxis: 1, yaxis: 2,
            label: "y vel",
            data: yvel_buf,
        },
        "x delta": {
            xaxis: 1, yaxis: 2,
            label: "x delta",
            data: xdelt_buf,
        },
        "x acc": {
            xaxis: 1, yaxis: 2,
            label: "x acc",
            data: xacc_buf,
        },
        "y delta": {
            xaxis: 1, yaxis: 2,
            label: "y delta",
            data: ydelt_buf,
        },
        "y acc": {
            xaxis: 1, yaxis: 2,
            label: "y acc",
            data: yacc_buf,
        },
    };

    var browSeriesContainer = $("#brow-series");
    var noseSeriesContainer = $("#nose-series");
    var now = (new Date()).getTime();
    var i = 0;
    $.each(brow_dataset, function(key, val) {
        initData(now, val.data);
        val.color = i;
        ++i;
        browSeriesContainer.append("<br/><input type='checkbox' name='" + key +
            "' checked='checked' id='id" + key + "'></input>" +
            "&nbsp;<label for='id" + key + "'>"
            + val.label + "</label>");
    });
    i = 0;
    $.each(nose_dataset, function(key, val) {
        initData(now, val.data);
        val.color = i;
        ++i;
        noseSeriesContainer.append("<br/><input type='checkbox' name='" + key +
            "' checked='checked' id='id" + key + "'></input>" +
            "&nbsp;<label for='id" + key + "'>"
            + val.label + "</label>");
    });

    var brow_plot = $.plot('#brow-chart', getBrowData(),
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

    var nose_plot = $.plot('#nose-chart', getNoseData(),
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

    function getBrowData() {
        var data = [];
        browSeriesContainer.find("input:checked").each(function () {
            var key = $(this).attr("name");
            if (key && brow_dataset[key]) {
                data.push(brow_dataset[key]);
            }
        });
        return data;
    }

    function getNoseData() {
        var data = [];
        noseSeriesContainer.find("input:checked").each(function () {
            var key = $(this).attr("name");
            if (key && nose_dataset[key]) {
                data.push(nose_dataset[key]);
            }
        });
        return data;
    }

    function update_brow() {
        brow_plot.setData(getBrowData());
        brow_plot.setupGrid(true);
        brow_plot.draw();

        if (realtime_brow === 'on') {
          setTimeout(update_brow, updateInterval);
        }
    }

    function update_nose() {
        nose_plot.setData(getNoseData());
        nose_plot.setupGrid(true);
        nose_plot.draw();

        if (realtime_nose === 'on') {
          setTimeout(update_nose, updateInterval);
        }
    }

    function initData(now, data) {
        for (var i = 0; i < maxbuf; i++) {
            data[i] = [now - updateInterval * (maxbuf-i), 0];
        }
    }

    //REALTIME TOGGLE
    $('#realtime-brow .btn').click(function () {
        $(this).addClass('active').siblings().removeClass('active');
        if ($(this).data('toggle') === 'on') {
            realtime_brow = 'on';
            var now = (new Date()).getTime();
            $.each(brow_dataset, function(key, val) {
                initData(now, val.data);
            });
            socket.emit('toggle_debug_data', { brows: true });
        }
        else {
            realtime_brow = 'off';
            socket.emit('toggle_debug_data', { brows: false });
        }
        update_brow();
    });
    $('#realtime-nose .btn').click(function () {
        $(this).addClass('active').siblings().removeClass('active');
        if ($(this).data('toggle') === 'on') {
            realtime_nose = 'on';
            var now = (new Date()).getTime();
            $.each(nose_dataset, function(key, val) {
                initData(now, val.data);
            });
            socket.emit('toggle_debug_data', { nose: true });
        }
        else {
            realtime_nose = 'off';
            socket.emit('toggle_debug_data', { nose: false });
        }
        update_nose();
    });

    $('#ns-kf-q-btn').click(function () {
        socket.emit('kf_update', { nose: { Q: $('#ns-kf-q').val(), R: $('#ns-kf-r').val() } });
    });

    $('#ns-kf-r-btn').click(function () {
        socket.emit('kf_update', { nose: { Q: $('#ns-kf-q').val(), R: $('#ns-kf-r').val() } });
    });

    socket.on('brow_data', function(data) {
        var now = (new Date()).getTime();
        var i = 0;
        $.each(brow_dataset, function(key, val) {
            if (key === 'raised') {
                var v = 0;
                if (data[i] === 'up ') {
                    v = 5;
                }
                else if (data[i] === 'up+') {
                    v = 10;
                }
                if (val.data.push([now, v]) > maxbuf) { val.data.shift(); }
            }
            else {
                if (val.data.push([now, data[i]]) > maxbuf) { val.data.shift(); }
            }
            ++i;
        });
    });

    socket.on('nose_data', function(data) {
        var now = (new Date()).getTime();
        var i = 0;
        $.each(nose_dataset, function(key, val) {
            if (val.data.push([now, data[i]]) > maxbuf) { val.data.shift(); }
            ++i;
        });
    });

    //INITIALIZE REALTIME DATA FETCHING
    if (realtime_brow === 'on') {
        $('#realtime-brow .btn').first().click();
    } else {
        $('#realtime-brow .btn').last().click();
    }
    if (realtime_nose === 'on') {
        $('#realtime-nose .btn').first().click();
    } else {
        $('#realtime-nose .btn').last().click();
    }
};
