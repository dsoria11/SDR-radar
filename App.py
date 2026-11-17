from flask import Flask, jsonify, render_template_string
import numpy as np

app = Flask(__name__)

running = False
buffer_size = 100
fft_size = 256

waterfall = np.zeros((buffer_size, fft_size))

html_code = """
<!DOCTYPE html>
<html>
<head>
    <title>SDR Radar Waterfall</title>
</head>

<body>
    <h2>SDR Radar Waterfall</h2>

    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>

    <br><br>
    <canvas id="waterfall" width="600" height="400"></canvas>

<script>
let canvas = document.getElementById("waterfall");
let ctx = canvas.getContext("2d");

function start() {
    fetch('/start');
}

function stop() {
    fetch('/stop');
}

function drawWaterfall(data) {
    let rows = data.length;
    let cols = data[0].length;

    let image = ctx.createImageData(cols, rows);

    for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
            let value = data[y][x];

            // Normalize value (adjust scale if needed)
            let color = Math.min(255, value * 5);

            let index = (y * cols + x) * 4;

            image.data[index] = color;       // R
            image.data[index+1] = 0;         // G
            image.data[index+2] = 255-color; // B
            image.data[index+3] = 255;       // Alpha
        }
    }

    ctx.putImageData(image, 0, 0);
}

async function update() {
    let response = await fetch('/data');
    let data = await response.json();

    drawWaterfall(data);
}

setInterval(update, 200);
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html_code)

@app.route("/start")
def start():
    global running
    running = True
    return "Started"

@app.route("/stop")
def stop():
    global running
    running = False
    return "Stopped"

@app.route("/data")
def data():
    global waterfall

    if running:
        t = np.linspace(0, 1, fft_size)

        # Cleaner signal for better visualization
        signal = np.cos(2*np.pi*10*t) + np.random.randn(fft_size)*0.1

        fft_data = np.abs(np.fft.fft(signal))

        waterfall = np.roll(waterfall, -1, axis=0)
        waterfall[-1, :] = fft_data

    return jsonify(waterfall.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)