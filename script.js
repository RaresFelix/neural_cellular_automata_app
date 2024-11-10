/**
 * script.js
 * Handles the functionality for Neural Cellular Automata Visualization.
 */

let session = null; // ONNX Runtime session
let feeds = null; // Current input tensor
let speedInterval = null; // Interval ID for evolution loop
let isRunning = false; // Flag to indicate if evolution is running
let isMouseDown = false; // Flag to indicate if mouse is down

/**
 * Initializes the ONNX Runtime session and the initial input tensor.
 * @param {string} modelPath - The path to the ONNX model file.
 */
async function initializeModel(modelPath) {
    console.log(`Loading model from ${modelPath}...`);

    try {
        // Initialize ONNX Runtime session with the selected model
        session = await ort.InferenceSession.create(modelPath);
        console.log(`Model loaded successfully from ${modelPath}.`);

        // Define input dimensions
        const batchSize = 1;      // Typically 1 for single inference
        const channels = 16;      // Number of channels
        const height = 64;        // Grid height
        const width = 64;         // Grid width

        // Create a Float32Array filled with zeros
        let inputData = new Float32Array(batchSize * channels * height * width);

        // Calculate the index for the center pixel
        const centerY = Math.floor(height / 2);
        const centerX = Math.floor(width / 2);
        const centerIndex = centerY * width + centerX;

        // Set RGBA channels of the center pixel to represent a gray cell
        inputData[0 * height * width + centerIndex] = 0.5; // Red
        inputData[1 * height * width + centerIndex] = 0.5; // Green
        inputData[2 * height * width + centerIndex] = 0.5; // Blue
        inputData[3 * height * width + centerIndex] = 1.0; // Alpha

        // Set remaining channels to 0 if needed
        for (let c = 4; c < channels; c++) {
            inputData[c * height * width + centerIndex] = 0.0;
        }

        // Create an ONNX tensor for the initial input
        let inputTensor = new ort.Tensor('float32', inputData, [batchSize, channels, height, width]);

        // Prepare feeds (input map)
        const inputName = session.inputNames[0]; // Automatically get the input name
        feeds = { [inputName]: inputTensor };

        console.log(`Initialization complete.`);
        
        // Start evolution automatically
        startEvolution();
    } catch (err) {
        console.error(`Error loading model: ${err.message}`);
    }
}

/**
 * Starts the real-time evolution loop based on the selected speed.
 */
function startEvolution() {
    if (isRunning) return; // Prevent multiple intervals

    isRunning = true;
    document.getElementById('stop-button').disabled = false;

    const speedValue = parseInt(document.getElementById('speed').value);
    const interval = getIntervalFromSpeed(speedValue);

    speedInterval = setInterval(runStep, interval);
    console.log(`Evolution started at speed setting ${speedValue}.`);
}

/**
 * Stops the real-time evolution loop.
 */
function stopEvolution() {
    if (!isRunning) return;

    clearInterval(speedInterval);
    speedInterval = null;
    isRunning = false;
    document.getElementById('stop-button').disabled = true;

    console.log(`Evolution stopped.`);
}

/**
 * Maps the speed slider value to the corresponding interval in milliseconds.
 * @param {number} speedValue - The value from the speed slider (0 to 6).
 * @returns {number} - The interval in milliseconds.
 */
function getIntervalFromSpeed(speedValue) {
    // Define speed mappings
    const speedMappings = {
        0: 200, // 1/10x = 5 steps/sec
        1: 100, // 1/5x = 10 steps/sec
        2: 40,  // 1/2x = 25 steps/sec
        3: 20,  // 1x = 50 steps/sec
        4: 10,  // 2x = 100 steps/sec
        5: 5,   // 4x = 200 steps/sec
        6: 2    // 8x = 400 steps/sec
    };

    return speedMappings[speedValue] || 20; // Default to 1x if undefined
}

/**
 * Runs a single step of the model and updates the visualization.
 */
async function runStep() {
    const canvas = document.getElementById('rgba-canvas');
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false; // Disable smoothing for pixelated look

    // Retrieve user inputs
    const threshold = parseFloat(document.getElementById('threshold').value) || 0.1;
    const mappingOption = document.getElementById('mapping').value;

    try {
        // Run the model
        const results = await session.run(feeds);

        // Extract output (assuming single output; adjust if multiple)
        const outputName = session.outputNames[0];
        const output = results[outputName];

        // Compute statistics
        const data = output.data;
        const totalElements = data.length;
        let sum = 0;
        let max = -Infinity;
        let min = Infinity;
        let activeCells = 0;

        for (let i = 0; i < totalElements; i++) {
            const value = data[i];
            sum += value;
            if (value > max) max = value;
            if (value < min) min = value;
            if (value > threshold) activeCells++;
        }

        const average = sum / totalElements;

        // Log statistics to the console
        console.log(`Step:`);
        console.log(`  Average Activation: ${average.toFixed(4)}`);
        console.log(`  Max Activation: ${max.toFixed(4)}`);
        console.log(`  Min Activation: ${min.toFixed(4)}`);
        console.log(`  Active Cells (> ${threshold}): ${activeCells}`);
        console.log('----------------------------------------');

        // Prepare the output as the next input
        feeds = { [session.inputNames[0]]: new ort.Tensor('float32', data, output.dims) };

        // Render RGBA Visualization
        renderVisualization(data, output.dims[2], output.dims[3], ctx, mappingOption);
    } catch (err) {
        console.error(`Error during inference: ${err.message}`);
        stopEvolution();
    }
}

/**
 * Renders the visualization based on the selected mapping option.
 * @param {Float32Array} data - The output tensor data.
 * @param {number} width - Width of the grid.
 * @param {number} height - Height of the grid.
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {string} mapping - The selected channel mapping option.
 */
function renderVisualization(data, width, height, ctx, mapping) {
    const imageData = ctx.createImageData(width, height);
    const pixelData = imageData.data;

    // Precompute channel offsets
    const channelOffsets = [];
    const channels = 16; // Assuming 16 channels as per the input
    for (let c = 0; c < channels; c++) {
        channelOffsets.push(c * height * width);
    }

    // Iterate over each pixel
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixelIndex = y * width + x;

            let r = 0, g = 0, b = 0, a = 255; // Default Alpha to 255

            switch(mapping) {
                case 'rgba':
                    r = data[channelOffsets[0] + pixelIndex];
                    g = data[channelOffsets[1] + pixelIndex];
                    b = data[channelOffsets[2] + pixelIndex];
                    a = data[channelOffsets[3] + pixelIndex];
                    break;
                case 'r':
                    r = data[channelOffsets[0] + pixelIndex];
                    g = 0;
                    b = 0;
                    a = 255;
                    break;
                case 'g':
                    r = 0;
                    g = data[channelOffsets[1] + pixelIndex];
                    b = 0;
                    a = 255;
                    break;
                case 'b':
                    r = 0;
                    g = 0;
                    b = data[channelOffsets[2] + pixelIndex];
                    a = 255;
                    break;
                case 'a':
                    r = 0;
                    g = 0;
                    b = 0;
                    a = data[channelOffsets[3] + pixelIndex];
                    break;
                // Add more cases if needed
                default:
                    r = data[channelOffsets[0] + pixelIndex];
                    g = data[channelOffsets[1] + pixelIndex];
                    b = data[channelOffsets[2] + pixelIndex];
                    a = data[channelOffsets[3] + pixelIndex];
            }

            // Normalize and map to [0, 255]
            const mappedR = clamp(Math.round(r * 255), 0, 255);
            const mappedG = clamp(Math.round(g * 255), 0, 255);
            const mappedB = clamp(Math.round(b * 255), 0, 255);
            const mappedA = clamp(Math.round(a * 255), 0, 255);

            const canvasIndex = pixelIndex * 4;
            pixelData[canvasIndex] = mappedR;     // Red
            pixelData[canvasIndex + 1] = mappedG; // Green
            pixelData[canvasIndex + 2] = mappedB; // Blue
            pixelData[canvasIndex + 3] = mappedA; // Alpha
        }
    }

    // Put the image data onto the canvas
    ctx.putImageData(imageData, 0, 0);
}

/**
 * Clamps a number between a minimum and maximum value.
 * @param {number} num - The number to clamp.
 * @param {number} min - The minimum value.
 * @param {number} max - The maximum value.
 * @returns {number} - The clamped number.
 */
function clamp(num, min, max) {
    return Math.min(Math.max(num, min), max);
}

/**
 * Handles changes in the speed slider to adjust the evolution rate.
 */
function handleSpeedChange() {
    const speedValue = parseInt(document.getElementById('speed').value);

    if (isRunning) {
        // Restart the interval with the new speed
        clearInterval(speedInterval);
        const interval = getIntervalFromSpeed(speedValue);
        speedInterval = setInterval(runStep, interval);
        console.log(`Speed changed to setting ${speedValue}.`);
    }
}

/**
 * Handles click events on the canvas
 * @param {MouseEvent} event - The click event
 */
function handleCanvasErase(event) {
    const canvas = event.target;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Calculate click position in canvas coordinates
    const x = Math.floor((event.clientX - rect.left) * scaleX);
    const y = Math.floor((event.clientY - rect.top) * scaleY);
    
    // Erase circle and update simulation
    eraseCircle(x, y, 5, canvas); // Pass canvas reference
}

/**
 * Erases a circle in the current simulation state
 * @param {number} centerX - Circle center X coordinate
 * @param {number} centerY - Circle center Y coordinate
 * @param {number} radius - Circle radius in pixels
 * @param {HTMLCanvasElement} canvas - The canvas element
 */
function eraseCircle(centerX, centerY, radius, canvas) { // Add canvas parameter
    if (!feeds || !session) return;

    const inputTensor = feeds[session.inputNames[0]];
    const data = inputTensor.data;
    const [batchSize, channels, height, width] = inputTensor.dims;

    // Iterate over pixels in bounding box of circle
    for (let y = Math.max(0, centerY - radius); y < Math.min(height, centerY + radius); y++) {
        for (let x = Math.max(0, centerX - radius); x < Math.min(width, centerX + radius); x++) {
            // Check if pixel is within circle
            const dx = x - centerX;
            const dy = y - centerY;
            if (dx * dx + dy * dy <= radius * radius) {
                // Erase all channels at this position
                const pixelIndex = y * width + x;
                for (let c = 0; c < channels; c++) {
                    data[c * height * width + pixelIndex] = 0;
                }
            }
        }
    }

    // Update feeds with modified data
    feeds = { [session.inputNames[0]]: new ort.Tensor('float32', data, inputTensor.dims) };
    
    // Force immediate visualization update
    const ctx = canvas.getContext('2d');
    const mappingOption = document.getElementById('mapping').value;
    renderVisualization(data, width, height, ctx, mappingOption);
}

// Attach event listeners once the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the model with the default selection
    const modelSelect = document.getElementById('model-select');
    initializeModel(modelSelect.value);

    // Add event listener for model selection change
    modelSelect.addEventListener('change', async () => {
        // If the simulation is running, stop it
        if (isRunning) {
            stopEvolution();
        }

        // Reinitialize the model with the selected model
        await initializeModel(modelSelect.value);

        // Start the evolution
        startEvolution();
        console.log(`Model changed to ${modelSelect.selectedOptions[0].text}.`);
    });

    const restartButton = document.getElementById('restart-button');
    const stopButton = document.getElementById('stop-button');
    const speedSlider = document.getElementById('speed');

    restartButton.addEventListener('click', async () => {
        // If the simulation is running, stop it
        if (isRunning) {
            stopEvolution();
        }

        // Reinitialize the model with a gray cell
        await initializeModel();

        // Start the evolution
        startEvolution();
        console.log(`Model restarted with a gray cell.`);
    });

    stopButton.addEventListener('click', stopEvolution);
    speedSlider.addEventListener('input', handleSpeedChange);

    // Replace canvas click listener with new listeners
    addCanvasListeners();
});

function addCanvasListeners() {
    const canvas = document.getElementById('rgba-canvas');
    
    canvas.addEventListener('mousedown', (e) => {
        isMouseDown = true;
        handleCanvasErase(e);
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (isMouseDown) {
            handleCanvasErase(e);
        }
    });
    
    canvas.addEventListener('mouseup', () => {
        isMouseDown = false;
    });
    
    canvas.addEventListener('mouseleave', () => {
        isMouseDown = false;
    });
}
