/**
 * script.js
 * Handles the functionality for Neural Cellular Automata Visualization.
 */

const DEBUG_MODE = false; // Set to true to enable dropdown for model selection

let session = null; // ONNX Runtime session
let feeds = null; // Current input tensor
let speedInterval = null; // Interval ID for evolution loop
let isRunning = false; // Flag to indicate if evolution is running
let isMouseDown = false; // Flag to indicate if mouse is down

// Add a global variable to store the current and last selected mapping
let currentMapping = 'rgba'; // Default mapping
let lastSelectedMapping = 'rgba'; // Last permanently selected mapping

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
        inputData[0 * height * width + centerIndex] = 0.8; // Red
        inputData[1 * height * width + centerIndex] = 0.8; // Green
        inputData[2 * height * width + centerIndex] = 0.8; // Blue
        inputData[3 * height * width + centerIndex] = 0.8; // Alpha

        // Set remaining channels to 0 if needed
        for (let c = 4; c < channels; c++) {
            inputData[c * height * width + centerIndex] = 0.8;
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
    ctx.imageSmoothingEnabled = false;

    const mappingOption = currentMapping;

    try {
        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        const output = results[outputName];

        // Compute statistics
        const data = output.data;
        const totalElements = data.length;
        let sum = 0;
        let max = -Infinity;
        let min = Infinity;

        for (let i = 0; i < totalElements; i++) {
            const value = data[i];
            sum += value;
            if (value > max) max = value;
            if (value < min) min = value;
        }

        const average = sum / totalElements;

        // Log statistics to the console
        console.log(`Step:`);
        console.log(`  Average Activation: ${average.toFixed(4)}`);
        console.log(`  Max Activation: ${max.toFixed(4)}`);
        console.log(`  Min Activation: ${min.toFixed(4)}`);
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

            if (mapping.startsWith('c')) {
                // Handle cell channels (4-15) as grayscale
                const channelNum = parseInt(mapping.slice(1));
                const value = data[channelOffsets[channelNum] + pixelIndex];
                r = g = b = value;
            } else {
                switch(mapping) {
                    case 'rgba':
                        r = data[channelOffsets[0] + pixelIndex];
                        g = data[channelOffsets[1] + pixelIndex];
                        b = data[channelOffsets[2] + pixelIndex];
                        a = data[channelOffsets[3] + pixelIndex];
                        break;
                    case 'r':
                        r = data[channelOffsets[0] + pixelIndex];
                        break;
                    case 'g':
                        g = data[channelOffsets[1] + pixelIndex];
                        break;
                    case 'b':
                        b = data[channelOffsets[2] + pixelIndex];
                        break;
                    case 'a':
                        r = g = b = data[channelOffsets[3] + pixelIndex];
                        break;
                }
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

// Add this helper function after clamp()
function getModelNameFromPath(path) {
    // Extract filename from path and remove .onnx extension
    return path.split('/').pop();
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
 * Handles click and touch events on the canvas to erase.
 * @param {MouseEvent|TouchEvent} event - The event object.
 */
function handleCanvasErase(event) {
    const canvas = event.target;
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if (event.type.startsWith('touch')) {
        const touch = event.touches[0] || event.changedTouches[0];
        clientX = touch.clientX;
        clientY = touch.clientY;
    } else {
        clientX = event.clientX;
        clientY = event.clientY;
    }
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Calculate position
    const x = Math.floor((clientX - rect.left) * scaleX);
    const y = Math.floor((clientY - rect.top) * scaleY);
    
    // Erase circle at position
    eraseCircle(x, y, 5, canvas);
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
    const mappingOption = currentMapping;
    renderVisualization(data, width, height, ctx, mappingOption);
}

/**
 * Populates the model selection UI based on DEBUG_MODE.
 */
async function populateModelSelect() {
    try {
        // Fetch models list from models.json
        const response = await fetch('models.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const modelEmojis = await response.json();

        // Convert modelEmojis to models array
        const models = Object.keys(modelEmojis).map(file => ({
            name: file.replace('.onnx', '').replace(/_/g, ' '),
            path: `models/${file}`, // Ensure models are in the 'models' folder
            emoji: modelEmojis[file]
        }));

        const modelSelect = document.getElementById('model-select');
        const emojiContainer = document.getElementById('model-emoji-container');
        modelSelect.innerHTML = ''; // Clear existing options
        emojiContainer.innerHTML = ''; // Clear existing emojis

        if (DEBUG_MODE) {
            // Populate dropdown menu
            if (models.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No models available';
                modelSelect.appendChild(option);
                modelSelect.disabled = true;
                console.log('No models found in models directory');
                return;
            }
            
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = model.name;
                modelSelect.appendChild(option);
            });
            
            // Initialize with the first model
            if (models.length > 0) {
                await initializeModel(models[0].path);
            }
        } else {
            // Populate emoji-based radio buttons
            if (models.length === 0) {
                const span = document.createElement('span');
                span.textContent = 'No models available';
                emojiContainer.appendChild(span);
                return;
            }
            
            models.forEach(model => {
                const label = document.createElement('label');
                label.classList.add('emoji-radio');
                
                const radio = document.createElement('input');
                radio.type = 'radio';
                radio.name = 'model';
                radio.value = model.path;
                if (model === models[0]) radio.checked = true;
                
                const emoji = document.createElement('span');
                emoji.textContent = model.emoji || '❓';
                
                label.appendChild(radio);
                label.appendChild(emoji);
                emojiContainer.appendChild(label);
                
                // Add event listener for radio buttons
                radio.addEventListener('change', async () => {
                    await initializeModel(model.path);
                });
            });
            
            // Initialize with the first model
            if (models.length > 0) {
                await initializeModel(models[0].path);
            }
        }
    } catch (err) {
        console.error('Error loading models list:', err);
        const modelSelect = document.getElementById('model-select');
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        modelSelect.disabled = true;
    }
}

// Modify the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', () => {
    // Replace the existing model initialization with populateModelSelect
    populateModelSelect();
    
    const modelSelect = document.getElementById('model-select');
    const emojiContainer = document.getElementById('model-emoji-container'); // New container
    const restartButton = document.getElementById('restart-button');
    const stopButton = document.getElementById('stop-button');
    const speedSlider = document.getElementById('speed');

    if (DEBUG_MODE) {
        modelSelect.addEventListener('change', async () => {
            if (isRunning) {
                stopEvolution();
            }
            await initializeModel(modelSelect.value);
        });
    }

    restartButton.addEventListener('click', async () => {
        if (isRunning) {
            stopEvolution();
        }
        // Reinitialize with current model path
        const selectedModel = DEBUG_MODE ? modelSelect.value : document.querySelector('input[name="model"]:checked').value;
        await initializeModel(selectedModel);
    });

    stopButton.addEventListener('click', stopEvolution);
    speedSlider.addEventListener('input', handleSpeedChange);

    // Replace canvas click listener with new listeners
    addCanvasListeners();

    // Channel mapping dropdown listeners (moved outside DEBUG_MODE condition)
    const dropdownItems = document.querySelectorAll('.dropdown-item');
    dropdownItems.forEach(item => {
        item.addEventListener('mouseover', () => {
            const value = item.getAttribute('data-value');
            updateMapping(value, false);
        });

        item.addEventListener('click', () => {
            const value = item.getAttribute('data-value');
            updateMapping(value, true);
        });
    });

    const dropdownContent = document.querySelector('.dropdown-content');
    dropdownContent.addEventListener('mouseleave', () => {
        currentMapping = lastSelectedMapping;
        // Re-render the visualization with the last selected mapping
        const canvas = document.getElementById('rgba-canvas');
        const ctx = canvas.getContext('2d');
        const data = feeds[session.inputNames[0]].data;
        const dims = feeds[session.inputNames[0]].dims;
        renderVisualization(data, dims[2], dims[3], ctx, currentMapping);
    });
});

/**
 * Updates the channel mapping and re-renders the visualization.
 * @param {string} mapping - The selected channel mapping option.
 * @param {boolean} isPermanent - Flag indicating if the mapping is permanent (clicked) or temporary (hovered).
 */
function updateMapping(mapping, isPermanent = false) {
    if (isPermanent) {
        // Update the last selected mapping
        lastSelectedMapping = mapping;
        // Update the dropdown button text
        document.querySelector('.dropdown-button').textContent = getMappingLabel(mapping);
    } else {
        // Temporary mapping on hover
        currentMapping = mapping;
    }

    // Re-render the visualization with the current mapping
    const canvas = document.getElementById('rgba-canvas');
    const ctx = canvas.getContext('2d');
    const data = feeds[session.inputNames[0]].data;
    const dims = feeds[session.inputNames[0]].dims;
    renderVisualization(data, dims[2], dims[3], ctx, currentMapping);
}

/**
 * Returns the display label for a given mapping value.
 * @param {string} mapping - The mapping value.
 * @returns {string} - The display label.
 */
function getMappingLabel(mapping) {
    const labels = {
        'rgba': 'RGBA (Channels 0-3)',
        'r': 'Red (Channel 0)',
        'g': 'Green (Channel 1)',
        'b': 'Blue (Channel 2)',
        'a': 'Alpha (Channel 3)',
        'c4': 'Cell Channel 4',
        'c5': 'Cell Channel 5',
        'c6': 'Cell Channel 6',
        'c7': 'Cell Channel 7',
        'c8': 'Cell Channel 8',
        'c9': 'Cell Channel 9',
        'c10': 'Cell Channel 10',
        'c11': 'Cell Channel 11',
        'c12': 'Cell Channel 12',
        'c13': 'Cell Channel 13',
        'c14': 'Cell Channel 14',
        'c15': 'Cell Channel 15'
    };
    return labels[mapping] || 'RGBA (Channels 0-3)';
}

function addCanvasListeners() {
    const canvas = document.getElementById('rgba-canvas');
    
    // Mouse Events
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

    // Touch Events
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault(); // Prevent scrolling
        isMouseDown = true;
        handleCanvasErase(e);
    }, { passive: false });
    
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault(); // Prevent scrolling
        if (isMouseDown) {
            handleCanvasErase(e);
        }
    }, { passive: false });
    
    canvas.addEventListener('touchend', () => {
        isMouseDown = false;
    });
    
    canvas.addEventListener('touchcancel', () => {
        isMouseDown = false;
    });
}
