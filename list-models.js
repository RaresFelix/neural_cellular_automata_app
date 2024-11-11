// list-models.js
const fs = require('fs');
const path = require('path');

// Print the models
function listModels() {
    const modelsDir = './models';
    try {
        const files = fs.readdirSync(modelsDir);
        const onnxFiles = files.filter(file => file.endsWith('.onnx'));
        
        console.log('Available ONNX models:');
        onnxFiles.forEach(file => {
            console.log(`- ${file}`);
        });
    } catch (err) {
        console.error('Error:', err.message);
    }
}

listModels();