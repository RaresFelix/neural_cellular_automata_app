const express = require('express');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const PORT = 3000;

// Ensure models directory exists
const modelsDir = path.join(__dirname, 'models');
(async () => {
    try {
        await fs.access(modelsDir);
    } catch {
        await fs.mkdir(modelsDir);
        console.log('Created models directory');
    }
})();

// Serve static files from the current directory
app.use(express.static(__dirname));

// API endpoint to list models
app.get('/api/models', async (req, res) => {
    try {
        const files = await fs.readdir(modelsDir);
        const models = files
            .filter(file => file.endsWith('.onnx'))
            .map(file => ({
                name: file.replace('.onnx', '').replace(/_/g, ' '),
                path: `/models/${file}` // Add leading slash
            }));
            
        if (models.length === 0) {
            console.log('No .onnx models found in models directory');
        }
        
        res.json(models);
    } catch (err) {
        console.error('Error reading models directory:', err);
        res.status(500).json({ error: 'Failed to list models' });
    }
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log(`Models directory: ${modelsDir}`);
});