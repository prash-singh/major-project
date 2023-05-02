const express = require('express');
const data = require("./image_class.json");
const multer = require('multer');
//const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');

const app = express();
//app.use(express.json());
app.use(express.static('./static') );
app.use(cors());

// Set up multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage }).single('image');

// Load your custom ML model using TensorFlow.js in function preload()
let modelPromise;
(async function preload(){
    modelPromise = await tf.loadLayersModel('http:localhost:8000/model/model.json');
})();



// Preprocess the image using TensorFlow.js
const preprocessImage = (buffer) => {
    // Load the image using tfnode modules
    const tfimage = tf.node.decodeImage(buffer);

    // Resize the image to your desired dimensions
    const resizedImage = tf.image.resizeBilinear(tfimage, [224, 224]);

    // Expand dimensions to match the model input shape [batchSize, height, width, channels]
    const expandedImage = resizedImage.expandDims();

    // Convert image to tensor
    const imageTensor = expandedImage.toFloat();

    // Normalize the pixel values
    const normalizedImage = imageTensor.div(tf.scalar(255));

    return normalizedImage;
};


// Define a route to handle image uploads
app.post('/upload', upload, async (req, res) => {
    try {
        const model = await modelPromise;

        // Preprocess the image
        const imageBuffer = req.file.buffer;
        const preprocessedImage = preprocessImage(imageBuffer);

        // Make predictions using the ML model
        const predictions = model.predict(preprocessedImage);

        // Convert predictions to JSON format
        const predictionsData = predictions.arraySync();

        const predictedClassIndex = predictionsData[0].indexOf(Math.max(...predictionsData[0]));
        // Return the predictions as a response
        res.json({ predictions: data[predictedClassIndex]});
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'An error occurred' });
    }
});

// Start the server
app.listen(8000, () => {
    console.log('Server is running on port 8000');
});
