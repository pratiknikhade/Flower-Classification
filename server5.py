
import http.server
import socketserver
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import cgi
import base64

# Load the trained model
model = tf.keras.models.load_model('flower_classifier_saved_model_4')

# Define image preprocessing function for prediction
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define HTML template with CSS and background color
HTML_PAGE = """
<!DOCTYPE html>
<!DOCTYPE html>
<html>
<head>
    <title>Flower Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #FFC0CB; /* Change this color to the desired background color */
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }
        .container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }
        h1 {
            text-align: center;
        }
        #upload-btn {
            display: block;
            width: 100%;
            margin: 20px 0;
            padding: 10px;
            font-size: 16px;
            color: #333;
            background-color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        #result {
            margin-top: 20px;
            text-align: center;
        }
        #uploaded-img {
            display: block;
            margin: 20px auto;
            max-width: 100%;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Flower Detection</h1>
        <input type="file" id="file-input" accept="image/*">
        <button id="upload-btn">Upload Image</button>
        <div id="result"></div>
        <img id="uploaded-img" src="" alt="Uploaded Image">
    </div>
    <script>
        document.getElementById('upload-btn').addEventListener('click', function() {
            var input = document.getElementById('file-input');
            var file = input.files[0];
            if (file) {
                var formData = new FormData();
                formData.append('file', file);
                fetch('/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    var resultDiv = document.getElementById('result');
                    resultDiv.innerText = "Predicted Flower: " + data.predicted_flower;
                    
                    var img = document.getElementById('uploaded-img');
                    img.src = URL.createObjectURL(file);  // Display uploaded image
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        });
    </script>
</body>
</html>


"""

# Define request handler
class CustomRequestHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Map URL path to a file path in the current directory
        root = './'  # Specify the directory containing your static files
        return super().translate_path(path, root)

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        content_type, _ = cgi.parse_header(self.headers['Content-Type'])
        if content_type == 'multipart/form-data':
            form_data = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            if 'file' in form_data:
                file_item = form_data['file']
                if file_item.file:
                    try:
                        img = Image.open(io.BytesIO(file_item.file.read()))
                        img_array = preprocess_image(img)
                        prediction = model.predict(img_array)
                        predicted_class_index = np.argmax(prediction)
                        
                        # Assuming class_names is a list of class names
                        class_names = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 
                                       'carnation', 'common_daisy', 'coreopsis', 'daffodil', 'dandelion', 'iris', 'magnolia', 'not_flower',
                                       'rose', 'sunflower', 'tulip', 'water_lilly']
                        
                        # Define a threshold for confidence
                        confidence_threshold = 0.3
                        
                        if predicted_class_index == class_names.index('not_flower'):
                            predicted_flower_name = "Not a flower"
                        elif np.max(prediction) < confidence_threshold:
                            predicted_flower_name = "Not confident"
                        else:
                            predicted_flower_name = class_names[predicted_class_index]
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        response = {'predicted_flower': predicted_flower_name}
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                        return
                    except Exception as e:
                        error_response = {'error': str(e)}
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(error_response).encode('utf-8'))

# Define server parameters
HOST = 'localhost'
PORT = 8000

# Create server instance with custom request handler
httpd = http.server.HTTPServer((HOST, PORT), CustomRequestHandler)

# Start the server
print(f'Server running on http://{HOST}:{PORT}')
httpd.serve_forever()
