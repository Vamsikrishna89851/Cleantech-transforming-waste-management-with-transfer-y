from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('waste_classifier.h5')
class_indices = {v: k for k, v in model._get_weights_manager().checkpoint.as_dict().items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        img = image.load_img(file, target_size=(224,224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        label = np.argmax(preds, axis=1)[0]
        return jsonify({'predicted_class': str(label)})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)