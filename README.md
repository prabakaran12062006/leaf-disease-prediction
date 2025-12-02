

Project Title: Early Detection of Plant Leaf Diseases Using Machine Learning

Features
* Detects multiple types of plant leaf diseases
* Uses a CNN-based deep learning model
* Image preprocessing and augmentation for better accuracy
* Easy-to-use script for prediction
* Well-structured dataset for training and testing

Technologies Used
* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib
* OpenCV
* Scikit-learn
📂 Project Structure

├── dataset/
│   ├── train/
│   ├── test/
│   └── valid/
├── models/
│   └── leaf_model.h5
├── notebooks/
│   └── training_notebook.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
├── README.md
└── requirements.txt


0⚙️ How to Run
1. Install Dependencies

```
pip install -r requirements.txt
```

2. Train the Model

```
python src/train.py
```

3. Predict Using an Image

```
python src/predict.py --image path_to_leaf_image.jpg
```
📊 Results
* Achieved high accuracy using CNN
* Model performs well with unseen test images
* Effective in distinguishing between healthy and diseased leaves

(Add accuracy % here if you have one)
🎯 Future Enhancements
* Add mobile app interface
* Deploy model with Flask or FastAPI
* Extend dataset with more leaf categories


