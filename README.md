# Diabetes Classification using Deep Learning

📌 Project Overview

This project focuses on building a deep learning model to classify whether a person has diabetes using the Pima Indians Diabetes Dataset. The dataset contains medical diagnostic measurements such as:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

The main objective was to experiment with different model architectures and hyperparameter tuning strategies to achieve an optimal balance between training accuracy and validation accuracy while reducing overfitting.

⚙️ Methodology

Data Preprocessing

Handled missing/invalid values in the dataset.

Normalized feature values for better convergence.

Split data into training and validation sets.

Model Building

Started with a basic feedforward neural network.

Used ReLU activation for hidden layers and sigmoid activation for the output layer (binary classification).

Hyperparameter Tuning

Optimizer Tuning: Experimented with optimizers like Adam, RMSprop, SGD.

Number of Nodes: Varied the number of neurons in hidden layers to find the best configuration.

Number of Layers: Tried deeper networks to improve representation capacity.

Achieved ~99% accuracy on training data, but validation accuracy remained low → sign of overfitting.

Overfitting Reduction

Introduced Dropout Layers.

Applied regularization strategies.

Tuned hyperparameters carefully to ensure balance.

Final Model Performance

Achieved ~80% accuracy on both training and validation datasets, indicating a well-generalized model.

📊 Results
Stage	Training Accuracy	Validation Accuracy
Initial Model	~76%	~74%
After Optimizer Tuning	~85–90%	~75%
After Node/Layer Tuning	~99%	~72%
Final Model (with Dropout)	~80%	~80%
🛠️ Tech Stack

Language: Python

Libraries: TensorFlow / Keras, NumPy, Pandas, Matplotlib, Scikit-learn

🚀 How to Run

Clone the repository:

git clone https://github.com/yourusername/diabetes-classification.git
cd diabetes-classification


Install dependencies:

pip install -r requirements.txt


Run the training script:

python train.py


Evaluate the model:

python evaluate.py

📈 Future Work

Implement cross-validation for more robust performance estimation.

Try advanced architectures (e.g., CNN on tabular data, or attention-based models).

Experiment with feature engineering for improved accuracy.

Deploy the model using Flask / FastAPI / Streamlit for real-world usage.
