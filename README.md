# Machine Learning Project

This folder contains the machine learning models and scripts used for incident detection in our project. The ML models are trained to detect incidents like fights, accidents, weapons, garbage, fires, potholes, and other events in real-time from surveillance footage.

## Prerequisites

Before you begin, make sure you have the following installed:
- **Python** (v3.8 or above)
- **pip** (Python package installer)
- **Virtual environment** (recommended for Python projects)
- **Jupyter Notebook** (optional, for running notebooks)

## Project Setup

### 1. Create a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies and avoid conflicts with global packages.

#### How to Create and Activate the Virtual Environment:
1. **Create the virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   - For macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - For Windows:
     ```bash
     .\venv\Scripts\activate
     ```

---

### 2. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file.

#### How to Install:
```bash
pip install -r requirements.txt
```

This will install dependencies such as TensorFlow, PyTorch, OpenCV, and others required for training and running the models.

---

## Available Scripts

### 1. `train_model.py`
This script is used to train the machine learning model on the dataset. You can customize the script to train for different incidents such as fights, accidents, and other events.

#### How to Run:
```bash
python scripts/train_model.py
```

Make sure you have the necessary data files in the `data/` folder before running the training script.

---

### 2. `evaluate_model.py`
This script is used to evaluate the performance of the trained model using a validation or test dataset.

#### How to Run:
```bash
python scripts/evaluate_model.py
```

---

### 3. Jupyter Notebooks
If you’re using Jupyter Notebooks for experiments and visualization, you can run the following command to start the Jupyter Notebook server.

#### How to Run:
```bash
jupyter notebook
```

You can then open the notebooks from the `/notebooks/` folder and run the cells interactively.

---

## Folder Structure

- **/data/**: Contains the training and testing datasets (make sure sensitive data is not uploaded to the repo).
- **/models/**: Stores the trained models and checkpoints.
- **/notebooks/**: Jupyter notebooks for experimentation and prototyping.
- **/scripts/**: Contains Python scripts for training, evaluation, and preprocessing.
- **requirements.txt**: Contains all the dependencies required for the ML project.

---

## Guidelines

- Use **camelCase** for script names (`trainModel.py`).
- Save all datasets in the `data/` folder (but make sure large datasets are not included in version control).
- Save trained models and checkpoints in the `models/` folder.
- Write modular and reusable code, especially when defining preprocessing steps and model architectures.




<!--
1) Get the location from the "cameras" table by camera_id = id (cameras table):
2) When the model detect any anomly then make a list of objects having each anomly with confidence score i-e (detections_confidence= { Fire: {confidence: [ 80,92,95,70]}, smoke: {confidence: [ 80,92,95,70]} , accident:{confidence: [ 80,92,95,70]}, pothole:{confidence: [ 80,92,95,70]}}  # correct the syntx .
2) now save the image (1 image of each anomly) of each anomly if it exits in the "detections_confidence" . (Fire, smoke, accident etc image in supabase storage ).
3) get the authority_role from the "role_mapping" dictnory and then find the email in users table with role = authoirty_role .
4) send the emails to each role of users if detected anomly .(if their is fire, smoke etc then send two emails to each role user).
5) save the data in the supabase database table ("detections") .
 6) After send the email save the detection_id with other data in "alert" table .
  -->
