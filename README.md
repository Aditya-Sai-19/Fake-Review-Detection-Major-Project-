# A Hybrid Fuzzy Logic-based Deep Learning Approach for Fake Review Detection and Sentiment Classification of Amazon Food Reviews

A web-based application that uses a hybrid deep learning approach combining **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks to classify Amazon food review sentiments and detect fake reviews.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [How It Works](#how-it-works)

---

## 🔍 Overview

This project applies deep learning techniques to analyze Amazon food reviews. It uses two neural network models:

1. **CNN Model** — Classifies review sentiment into **Positive**, **Neutral**, or **Negative** (rating 1–5).
2. **LSTM Model** — Detects whether a review is **Genuine** or **Fake**.

The system leverages a hybrid fuzzy logic approach to combine predictions from both models, providing comprehensive review analysis for retailers.

---

## ✨ Features

- **Sentiment Classification** — Predicts review sentiment (Positive / Neutral / Negative) using a CNN model trained on TF-IDF features.
- **Fake Review Detection** — Identifies whether a review is genuine or fake using an LSTM model.
- **Hybrid Prediction** — Combines CNN and LSTM outputs for comprehensive review analysis.
- **Dataset Preprocessing** — Automatic cleaning of text data (stopword removal, lemmatization, punctuation removal).
- **Feature Extraction** — TF-IDF vectorization with 500 features for model input.
- **Retailer Portal** — Signup/login system for retailers to manage and view review classifications.
- **Visualization** — Bar charts and graphs for performance metrics (Accuracy, Precision, Recall, F-Score) and review sentiment distribution.
- **Pre-trained Models** — Loads saved model weights (`.h5`) and architecture (`.json`) for faster startup.

---

## 🏗 Architecture

```
User Review Input
       │
       ▼
┌─────────────────┐
│  Text Preprocessing │  (Stopword removal, Lemmatization, Punctuation removal)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TF-IDF Vectorizer │  (500 features)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│  CNN  │ │  LSTM │
│Model  │ │Model  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
Sentiment   Fake/Genuine
Prediction  Prediction
    │         │
    └────┬────┘
         ▼
   Combined Result
```
<img width="300" height="168" alt="WhatsApp Image 2026-08-23 at 23 24 48 (1)" src="https://github.com/user-attachments/assets/b864d8e1-57e6-40e0-92e3-2504db78a2ce" />



---

## 🛠 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| **Backend**      | Python, Flask                       |
| **Deep Learning**| Keras (CNN + LSTM)                  |
| **ML Libraries** | scikit-learn, NLTK, pandas, numpy   |
| **Database**     | MySQL (PyMySQL connector)           |
| **Frontend**     | HTML, CSS, JavaScript               |
| **Visualization**| Matplotlib                          |

---

## 📁 Project Structure

```
.
├── Main.py                  # Flask application with routes and model logic
├── requirement.txt          # Python dependencies
├── run.bat                  # Windows batch file to run the application
├── DB.txt                   # SQL script to create database and tables
├── index.html               # Home page
├── Login.html               # Retailer login page
├── Signup.html              # Retailer signup page
├── SubmitReview.html        # Review submission page
├── RetailerScreen.html      # Retailer dashboard
├── Train.html               # Training results display page
├── ViewClassification.html  # Classification results display page
├── style.css                # Application stylesheet
│
├── model/                   # Saved ML models (auto-generated)
│   ├── cnn.json             # CNN model architecture
│   ├── cnn_weights.h5       # CNN model weights
│   ├── lstm.json            # LSTM model architecture
│   ├── lstm_weights.h5      # LSTM model weights
│   ├── reviews.txt.npy      # Preprocessed reviews cache
│   ├── rating.txt.npy       # Ratings cache
│   └── fake.txt.npy         # Fake labels cache
│
└── Dataset/
    └── Reviews.csv          # Amazon food reviews dataset
```

---

## ⚙️ Prerequisites

- **Python 3.6+**
- **MySQL Server** — Required for user registration and review storage
- **pip** — Python package manager
- **NLTK Data** — Downloaded via NLTK downloader (punkt, stopwords, wordnet)

---

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirement.txt
   ```

   Or install packages individually:

   ```bash
   pip install pandas==0.25.3
   pip install numpy==1.19.2
   pip install scikit-learn==0.22.2.post1
   pip install matplotlib==3.1.1
   pip install nltk==3.4.5
   pip install Flask==1.1.2
   pip install keras==2.3.1
   pip install tensorflow==1.14.0
   pip install h5py==2.10.0
   pip install PyMySQL==0.9.3
   ```

3. **Download NLTK data**

   Run this Python script to download required NLTK packages:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

---

## 🗄 Database Setup

1. **Install MySQL Server** if not already installed.

2. **Create the database** by running the SQL commands in `DB.txt`:

   ```sql
   create database amazonreviews;
   use amazonreviews;

   create table register(
       retailer_name varchar(50),
       gender varchar(20),
       contact_no varchar(12),
       address varchar(60),
       email varchar(40),
       username varchar(40) primary key,
       password varchar(40)
   );

   create table submit_reviews(
       reviewer_name varchar(50),
       review_text varchar(200),
       ratings int,
       review_date date
   );
   ```

3. **Update database credentials** in `Main.py` if your MySQL setup uses different credentials (default: `root`/`root` on port `3306`).

---

## 🚀 Running the Application

**On Windows:**
```bash
python Main.py
```
Or simply double-click `run.bat`.

**On Linux/macOS:**
```bash
python Main.py
```

The application will start at **http://127.0.0.1:9999**

---

## 📖 Usage

### 1. Train Models
- Navigate to **Train Hybrid CNN & LSTM** from the navigation menu.
- The first run will process the `Dataset/Reviews.csv` dataset, train both CNN and LSTM models, and save weights to the `model/` directory.
- Subsequent runs will load the saved models for faster startup.
- Training results (Accuracy, Precision, Recall, F-Score) are displayed in a table with comparison charts.

### 2. Submit a Review
- Click **Submit Your Reviews** in the navigation menu.
- Enter your name and review text.
- The system will:
  - Predict the **sentiment** (Positive / Neutral / Negative) using the CNN model.
  - Detect whether the review is **Genuine** or **Fake** using the LSTM model.
  - Store the result in the database.

### 3. Retailer Portal
- **Signup** — Retailers can register with their details (name, gender, contact, address, email, username, password).
- **Login** — Retailers can log in to access the dashboard.
- **View Classification** — Retailers can view all submitted reviews with their predicted ratings and sentiments, along with distribution graphs.

<img width="640" height="480" alt="WhatsApp Image 2026-08-23 at 23 24 49" src="https://github.com/user-attachments/assets/5d76b78e-99c3-4878-b45c-59550cc0a405" />



---

## 🧠 How It Works

### Data Preprocessing
1. **Text Cleaning** — Removes punctuation, special characters, and non-alphabetic tokens.
2. **Stopword Removal** — Filters out common English stopwords (e.g., "the", "is", "at").
3. **Lemmatization** — Reduces words to their base form using WordNet lemmatizer.

### Feature Extraction
- **TF-IDF Vectorization** transforms cleaned reviews into a 500-dimensional feature vector capturing term importance.

### CNN Model (Sentiment Classification)
- Two convolutional layers (32 filters each) with max-pooling
- Fully connected layer (256 units, ReLU)
- Output layer with softmax activation (5 classes for ratings 1–5)
- Trained for 10 epochs with Adam optimizer

### LSTM Model (Fake Review Detection)
- Three stacked LSTM layers (32 → 16 → 8 units) with dropout (0.3) for regularization
- Output layer with softmax activation (2 classes: Genuine / Fake)
- Trained for 10 epochs with Adam optimizer

### Fake Detection Logic
- A review is labeled **Fake** if the helpfulness numerator is less than the denominator (indicating the review was not found helpful by other users).

---

## 📊 Performance Metrics

The training page displays the following metrics for both models:

| Metric     | Description                                              |
|------------|----------------------------------------------------------|
| Accuracy   | Overall correct predictions / total predictions          |
| Precision  | Correct positive predictions / all positive predictions  |
| Recall     | Correct positive predictions / all actual positives      |
| F-Score    | Harmonic mean of Precision and Recall                    |

<img width="640" height="480" alt="WhatsApp Image 2026-08-23 at 23 24 48" src="https://github.com/user-attachments/assets/4ff2d85e-334e-4781-bfa9-501541152609" />


---

## ⚠️ Important Notes

- The application uses **MySQL** for data storage. Ensure MySQL is running on `localhost:3306` before starting.
- The default dataset is `Dataset/Reviews.csv`. Make sure this file exists before training.
- Model weights are saved in the `model/` directory after training. Delete these files to retrain from scratch.
- This project uses **TensorFlow 1.14** and **Keras 2.3.1**. If using newer versions, some API calls may need updating.

---

## 📄 License

This project is for academic and research purposes.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.
