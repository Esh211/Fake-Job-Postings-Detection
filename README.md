# Fake-Job-Postings-Detection
A project that detects fake job postings using natural language processing (NLP) and machine learning techniques. The model is trained on real and fraudulent job postings to classify them accurately based on their text features.

## What the Project Does
This project implements a machine learning model to identify fraudulent job postings from text data. It uses NLP for feature extraction and applies models like Naive Bayes and neural networks for classification, achieving high accuracy in detecting fake job postings.

## Why the Project is Useful
Fake job postings are a growing issue, affecting job seekers and companies. This project provides a robust solution for detecting fraud in job listings, helping organizations and individuals avoid scams. It's useful for job boards, recruiters, and job seekers alike.

## Technologies Used
- **Python**: Programming language used for implementation.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Scikit-learn**: Used for machine learning models and feature extraction (CountVectorizer, TfidfVectorizer).
- **Seaborn & Matplotlib**: Libraries for data visualization.
- **Keras**: For building and training neural network models.

## Key Features
-  **Naive Bayes and Neural Networks**: Used for classification with high accuracy.
- **Text Preprocessing**: Includes tokenization, lemmatization, and removal of stopwords.
- **Feature Extraction**: Utilizes Bag of Words (BOW) and TF-IDF to convert text data into numerical form.
- **Confusion Matrix and Classification Report**: Used for performance evaluation and model comparison.
- **Accuracy**: Achieves strong accuracy in identifying fraudulent job postings.

## Getting Started
1. **Clone the Repository**:
git clone https://github.com/your_username/Fake-Job-Postings-Detection.git
cd Fake-Job-Postings-Detection

2. **Install Required Libraries**:
pip install -r requirements.txt

3. **Run the Jupyter Notebook**: Open fake_job_posting_detection.ipynb in Jupyter or Google Colab and follow the instructions to execute the code.

4. **Evaluate Results**: After running the notebook, you can evaluate the model’s performance using the following metrics:

- **Accuracy**:
  - **Bag of Words (BOW) Naive Bayes model**: mnb_bow_score = 0.85
  - **TF-IDF Naive Bayes model**: mnb_tfidf_score = 0.87
  - **Neural Network model**: accuracy_score = 0.89
     
- **Confusion Matrix**: Visualize the confusion matrix for both Naive Bayes and Neural Network models:
  - **BOW Naive Bayes model**
  - **TF-IDF Naive Bayes model**
  - **Neural Network model**
  
**Classification Report**:
- recision, recall, and F1-score for both real and fake job postings:
  java
    - **Copy code**:
      Classification Report for Naive Bayes (BOW and TF-IDF) rust
  - **Copy code**:
      Classification Report for Neural Network

**Visualizations**:
- **Word Clouds**: Visualize the most frequent words in real and fraudulent job postings:
  - **Real Job Postings**
  - **Fake Job Postings**
- **Model Accuracy and Loss**: Plot the accuracy and loss for the neural network model during training.

## Results
The model can accurately classify job postings as fraudulent or real, based on the text data provided. You will see the model’s performance through confusion matrices and classification reports.

## Getting Help
For any questions or issues, open an issue on the repository, and I will be happy to help.

## Maintainers and Contributors
This project is maintained by **Esha Govardhan, Pragati Gadkar, Tisha Nawani, Amrapali Chandanshiv**. Contributions are welcome! Fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

