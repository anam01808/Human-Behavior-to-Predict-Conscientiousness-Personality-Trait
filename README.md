Personality Trait Detection: Conscientiousness 


Overview

This project focuses on detecting the personality trait "Conscientiousness " using a dataset derived from the Myers-Briggs Type Indicator (MBTI) available on Kaggle. The primary goal is to classify text data according to the "Openness" trait, leveraging natural language processing (NLP) techniques and machine learning algorithms. This project implements a text classification pipeline using various machine and deep learning models and natural language processing (NLP) techniques. The pipeline includes feature extraction with methods like textual and word embeddings , and models such as SVM, XGBoost, and neural networks using TensorFlow and Hugging Face transformers.

Dataset

Name: MBTI Type Dataset

Available Link: https://www.kaggle.com/datasets/datasnaek/mbti-type

Focus: This project focuses specifically on the "Openness" aspect, inferred from the Judging (J) vs. Perceiving (P) dimension.

Setup Instruction: To run the code for this project, we use Google Colab GPU and Kaggle GPU for faster computation. The following libraries are required: numpy, spacy, pandas, matplotlib, nltk, and sklearn. These libraries and tools provide the necessary computational resources and functions for text processing, model training, and evaluation.

Repository Structure

conscientiousness s_Personality_Trait_JP.ipynb: The primary notebook for personality trait analysis.

mbti_1.rar: Dataset file used for training and evaluation.

README.md: Project documentation.

1. Clone the Repository First, clone this repository to your local machine:

git clone https://github.com/anam01808/Human-Behavior-to-Predict-Conscientiousness-Personality-Trait.git

cd Personality-Code

2. Running on Google Colab You can run the notebook directly on Google Colab. Follow these steps:

a. Open Google Colab at colab.research.google.com.

b. Upload the notebook or use the Colab link from the repository.

c. Install the required dependencies by adding this to a code cell at the start:

pip install numpy
pip install spacy
pip install tqdm
pip install xgboost
pip install nltk
pip install emoji
pip install scikit-learn
pip install transformers
pip install tensorflow
pip install pandas
pip install gensim
pip install sentence-transformers
This command installs all the base libraries

3. Dataset (mbti_1.rar) Google Colab: Upload the dataset manually or link it from Google Drive.

4. Key Libraries Used

NumPy: For numerical computations
SpaCy: For NLP tasks
NLTK: Natural language processing toolkit
Scikit-learn: For model evaluation and feature extraction
XGBoost: Gradient boosting models
TensorFlow: For neural network models
Transformers: For fine-tuning pre-trained models
Gensim: For Word2Vec and GloVe embeddings
sentence-transformers: For advanced BERT sentence embeddings
Word2Vec: Pre-trained word vectors for semantic understanding.
GloVe: Global Vectors for word representation.
Sentence Embeddings: A transformer-based model for sentence embeddings using sentence-transformers.
Large Language Model: BERT Large
5. Run the Notebook Launch Notebook and open the file

6. Training the Model The notebook allows you to train and evaluate various models such as SVM, XGBoost, Random Forest, and a neural network. Make sure you have a dataset to use for training and adjust the train_dataloader and val_dataloader sections accordingly.

Notes

The code uses NLTK resources like punkt, averaged_perceptron_tagger, stopwords, and wordnet, which require downloading.

TensorFlow is used for neural network-based text classification.

Hugging Face's transformers library is used, which involves downloading pre-trained models and tokenizers.

Ensure that you have sufficient computational resources (GPU is recommended) when training models, especially when using deep learning methods.

If using Google Colab, you can enable GPU for faster training by navigating to Runtime > Change runtime type > Hardware accelerator > GPU.

Running the Code: Here is brief analysis of how to run the scripts or notebooks, including data preprocessing steps, training the models, and evaluating the results.

Preprocessing:

The preprocessing steps include the following:
Removal of URLs, emojis, digits, "|||", punctuation, and stopwords from the text data.
Lemmatization is applied to reduce words to their base forms.
Features Applied:

For Shallow Machine Learning and Ensemble Models, the following text features are used:
TF-IDF and POS tagging are integrated with models like Support Vector Machine (SVM), Decision Tree (DT), Logistic Regression (LR), Naive Bayes (NB), K-Nearest Neighbors (KNN), Random Forest (RF), XGBoost (XGB), Gradient Boosting (GB), and AdaBoost.
For Deep Learning Models, the following word embeddings are used:
Word2Vec, GloVe, and Sentence Embeddings are applied with deep models such as Long Short-Term Memory (LSTM), Bidirectional LSTM (Bi-LSTM), and LLM based models like BERT Large.
Performance Evaluation Measures:

The models are evaluated based on the following metrics:
Accuracy, Precision, Recall, and F1-score to provide a comprehensive assessment of the model’s performance.
These steps ensure that the data is properly prepared, features are extracted, and appropriate models are trained and evaluated.
