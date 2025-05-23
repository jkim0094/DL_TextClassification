# DL_TextClassification

**Text Classification with Word Embeddings and Convolutional Neural Networks (CNNs)**

## Project Overview

This project focuses on sentence-level text classification using a question dataset labeled into six coarse-grained categories:

* ABBR (abbreviation)
* ENTY (entity)
* DESC (description)
* HUM (human)
* LOC (location)
* NUM (numeric)

## Part 1: Word Embeddings and Logistic Regression

* Download and use the pretrained GloVe model (`glove-wiki-gigaword-100`) to convert each word into a 100-dimensional vector.
* Construct sentence-level vectors using either average pooling or weighted average pooling via softmax-normalized importance scores.
* Normalize sentence vectors using `MinMaxScaler` to fit the range (-1, 1).
* Train a Logistic Regression model using the transformed training data and evaluate on a held-out validation set.

This part emphasizes text preprocessing and traditional machine learning classification using word embeddings.

## Part 2: Deep Learning with TextCNN

* Implement a TextCNN architecture based on the original paper, designed for sentence classification.

  * Use three Conv1D layers with kernel sizes 3, 5, and 7.
  * Apply Global Max Pooling to the output of each Conv1D.
  * Concatenate pooled outputs and pass through a dense layer for classification.
* The model uses learnable word embeddings, processed via a `TextDataManager`.

This part focuses on building and training a deep learning model (CNN) for text classification from scratch.

---

## Part 3: RNNs and Attention-based Text Classification

* Implement and compare multiple RNN variants: Vanilla RNN, GRU, and LSTM

  * Support for multiple hidden layers and output types (`last`, `mean`, `max`)
* Utilize pretrained GloVe embeddings with three initialization modes:

  * `scratch`, `init-only` (frozen), `init-fine-tune` (trainable)
* Implement an attention mechanism on the final RNN layer:

  * Compute alignment scores and generate a context vector for classification

This part focuses on capturing long-term dependencies and improving sequence representation through attention.

---

## Part 4: Transformer and Prefix-Tuning for Text Classification

* Build a `TransformerClassifier` using stacked encoder layers with mean pooling

  * Final classification is based on averaged token embeddings
* Implement `Prefix Tuning` on top of a pretrained BERT model

  * Add learnable prefix embeddings while freezing the BERT backbone
* Perform hyperparameter tuning and report the best-performing model

This part leverages state-of-the-art transformer-based techniques for efficient and effective fine-tuning.


