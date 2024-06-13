# Sentiment Analysis Project

## Project Overview

This repository contains the code and report for a sentiment analysis project conducted by Group 5 (LI Yanjia, SONG Wenxin, Tam Wui Wo, and Yeung Ngo Yan) as part of COMP 4332. The objective of the project is to design a model that predicts comment scores given by reviewers based on the features of the comment.

## Table of Contents

- [Objective](#objective)
- [Data Exploration](#data-exploration)
- [Data Pipeline Design](#data-pipeline-design)
  - [Model Design](#model-design)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training/Fine-tuning](#model-trainingfine-tuning)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

## Objective

The objective of the project is to predict the comment scores based on the features of the comments. The main focus is on using text information to achieve this goal.

## Data Exploration

### Numerical Variables

The numerical variables in the dataset are `cool`, `funny`, and `useful`, which represent reactions given to reviews by other users. Their correlation with review stars was analyzed and found to be minimal.

## Data Pipeline Design

### Model Design

Due to the minimal influence of numerical attributes on review stars, the focus was shifted to text information. The BERT-class models (BERT-base-uncased and RoBERTa) were fine-tuned on the training set. RoBERTa was chosen for its improved accuracy.

**BERT-class Models**

BERT is a state-of-the-art bi-directional transformer that excels in language representation. Unlike conventional models like LSTM, BERT uses attention mechanisms and transformers to learn from all positions of a sentence simultaneously. RoBERTa, an improvement over BERT, employs better training methodology, a larger dataset, and more compute power.

### Data Preprocessing

The preprocessing involved:
- Removing special symbols (e.g., `#`, `\n`, `\r`).
- Converting text to lowercase.
- Using a pre-trained RoBERTa tokenizer to encode text and convert strings to sequences of IDs.
- Fixing sentence lengths by truncating longer sentences and padding shorter ones.
- Creating attention masks to differentiate real words from padding words.
- Storing preprocessed text and labels in PyTorch DataLoader before training.

### Model Training/Fine-tuning

The model was fine-tuned by adding a dense layer with softmax activation to the pre-trained RoBERTa model. The training utilized the Adam optimizer and a linear scheduler to optimize the model, with the learning rate gradually reduced during training.

**Hyper-parameter Tuning:**
1. Learning rate: 1.5e-5
2. Number of epochs: 3
3. Batch size: 32
4. Max sentence length: 256

**Results:**
- Training duration: 20 minutes (on GPU in Google Colab)
- Training loss: Decreased from 1.64 to 0.73
- Training accuracy: Increased from 0.25 to 0.739
- Validation accuracy: Achieved 68.7% after 3 epochs

## Discussion

Pre-processing techniques like stemming and stopword removal were found to be less effective as they remove contextual information essential for BERT-class models. The optimal max sentence length for RoBERTa was empirically determined to be 256, covering almost 90% of the training samples.

## Conclusion

The final RoBERTa model achieved an accuracy of 68.7% on the validation set, demonstrating its effectiveness in sentiment analysis.

## References

1. Singh, A. (2020, January 15). Building State-of-the-Art Language Models with BERT. Retrieved from [Medium](https://medium.com/saarthi-ai/bert-how-to-build-state-of-the-art-language-models-59dddfa9ac5d)

