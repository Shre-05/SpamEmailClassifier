# 📧 Spam Email Classifier

A beginner-friendly Machine Learning project to classify SMS or email messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and the **Naive Bayes algorithm**.

---

## 📌 Project Description

This project uses the **SMS Spam Collection dataset** to build a spam classifier. It leverages TF-IDF for converting text into numerical features and Multinomial Naive Bayes for classification. You can easily use it for learning NLP basics and building real-world applications.

---

## 🧠 Technologies Used

- **Python**
- **Pandas** – Data manipulation
- **Scikit-learn** – Model training and evaluation
- **TfidfVectorizer** – Convert text into numerical format

---

## 📂 Dataset

- **Source**: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Fields**:
  - `v1`: Label (`ham` or `spam`)
  - `v2`: Message content

Rename the columns to `label` and `message` for clarity.

---

## 🚀 How to Run the Code

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
