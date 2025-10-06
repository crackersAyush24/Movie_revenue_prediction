
# Movie Revenue Prediction by Ayush Chaubey

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-grey.svg?logo=scikit-learn)](https://scikit-learn.org/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-brightgreen.svg)](https://github.com/crackersAyush24/Movie_revenue_prediction)

---

## Introduction

I created this **Movie Revenue Prediction** project to explore how machine learning can forecast box office earnings based on a movie’s features. The goal is to help filmmakers, producers, and enthusiasts get a better understanding of which factors influence revenue the most.

The system predicts movie revenue using features such as:

* Movie Name, Genre, Director, Writer, Leading Cast
* Year of Release, IMDb Score, Votes
* Country of Production, Production Company
* Budget, Runtime

I enhanced the dataset and created new features that capture relationships between budget, votes, ratings, and runtime. These engineered features help the model identify patterns in movie success more effectively.

The project also includes a **Streamlit web interface**, allowing users to enter movie details and receive predicted revenue ranges instantly.

---

## Getting Started

### Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/crackersAyush24/Movie_revenue_prediction.git
cd Movie_revenue_prediction
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv env
source env/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

### Running Models

Train and test the machine learning models using:

```bash
python models/<model_name>.py
```

Available models: `linear_regression`, `random_forest`, `gradient_boost`, `XGBoost`.

---

### Streamlit Web App

Use the interactive web interface to predict movie revenue:

```bash
streamlit run streamlit_app.py
```

Enter movie details and select a model to get revenue predictions in real-time.

---

### Data Preprocessing

* Handled missing values using median imputation.
* Encoded categorical variables using Label Encoding.
* Scaled numerical features with StandardScaler.
* Applied log transformations to skewed features like budget and revenue.

#### Engineered Features

* Ratios like `vote_score_ratio`, `budget_year_ratio`, `score_runtime_ratio`
* Binary features like `is_recent`, `is_high_budget`, `is_high_votes`, `is_high_score`

These features help the model understand complex patterns in the dataset, improving prediction quality.

---

### Revenue Prediction Ranges

Predicted movie revenue is categorized as:

* Low: ≤ $10M
* Medium-Low: $10M–$40M
* Medium: $40M–$70M
* Medium-High: $70M–$120M
* High: $120M–$200M
* Ultra High: ≥ $200M

---

## Contact

* GitHub: [crackersAyush24](https://github.com/crackersAyush24)
* Email: [chaubeyayush04@gmail.com](mailto:chaubeyayush04@gmail.com)

---

## License
(c) 2025 Ayush Chaubey


