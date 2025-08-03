# Data-Science
A complete A-Z notebook for data science 


# RoadMap

## 📍 Phase 0: Prerequisites (1 Month) {#phase-0}

### ✅ Math Essentials (Only what's needed)
- **Linear Algebra**
  - Vectors and Matrices
  - Matrix Multiplication
  - Eigenvalues and Eigenvectors
  - **Links**
    - [Linear Algebra Playlist](https://www.youtube.com/watch?v=RlHmflqeH3s&list=PLdKd-j64gDcBLV-vG6C0l6rxYQ0eLu2Zj&index=4&ab_channel=AnalyticsVidhya)
    - [3Blue 1 Brown - Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=1&ab_channel=3Blue1Brown)
    - [Linear Algebra Code](Linear%20Algebra/)
    
- **Calculus**
  - Derivatives and Integrals
  - Chain Rule and Partial Derivatives
  - **Links**
    - [3Blue 1 Brown - Calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr&ab_channel=3Blue1Brown)
  
- **Probability & Statistics**
  - Mean, Median, Mode, Variance, Standard Deviation
  - Probability Theory and Bayes' Theorem
  - Distributions (Normal, Binomial, Poisson)
  - Hypothesis testing
  - **Links**
    - [StatsQuest - Fundamentals of Statistics](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9&ab_channel=StatQuestwithJoshStarmer)

### 💻 Programming (Python)
- Python Basics: Data Types, Loops, Functions, OOP
- File and Error Handling
- Important Libraries:
    - `NumPy`, `Pandas` – data manipulation
    - `Matplotlib`, `Seaborn`, `Plotly` – visualization
    - `scikit-learn` – core ML package

---
## 🔄 Core Data Science Life Cycle

### 1️⃣ Problem Formulation
- **Question Definition**
  - What are the questions we are aiming to answer?
  - Define success metrics and objectives
- **Challenge Identification**
  - What are the challenges that need to be addressed?
  - Identify constraints and limitations
  - Stakeholder requirements analysis

### 2️⃣ Data Preparation/ Preprocessing / Handling → [Phase 1](#phase-1)
- Data Collection
- Data Cleaning & Curation

### 3️⃣ Exploratory Data Analysis (EDA) → [Phase 1](#phase-1)
- Descriptive Statistics such as mean, median, mode
- Data Visualization
- Basic Statistical Modeling like correlation analysis etc.
- **Note**: In real life EDA is an iterative process. It is done before data preprocessing to identify inconsistencies and patterns. And after data preprocessing to understand the cleaned data.

### 4️⃣ Data Modeling → [Phase 2](#phase-2), [Phase 3](#phase-3), [Phase 4](#phase-4)
- Algorithm Selection
- Model Development

### 5️⃣ Model Evaluation & Testing → [Phase 2](#phase-2), [Phase 3](#phase-3), [Phase 4](#phase-4)
- Validation Strategies
- Performance Assessment
- Testing & Validation
  
### 6️⃣ Deployment & Maintenance → [Phase 5](#phase-5)
- Deployment Planning
- Continuous Monitoring
- Maintenance & Updates


---
<a id="phase-1"></a>
## 📍 Phase 1: Core Data Science (1–1.5 Months)

### Data Fetching & Preparation
- Data Collection: APIs, Web Scraping, Databases
  - ** Links**
    - [Medium - Data Collection](https://ianclemence.medium.com/day-7-data-collection-methods-apis-web-scraping-and-databases-2db1064741c1)
- Data Ingestion: Batch vs Real-time
  - **Links**
    - [Datacamp - Data Ingestion](https://www.datacamp.com/blog/batch-vs-stream-processing)

### ✅ Data Handling
- Cleaning missing/null data
- Handling outliers and duplicates
- Feature Engineering and Transformation
- Grouping, Merging, Reshaping
- **Links**
  - [Medium - Data Cleaning](https://medium.com/pythoneers/practical-examples-of-data-cleaning-using-pandas-and-numpy-5f59021f0144)
  - [Datacamp - Feature Engineering](https://www.datacamp.com/tutorial/feature-engineering)
  - [Data cleaning practices](Data%20Cleaning.md/)

### ✅ EDA (Exploratory Data Analysis)

#### 🎯 Visualization Types & Use Cases

**Distribution Analysis:**
- **Histograms** - Show frequency distribution of numerical variables
- **Box plots** - Identify outliers, quartiles, and data spread
- **Violin plots** - Combine box plots with kernel density estimation
- **Density plots** - Smooth distribution curves

**Relationship Analysis:**
- **Scatter plots** - Explore relationships between two continuous variables
- **Pair plots** - Matrix of scatter plots for multiple variables
- **Heatmaps** - Visualize **correlation matrices** and pivot tables
- **Joint plots** - Combine scatter plots with marginal distributions

**Categorical Analysis:**
- **Bar plots** - Compare categories and frequencies
- **Count plots** - Show frequency of categorical variables
- **Pie charts** - Show proportions (use sparingly)

**Time Series Analysis:**
- **Line plots** - Show trends over time
- **Area plots** - Stacked trends visualization
- Distributions and skewness

**Links**
  - [Kaggle - EDA Visualization](https://www.kaggle.com/code/robikscube/introduction-to-exploratory-data-analysis)
  - [Matplotlib - Plot types](https://matplotlib.org/stable/plot_types/index.html)

**Distribution & Skewness Analysis:**
- **Distribution Types**
  - Normal, uniform, skewed, bimodal distributions
  - Heavy-tailed vs light-tailed distributions
- **Skewness Measurement**
  - Right skewed (positive): mean > median, long right tail
  - Left skewed (negative): mean < median, long left tail
  - Symmetric: mean ≈ median
  - **Skewness values**: -0.5 to 0.5 (symmetric), ±0.5 to ±1 (moderate), >±1 (high)
- **Normality Testing**
  - Shapiro-Wilk test (small samples)
  - Kolmogorov-Smirnov test (larger samples)
  - Visual inspection with Q-Q plots

**NoteBook:** : [EDA Notebook](EDA_Visualization_Guide.ipynb)

---
<a id="phase-2"></a>
## 📍 Phase 2: Machine Learning Models (2 Months)

### 🤖 ML Fundamentals
- Types of Learning: Supervised, Unsupervised, Reinforcement, Semi-supervised, Self-supervised, Advanced: Transfer Learning, Active Learning
- Model Lifecycle: Train/Test Split, Validation
- Concepts: Overfitting, Underfitting, Bias-Variance Tradeoff

### ✅ Supervised Learning
- Linear Regression
- Logistic Regression
- Decision Trees, Random Forests
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting (XGBoost, LightGBM)

### ✅ Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- PCA for Dimensionality Reduction

### ✅ Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression: MAE, RMSE
- Cross-validation techniques

### 🔬 Feature Engineering (Advanced)
- Encoding Categorical Features
- Feature Scaling: StandardScaler, MinMaxScaler
- Feature Selection Techniques
- Handling Imbalanced Data: SMOTE, Class Weights

### ✅ Model Selection & Tuning (Advanced)
- Hyperparameter tuning: `GridSearchCV`, `RandomizedSearchCV`
- Train/Validation/Test Split

### 📚 Libraries
- `scikit-learn`
- `xgboost`
- `lightgbm`


---
<a id="phase-3"></a>
## 📍 Phase 3: Deep Learning (2 Months)

### ✅ Core Concepts
- Neurons, Perceptrons
- Neural Networks: input, hidden, output layers
- Activation Functions: ReLU, Sigmoid, Tanh
- Forward and Backward Propagation
- Loss Functions: MSE, Cross Entropy
- Optimizers: SGD, Adam

### 🧠 Neural Networks
- Feedforward Neural Networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- LSTM, GRU

### ✅ Frameworks
- `TensorFlow + Keras` OR `PyTorch`
- `torchvision`, `datasets`

---

<a id="phase-4"></a>
## 📍 Phase 4: NLP for Production (1–1.5 Months)

### ✅ Classical NLP
- Text Preprocessing: Lemmatization, tokenization, stopwords, POS Tagging
- TF-IDF, Bag of Words
- Naive Bayes, Logistic Regression on text
- Word Embeddings: Word2Vec, GloVe

### ⚡ Transformers
- Attention Mechanism
- Transformers Architecture
- Models: BERT, GPT, T5
- HuggingFace Transformers Library
    - Sentiment Analysis
    - Text Classification
    - NER (Named Entity Recognition)

---
<a id="phase-5"></a>
## 📍 Phase 5: Deployment & MLOps (1.5 Months)

### ✅ Model Deployment
- Save & Load Models: `pickle`, `joblib`
- Create REST APIs with `Flask` / `FastAPI`
- Web Apps using `Streamlit`, `Gradio`

### ✅ Docker & Cloud
- Docker basics: containerize and run models
- AWS EC2 + S3: cloud deployment
- (Optional) GCP or Azure

### ✅ MLOps Basics
- Git + GitHub for version control
- CI/CD with GitHub Actions
- `MLflow` for model tracking


---
<a id="phase-6"></a>
## 📍 Phase 6: Specialization (Optional / Parallel)

### 🔍 Areas of Specialization
- Computer Vision (YOLO, SSD, OpenCV)
- Time Series Forecasting (ARIMA, LSTM, Prophet)
- Reinforcement Learning (Q-Learning, DQN)
- Graph Neural Networks
- Generative AI (GANs, VAEs, Diffusion Models)

---
