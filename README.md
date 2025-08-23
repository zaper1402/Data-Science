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
  - Standardization vs Normalisation
  - [Scaling](https://ethans.co.in/blogs/different-types-of-feature-scaling-and-its-usage/)
- Grouping, Merging, Reshaping
- **Links**
  - [Medium - Data Cleaning](https://medium.com/pythoneers/practical-examples-of-data-cleaning-using-pandas-and-numpy-5f59021f0144)
  - [Datacamp - Feature Engineering](https://www.datacamp.com/tutorial/feature-engineering)
  - [Data cleaning practices](Data%20Cleaning.md/)

### ✅ EDA (Exploratory Data Analysis)

#### 🎯 Visualization Types & Use Cases

**Distribution Analysis:**
- **Histograms** - Show frequency distribution of numerical variables
- [**Box plots**](https://www.youtube.com/results?search_query=statsquid+violin+plot) - Identify outliers, quartiles, and data spread
- [**Violin plots**](https://www.youtube.com/results?search_query=statsquid+violin+plot) - Combine box plots with kernel density estimation
- [**Density plots**](https://www.youtube.com/watch?v=CbqoxkkJyzY&ab_channel=Biostatsquid) - Smooth distribution curves

**Relationship Analysis:**
- **Scatter plots** - Explore relationships between two continuous variables
- **Pair plots** - Matrix of scatter plots for multiple variables
- **Heatmaps** - Visualize **correlation matrices** and pivot tables
- [**Joint plots**](https://www.youtube.com/watch?v=56AQl9L6V8A&ab_channel=Learnerea) - Combine scatter plots with marginal distributions

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
- [**Skewness Measurement**](https://www.youtube.com/watch?v=U0NZu6f5TMI&t=308s&ab_channel=TheOrganicChemistryTutor)
  - Right skewed (positive): mean > median, long right tail
  - Left skewed (negative): mean < median, long left tail
  - Symmetric: mean ≈ median
  - **Skewness values**: -0.5 to 0.5 (symmetric), ±0.5 to ±1 (moderate), >±1 (high)
- **Normality Testing**
  - Shapiro-Wilk test (small samples)
  - Kolmogorov-Smirnov test (larger samples)
  - Visual inspection with [Q-Q plots](https://www.youtube.com/watch?v=okjYjClSjOg&ab_channel=StatQuestwithJoshStarmer)

**NoteBook:** : [EDA Notebook](EDA_Visualization_Guide.ipynb)

---
<a id="phase-2"></a>
## 📍 Phase 2: Machine Learning Models (2 Months)

### 🤖 ML Fundamentals
- Types of Learning: Supervised, Unsupervised, Reinforcement, Semi-supervised, Self-supervised, Advanced: Transfer Learning, Active Learning [Flowchart](full_machine_learning_flowchart.html/)
- Model Lifecycle: Train/Test Split, Validation
- Concepts: [Overfitting, Underfitting, Bias-Variance Tradeoff](https://www.youtube.com/watch?v=EuBBz3bI-aA&ab_channel=StatQuestwithJoshStarmer)

### ✅ Supervised Learning

#### 🔍 Classification Algorithms
- [Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe&ab_channel=StatQuestwithJoshStarmer)
  - [IBM Document](https://www.ibm.com/think/topics/logistic-regression)
  - [CampusX In-depth (Hindi)](https://www.youtube.com/watch?v=XNXzVfItWGY&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=69&ab_channel=CampusX)
- [Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
  - [Gaussian Naive Bayes](https://www.youtube.com/watch?v=H3EjCKtlVog&pp=0gcJCccJAYcqIYzv)   
  - [CampusX in-depth (Hindi)](https://www.youtube.com/watch?v=Ty7knppVo9E&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=118&ab_channel=CampusX)     

#### 📈 Regression Algorithms
- [Linear Regression](https://www.youtube.com/watch?v=7ArmBVF2dCs&ab_channel=StatQuestwithJoshStarmer)
  - [CampusX in-depth (Hindi)](https://www.youtube.com/watch?v=UZPfbG0jNec&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=50&ab_channel=CampusX)   
- [Polynomial Regression](https://www.youtube.com/watch?v=BNWLf3cKdbQ&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=60&ab_channel=CampusX)
- [Ridge/Lasso/Elastic Net Regression](https://www.youtube.com/watch?v=Q81RR3yKn30&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=25&ab_channel=StatQuestwithJoshStarmer)
- [Gaussian Process Regression](https://www.youtube.com/watch?v=UBDgSHPxVME&list=PL1iHuxEW9u9iz1NVUKwH8VRCpj3xB5NMY&ab_channel=MutualInformation)

#### 🎯 Both Regression and Classification
- [Decision Trees](https://youtu.be/_L39rN6gz7Y?feature=shared)
- [Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&list=PLblh5JKOoLUIE96dI3U7oxHaCAbZgfhHk&ab_channel=StatQuestwithJoshStarmer)
- [Boosting Algorithm](https://aws.amazon.com/what-is/boosting/#:~:text=Boosting%20algorithms%20combine%20multiple%20weak,common%20in%20machine%20learning%20models.)
  - [AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA&ab_channel=StatQuestwithJoshStarmer)
  - [Gradient Boosting](https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6&ab_channel=StatQuestwithJoshStarmer)
  - [XGBoost](https://www.youtube.com/watch?v=OtD8wVaFm6E&list=PLblh5JKOoLULU0irPgs1SnKO6wqVjKUsQ&ab_channel=StatQuestwithJoshStarmer)
  - LightGBM
  - CatBoost
- [K-Nearest Neighbors](https://www.youtube.com/watch?v=b6uHw7QW_n4&ab_channel=IBMTechnology)
- [Support Vector Machines (SVM)](https://www.youtube.com/watch?v=efR1C6CvhmE&list=PLblh5JKOoLUL3IJ4-yor0HzkqDQ3JmJkc&ab_channel=StatQuestwithJoshStarmer)

### ✅ [Unsupervised Learning](https://cloud.google.com/discover/what-is-unsupervised-learning)

#### 🎯 [Clustering Algorithms](https://developers.google.com/machine-learning/clustering/overview)
- [K-Means](https://www.youtube.com/watch?v=4b5d3muPQmA&t=33s&ab_channel=StatQuestwithJoshStarmer)
- [Hierarchical Clustering](https://www.youtube.com/watch?v=8QCBl-xdeZI&ab_channel=DATAtab)
- [DBSCAN](https://www.youtube.com/watch?v=RDZUdRSDOok&ab_channel=StatQuestwithJoshStarmer)

#### 📊 Dimensionality Reduction
- [Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=FgakZw6K1QQ&ab_channel=StatQuestwithJoshStarmer)
  - [Visualisation of PCA](https://www.youtube.com/watch?v=FD4DeN81ODY&ab_channel=VisuallyExplained) 
- [Linear Discriminant Analysis (LDA)](https://www.youtube.com/watch?v=azXCzI57Yfc&ab_channel=StatQuestwithJoshStarmer)
- [t-SNE](https://www.youtube.com/watch?v=NEaUSP4YerM&ab_channel=StatQuestwithJoshStarmer)
- [UMAP](https://www.youtube.com/watch?v=eN0wFzBA4Sc&ab_channel=StatQuestwithJoshStarmer)
- [PCA vs t-SNE vs UMAP](https://www.youtube.com/watch?v=o_cAOa5fMhE&ab_channel=Deepia)

#### 🎨 [Association Rules](https://www.youtube.com/watch?v=guVvtZ7ZClw&t=120s&ab_channel=edureka%21)

### ✅ Evaluation Metrics

#### 🎯 [Classification Metrics](https://developers.google.com/machine-learning/crash-course/classification/thresholding)
- [**Confusion Matrix**](https://www.youtube.com/watch?v=Kdsp6soqA7o&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=3&ab_channel=StatQuestwithJoshStarmer): True/False Positives and Negatives
- **Basic Metrics**: [Accuracy, Precision, Recall, F1-Score](https://www.youtube.com/watch?v=4i4C3ejTdgs&ab_channel=ML%26DLExplained)
- **Medical/Binary Classification**: [Sensitivity (True Positive Rate), Specificity (True Negative Rate)](https://www.youtube.com/watch?v=vP06aMoz4v8&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=4&ab_channel=StatQuestwithJoshStarmer)
- [**ROC Curve & AUC**](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

#### 📊 Regression Metrics
- **Error Metrics**: [MAE (Mean Absolute Error)](https://youtu.be/Ti7c-Hz7GSM?t=92&feature=shared), [MSE](https://youtu.be/Ti7c-Hz7GSM?t=92&feature=shared) , [RMSE (Root Mean Square Error)](https://youtu.be/Ti7c-Hz7GSM?t=749&feature=shared)
- **Relative Metrics**: [R²](https://www.youtube.com/watch?v=bMccdk8EdGo&ab_channel=StatQuestwithJoshStarmer), [Adjusted R² Score](https://youtu.be/Ti7c-Hz7GSM?t=1626&feature=shared)

#### ⚖️ [Model Bias & Fairness](https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias)
- [Bias-Variance Tradeoff](https://www.youtube.com/watch?v=EuBBz3bI-aA&ab_channel=StatQuestwithJoshStarmer)
- [Fairness Measures](https://www.youtube.com/watch?v=3UcSq1dGW2c&ab_channel=ADataOdyssey)

#### 🔄 Cross-Validation Techniques
-  **K-Fold Cross-Validation**: Standard approach for model validation
- **Stratified Cross-Validation**: Maintaining class distribution
- **Leave-One-Out Cross-Validation**: For small datasets
  - [Video for K-Fold, Stratified, Leave-One-Out](https://www.youtube.com/watch?v=PF2wLKv2lsI&t=36s&ab_channel=MaheshHuddar)
- [**Time Series Cross-Validation**](https://www.youtube.com/watch?v=1rZpbvSI26c&ab_channel=EgorHowell): Walk-forward validation

### ✅ Model Selection & Tuning (Advanced)
- [Hyperparameter tuning](https://www.youtube.com/watch?v=lfiw2Rh2v8k&ab_channel=AIForBeginners): `GridSearchCV`, `RandomizedSearchCV`
- Train/Validation/Test Split

### 📚 Some useful Libraries
- `scikit-learn`
- `xgboost`
- `lightgbm`

**Links**
- [Stanford CS229 - Machine Learning](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&ab_channel=StanfordOnline)
- [Google Machine Learning Course](https://developers.google.com/machine-learning)

**CheatSheet* : [Scikit-learn CheatSheet](https://scikit-learn.org/stable/machine_learning_map.html)


---
<a id="phase-3"></a>
## 📍 Phase 3: Deep Learning (2 Months)

### ✅ Core Concepts

- Neurons, Perceptrons
  - [Medium Blog](https://medium.com/@abhishekjainindore24/perceptron-vs-neuron-single-layer-perceptron-and-multi-layer-perceptron-68ce4e8db5ea)
  - [Video](https://www.youtube.com/watch?v=OFbnpY_k7js&list=PL2zRqk16wsdo3VJmrusPU6xXHk37RuKzi&index=2&pp=iAQB)
- [Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown): input, hidden, output layers
- [Activation Functions](https://www.youtube.com/watch?v=Y9qdKsOHRjA&ab_channel=LearnWithJay): ReLU, Sigmoid, Tanh, SoftMax
  - [Blog](https://www.v7labs.com/blog/neural-networks-activation-functions)   
- [Gr adient Descent, Cost Function](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2&pp=iAQB)
- [Forward](https://www.youtube.com/watch?v=99CcviQchd8&ab_channel=SatyajitPattnaik) and [Backward](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3&pp=iAQB0gcJCa0JAYcqIYzv) Propagation
- [Loss Functions](https://nadeemm.medium.com/cost-function-loss-function-c3cab1ddffa4): MSE, Cross Entropy
- [Optimizers](https://www.youtube.com/watch?v=mdKjMPmcWjY&ab_channel=CodeEmporium): SGD, Adam

### 🧠 Neural Networks
- [Feedforward Neural Networks](https://www.youtube.com/watch?v=QK7GJZ94qPw&ab_channel=NatalieParde)
  - [Multilayer Perceptron (MLP)](https://www.youtube.com/watch?v=7YaqzpitBXw&ab_channel=IBMTechnology)
  - [Autoencoders](https://www.youtube.com/watch?v=hZ4a4NgM3u0&ab_channel=Deepia)
    - [Undercomplete, Overcomplete](https://medium.com/@piyushkashyap045/a-comprehensive-guide-to-autoencoders-8b18b58c2ea6)
    - [Variational Autoencoders (VAE)](https://www.youtube.com/watch?v=qJeaCHQ1k2w&ab_channel=Deepia)
- [Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=pj9-rr1wDhM&ab_channel=Futurology%E2%80%94AnOptimisticFuture)
  - [MIT- Lecture](https://www.youtube.com/watch?v=oGpzWAlP5p0&ab_channel=AlexanderAmini)
  - [Visualisation tool](https://adamharley.com/nn_vis/)
- [Recurrent Neural Networks (RNNs)](https://www.youtube.com/watch?v=AsNTP8Kwu80&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=15&pp=iAQB)
  - [Long Short-Term Memory (LSTM)](https://www.youtube.com/watch?v=YCzL96nL7j0&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=16&ab_channel=StatQuestwithJoshStarmer): RNN using gates, used for time-series forecasting
  - [Gated Recurrent Unit (GRU)](https://www.youtube.com/watch?v=tOuXgORsXJ4&ab_channel=codebasics)
- Generative Adversarial Networks (GANs)

- Links:
  - [Google: Course](https://developers.google.com/machine-learning/crash-course/neural-networks)   


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
