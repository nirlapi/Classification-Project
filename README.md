# Titanic: Machine Learning from Disaster

**Authors:** Shiri Guniman & Nir Lapidot  
**Kaggle Profile:** [nirlapidot](https://www.kaggle.com/nirlapidot)

## 📌 Project Overview
This project is a machine learning binary classification task developed for the Kaggle competition **[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)**. The goal of this project is to accurately predict whether a passenger survived the Titanic sinking based on various features such as age, sex, and ticket class.

## 📊 Dataset
The project utilizes the standard Kaggle competition datasets:
* `train.csv`: Used for training and validating the model, includes the target variable (`Survived`).
* `test.csv`: Used for the final Kaggle submission, does not include the target variable.

## 🛠️ Data Preprocessing & EDA
We performed Exploratory Data Analysis (EDA) and data cleaning to prepare the dataset for our machine learning model:
* **Initial Exploration:** Utilized statistical summaries (`describe()`) and integrated automated visual tools like `AutoViz` to gain preliminary insights into data distributions and identify outliers.
* **Handling Missing Values:** Applied strategic imputation for missing values, such as filling missing values in the `Embarked` column with the mode.
* **Feature Engineering & Encoding:** * Utilized `Scikit-Learn` pipelines for clean transformations.
  * Scaled numerical columns using `MinMaxScaler` followed by `StandardScaler`.
  * Used `OrdinalEncoder` for categorical variables exclusively during the feature selection phase to prevent splitting categories into multiple binary columns prematurely.
  * Applied `OneHotEncoder` on categorical variables *after* feature selection for the final model pipeline.
* **Feature Selection:** * Initially filtered the top features using `SelectKBest` with `mutual_info_classif`.
  * Applied a Wrapper method using `SequentialFeatureSelector` (forward selection) with a KNN classifier to select the most impactful subset of features. The final selected features were: `Sex`, `Embarked`, `Pclass`, and `Parch`.

## 🤖 Modeling & Machine Learning
For this binary classification task, we selected the **K-Nearest Neighbors (KNN)** algorithm.
* **Validation Strategy:** Implemented **Bootstrap Validation** and **Cross-Validation** to robustly evaluate model performance and prevent overfitting.
* **Hyperparameter Tuning:** Conducted a manual Grid Search to iterate over different values of `k` (neighbors), `weights` (uniform vs. distance), and `p` (Manhattan vs. Euclidean distance).
* **Final Model Evaluation:** We used tools like Confusion Matrices and ROC-AUC scores to compare results. After uploading predictions from multiple configurations, the model with **k = 5** achieved the highest score on the Kaggle leaderboard and was established as our optimal final model.

## 💻 Technologies & Libraries
* **Language:** Python
* **Data Manipulation:** NumPy, Pandas
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn, Plotly, AutoViz
* **Environment:** Google Colab / Jupyter Notebook

## 🚀 How to Run
1. Clone the repository and ensure the dataset files (`train.csv` and `test.csv`) are located in the same directory.
2. Install the required packages: `pip install pandas numpy scikit-learn matplotlib seaborn plotly autoviz cairosvg textblob`
3. Run the Jupyter Notebook `ML3_.ipynb` sequentially to execute the EDA, preprocessing, feature selection, model training, and prediction generation.
