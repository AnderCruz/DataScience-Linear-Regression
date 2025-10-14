# Real Estate Price Prediction with Linear Regression

## Project Overview

This project focuses on building, evaluating, and interpreting a Linear Regression model to predict the price of real estate properties. The entire machine learning workflow is implemented in a Jupyter Notebook, covering data loading, cleaning, exploratory analysis, model training, and performance evaluation.

The analysis begins by importing a dataset of house sales, followed by data preprocessing, feature engineering, and extensive exploratory data analysis (EDA) to understand feature relationships. A multivariate Linear Regression model is then trained, and its performance is rigorously evaluated using statistical metrics and visualizations. The project culminates in using the trained model to predict the price of a sample property.

## Dataset

The project uses a real estate dataset sourced from a public `.csv` file containing information on house sales.

  - **Source URL:** `https://raw.githubusercontent.com/alura-cursos/statistics-basics/master/real_estate_data.csv`

The dataset includes the following features:

  - `price`: The sale price of the house (Target Variable).
  - `bedrooms`: Number of bedrooms.
  - `bathrooms`: Number of bathrooms.
  - `sqft_living`: Living area in square feet.
  - `sqft_lot`: Lot size in square feet.
  - `floors`: Number of floors.
  - `waterfront`: A binary indicator for whether the house has a waterfront view.
  - `view`: A rating of the view quality.
  - `condition`: A rating of the house's condition.
  - `grade`: An index from 1 to 13 representing the quality of construction and design.
  - `sqft_above`: Square footage of the house above ground level.
  - `sqft_basement`: Square footage of the basement.
  - `yr_built`: The year the house was built.
  - `yr_renovated`: The year the house was renovated.
  - `zipcode`: The zip code of the property's location.
  - `lat`: Latitude coordinate.
  - `long`: Longitude coordinate.


### Analysis and Modeling Workflow

The notebook is structured to follow a comprehensive machine learning pipeline.

#### 1\. Data Loading and Initial Inspection

  - **Data Ingestion**: The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`.
  - **Initial Exploration**: The first few rows are inspected with `df.head()`, and `df.info()` is used to get a summary of data types and check for missing values.
  - **Duplicate Check**: The notebook verifies the absence of duplicate rows using `df.duplicated().sum()`.

#### 2\. Data Cleaning and Preprocessing

  - **Handling Missing Values**: The code confirms there are no null values using `df.isnull().sum()`.
  - **Feature Engineering**: A new feature, `usable_area_percentage`, is created to represent the ratio of living space to the total lot size. This helps in capturing the efficiency of land use.
  - **Outlier Detection and Removal**:
      - Boxplots are generated using `seaborn.boxplot()` to visualize the distribution of key features (`price`, `bedrooms`, `bathrooms`) and identify potential outliers.
      - Based on the visualization, a property with an unusually high number of bedrooms (33) is identified as an outlier and removed from the dataset using `df.drop()`.

#### 3\. Exploratory Data Analysis (EDA) and Visualization

EDA is performed to uncover patterns and relationships between features and the target variable (`price`).

  - **Histograms**: The distributions of numerical features are visualized using histograms (`df.hist()`) to understand their shape and spread.
  - **Correlation Analysis**:
      - A correlation matrix is computed using `df.corr()`.
      - A **heatmap** is generated with `seaborn.heatmap()` to visually represent the strength and direction of linear relationships. This is critical for selecting predictive features. Features like `sqft_living`, `grade`, and `sqft_above` are shown to have a strong positive correlation with `price`.
  - **Scatter Plots**: The relationship between `sqft_living` and `price` is examined with a scatter plot, confirming a strong positive linear trend.
  - **Boxplots by Category**: `seaborn.boxplot()` is used to analyze how `price` varies across different categories of features like `bedrooms`, `bathrooms`, `floors`, and `waterfront`.

#### 4\. Feature Selection and Model Preparation

  - **Feature Selection**: Based on the insights from the correlation analysis, a set of highly correlated and relevant features is selected for the model: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, and `waterfront`.
  - **Defining Variables**: The selected features are assigned to the independent variable `X`, and the target variable `price` is assigned to `y`.
  - **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets using the `train_test_split` function from Scikit-learn to ensure the model can be evaluated on unseen data.

#### 5\. Model Training

  - **Linear Regression**: A `LinearRegression` model is instantiated from the Scikit-learn library.
  - **Fitting the Model**: The model is trained on the training data (`X_train`, `y_train`) using the `.fit()` method.

#### 6\. Model Evaluation

The model's predictive performance is assessed on the test set.

  - **R-squared (R²)**: The coefficient of determination is calculated using `model.score(X_test, y_test)` to measure how much of the variance in the house prices is explained by the model.
  - **Error Metrics**: The `mean_absolute_error` (MAE) and `mean_squared_error` (MSE) are calculated using `sklearn.metrics` to quantify the average magnitude of the prediction errors.
  - **Visualization of Predictions**: A scatter plot is created to compare the model's predictions (`y_pred`) against the actual prices (`y_test`). A line of perfect fit (y=x) is overlaid to visually assess the model's accuracy.

#### 7\. Making a Prediction

  - The trained model is used to predict the price of a new, sample property. A DataFrame is created for this single entry, and the `model.predict()` method is called to generate the price estimate.


### Technologies and Libraries Used

  - **Language**: **Python 3**
  - **Core Libraries for Data Science**:
      - **Pandas**: For data loading, manipulation, cleaning, and analysis.
      - **NumPy**: For numerical operations, often used implicitly by Pandas.
  - **Visualization Libraries**:
      - **Matplotlib**: The primary library for creating static, annotated, and interactive visualizations like histograms and scatter plots.
      - **Seaborn**: A high-level interface built on Matplotlib for creating attractive and informative statistical graphics, such as boxplots and heatmaps.
  - **Machine Learning Library**:
      - **Scikit-learn**: Used for splitting the data (`train_test_split`), implementing the `LinearRegression` model, and evaluating its performance with metrics like `r2_score`, `mean_absolute_error`, and `mean_squared_error`.
  - **Environment**: **Jupyter Notebook**, an interactive environment for writing and executing code, visualizing data, and documenting the entire analysis process.

### How to Run This Project

1.  Clone this repository to your local machine.
2.  Ensure you have Python and the required libraries installed. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  Open the `Projeto Final - Regressão Linear.ipynb` file in a Jupyter Notebook environment.
4.  Run the cells sequentially to reproduce the data analysis, model training, and evaluation steps.
