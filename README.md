# AI for IoT Predictive Maintenance

## Project Overview

This project demonstrates a complete workflow for an AI-powered predictive maintenance solution, a critical application in the Internet of Things (IoT) industry. The goal is to predict imminent device failures based on time-series sensor data. 

By analyzing simulated data from an industrial device—including temperature, vibration, and rotation speed—this project builds a machine learning model capable of identifying patterns that precede a failure. This allows for proactive maintenance, which can save costs and prevent downtime in a real-world scenario.

This repository serves as a portfolio piece showcasing skills in data analysis, feature engineering, and machine learning for IoT applications.

## Methodology

1.  **Data Simulation:** A Python script (`simulate_data.py`) was created to generate a realistic time-series dataset. It produces normal operational data and injects anomalies (e.g., rising temperature, increased vibration) in the periods leading up to a failure event.
2.  **Exploratory Data Analysis (EDA):** The data was loaded into a Jupyter Notebook (`predictive_maintenance_analysis.ipynb`). Using `pandas`, `matplotlib`, and `seaborn`, the sensor readings were visualized to confirm the presence of clear patterns distinguishing normal operation from pre-failure states.
3.  **Feature Engineering:** To provide the model with historical context, rolling-window features (mean and standard deviation) were calculated for each sensor. These features help the model learn from trends over time rather than just single data points.
4.  **Model Training:** A `RandomForestClassifier` from `scikit-learn` was chosen for the classification task. This model is well-suited for this problem due to its ability to handle complex, non-linear relationships and provide insights into feature importance. The model was trained on a stratified training set to handle the class imbalance between failure and non-failure events.
5.  **Evaluation:** The model's performance was evaluated on an unseen test set. Key metrics such as Precision, Recall, and the F1-score were calculated, and a confusion matrix was generated to visualize the model's accuracy in correctly identifying failures.

## Technologies Used

*   **Language:** Python 3
*   **Libraries:**
    *   `pandas` for data manipulation and analysis.
    *   `numpy` for numerical operations.
    *   `scikit-learn` for machine learning (data splitting, `RandomForestClassifier`, metrics).
    *   `matplotlib` & `seaborn` for data visualization.
    *   `Jupyter Notebook` for interactive analysis and documentation.

## Results

The trained model demonstrated a high capability of predicting failures based on the engineered features. The classification report and confusion matrix from the analysis show high precision and recall, indicating that the model can not only accurately predict failures but also minimize false alarms. The most important features identified by the model were related to the rolling averages of temperature and vibration, confirming the initial hypothesis from the EDA.

*(This is where you would paste the final visualizations from your notebook, like the feature importance plot or the confusion matrix PNG)*

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
    ```
3.  **Generate the data:**
    ```bash
    python3 simulate_data.py
    ```
4.  **Run the analysis:**
    ```bash
    jupyter lab predictive_maintenance_analysis.ipynb
    ```
    Inside Jupyter, run all cells to see the complete analysis and results.
