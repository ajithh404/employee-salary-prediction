# Employee Salary Prediction

A machine learning web app built with Streamlit that predicts employee salaries based on various input features like education, experience, job title, location, age, and gender.

---

## Features

* Single prediction using an interactive sidebar
* Batch prediction using CSV upload
* Preprocessing with label encoders and scaler
* Simple, clean UI with helpful tooltips
* Downloadable batch prediction results

---

## Project Structure

```
employee-salary-prediction/
├── app.py                      # Streamlit web app
├── salary_prediction_data.csv  # Dataset used for training
├── best_salary_model.pkl   # Trained ML model
├── label_encoders.pkl      # Encoders for categorical features
├── scaler.pkl              # Scaler for numerical features
├── model_performance_metrics.pkl #Performance metrics of the model
├── employee-salary-prediction.ipynb  # Training and evaluation notebook
├── requirements.txt
└── README.md
```

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/ajithh404/employee-salary-prediction.git
   cd employee-salary-prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the following files are present in the ****`model/`**** folder:**

   * `best_salary_model.pkl`
   * `label_encoders.pkl`
   * `scaler.pkl`
   * `model_performance_metrics.pkl`

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## Data Format

Your input data (for training or batch prediction) should contain the following columns:

* `Education`
* `Experience`
* `Location`
* `Job_Title`
* `Age`
* `Gender`

---

## License

This project is licensed under the MIT License.
