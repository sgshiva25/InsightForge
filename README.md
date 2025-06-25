
# **InsightForge**

## **Overview**
InsightForge is a data narration tool that provides a comprehensive analysis of your dataset. Simply upload a CSV file, and the tool will automatically:
- Identify and summarize missing values, duplicates, mean, standard deviation, and outliers.
- Perform a detailed column-wise analysis.
- Use machine learning algorithms for value prediction, leveraging Langchain.

## **Features**
- Automated exploratory data analysis (EDA).
- Detailed data insights and visualizations.
- Predictive modeling using integrated ML techniques.
- Interactive user interface powered by Streamlit.

## **Setup Instructions**

### **1. Create a Virtual Environment**
It is recommended to create a virtual environment to manage dependencies:
```bash
python -m venv insightforge_env
source insightforge_env/bin/activate   # On Windows, use insightforge_env\Scripts\activate
```

### **2. Install Dependencies**
Install the required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### **3. Run the Application**
Start the Streamlit application:
```bash
streamlit run app_EDA.py
```

## **Usage**
1. Launch the app as described above.
2. Upload your CSV file through the user interface.
3. Explore the generated insights, including:
   - Summary statistics.
   - Visualizations of dataset features.
   - Predictions based on selected columns.

## **Dependencies**
- Python 3.8 or above.
- Libraries included in `requirements.txt`.

## **Acknowledgments**
- Langchain for powering predictive modeling.
- Streamlit for creating an intuitive and interactive UI.

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
