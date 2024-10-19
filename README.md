---

## This project includes code based on publicly available content and might not have been entirely written by me. 

---
# ML-Based Phishing Website Detection System (Content-Based)

This project is part of a research-based effort focused on detecting phishing websites using content-based features like HTML tags. The repository includes code for feature extraction, data collection, preparation, and building machine learning models to classify websites as phishing or legitimate.

## File Inputs
- CSV files containing phishing and legitimate URLs:
  - `verified_online.csv` - Phishing URLs from Phishtank.org
  - `tranco_list.csv` - Legitimate URLs from Tranco-list.eu

## Process Overview
- Load URLs from CSV files.
- Fetch content for each URL using Python's `requests` library.
- Parse the content with BeautifulSoup to extract numerical features.
  
- Create a structured dataframe, add labels (1 for phishing, 0 for legitimate), and save the data as CSV files.
- See: `structured_data_legitimate.csv` and `structured_data_phishing.csv`
    
- Split the data for training and testing or use K-fold cross-validation (K=5) as shown in the `machine_learning.py` script.

- Implemented five machine learning models:
  - SVM
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - AdaBoost
    
- Evaluate models using accuracy, precision, recall, and visualize performance.

## Dataset
- You can create your own dataset using the `data_collector.py` script with a custom URL list.
- I used "phishtank.org" & "tranco-list.eu" as data sources.
- Totally 24228 websites ==> 12001 legitimate websites | 12227 phishing websites
- Data set was created in January 2024.

## I may update dataset every year.

---
## Website: https://idsphishing-j75a9csnnmpjqgjtmmbg72.streamlit.app/
---
In case if its showing freezing, ping me.
