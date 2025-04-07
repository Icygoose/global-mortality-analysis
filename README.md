# 🌍 Exploratory Data Analysis and Predictive Modeling of Global Mortality Rates

## 📌 Overview
This project investigates the factors that influence mortality rates across countries by integrating health data from the **World Health Organization (WHO)** and economic indicators from the **World Bank**. The analysis examines how variables such as GDP, environmental conditions, and lifestyle factors impact public health outcomes.

## 📂 Key Files

| File           | Purpose                                                  |
|----------------|----------------------------------------------------------|
| `project.Rmd`  | Main analysis and modeling file                          |
| `project.html` | Rendered report for quick viewing                        |
| `app.R`        | Shiny app for interactive data exploration               |

> ✅ **Note:** Run `project.Rmd` to reproduce the full workflow.  
> 🖱️ Open `project.html` in a browser to read the report.  
> 💻 Launch `app.R` in RStudio to explore the data interactively.

## 📁 Project Structure

- **🧪 Exploratory Data Analysis**  
  Visual exploration of health and economic indicators using univariate, multivariate, and time series plots.

- **🔍 Predictive Modeling**  
  Implementation of **Decision Tree** and **Logistic Regression** classifiers to predict mortality rates.

- **🧬 Clustering Analysis**  
  Application of **K-Means Clustering** to categorize countries based on economic and dietary profiles.

- **📊 Interactive Visualization**  
  A **Shiny app** for dynamic, user-driven data exploration.

## 🧠 Key Findings

- Countries with **higher GDP per Capita** tend to have **lower mortality rates**.
- Factors such as **outdoor air pollution**, **high systolic blood pressure**, **smoking**, and **excessive sodium intake** are associated with increased mortality.
- **Decision Tree models** outperformed Logistic Regression in predictive accuracy.

## 🎥 Demo
👉 [Watch the Demonstration Video](https://youtu.be/9IIQfb88t9U)

## 📦 Required R Libraries

```r
library(ggplot2)      # Visualization
library(dplyr)        # Data manipulation
library(tidyverse)    # Data science packages
library(tidyr)        # Data tidying
library(knitr)        # Tables in R Markdown
library(factoextra)   # Clustering visualization
library(cluster)      # Clustering algorithms
library(corrplot)     # Correlation matrices
library(caret)        # Model training/evaluation
library(pROC)         # ROC/AUC analysis
library(car)          # Multicollinearity check
library(lime)         # Model explanation
library(rpart)        # Decision Trees
library(rpart.plot)   # Tree visualization
library(scales)       # Scale formatting
```

## 📂 Data Sources

- **WHO**: `Countries_and_Death_Causes.csv`
- **World Bank**: `GDP_per_Capita_Data.csv`
- **Population Dataset**: `Population_Data.csv`

## 👤 Author

**Binghan CHEN**  
Student ID: 23965953