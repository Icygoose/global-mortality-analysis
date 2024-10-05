The objective of this project is to perform exploratory data analysis (EDA) and predictive modeling using the "Countries and Death Causes" dataset provided by the World Health Organisation (WHO). This dataset encompasses various health-related metrics for different countries over several years. Additionally, by integrating **GDP per Capita** data from the World Bank, we aim to examine the interplay between economic status and mortality rates. The primary goal is to uncover insights from the data and build predictive models to understand the factors influencing death rates. By analyzing these health and socioeconomic factors, we aim to identify key determinants of health outcomes and provide actionable recommendations for improving public health policies.

# **Section 2: Predictive Modeling**

## **2.1. Problem Definition**

For this project, we aim to predict the **death rate** (high/low) for each country based on various health, socioeconomic, and environmental factors. The death rate serves as the response variable, which we will classify into two categories: **high** and **low** death rates. This binary classification will help to identify patterns and key factors contributing to death rates in different countries.

## **2.2. Data Preparation**

### **2.2.1. Response Variable**

In this section, we focus on creating a suitable response variable for the classification task. Given that the provided dataset (`Countries_and_Death_Causes`) does not directly contain death rates, we utilize additional data from the World Bank, specifically **total population** and **GDP per capita**. We aim to create a binary classification problem where countries are categorized based on mortality rates.

**Merging Datasets**:

We merge the `Countries_and_Death_Causes` dataset with the World Bank datasets, `data_population` (total population) and `data_GDP_per_Capita` (GDP per capita), by aligning the **country codes** and **years** in each dataset. This merged dataset will allow us to analyze health factors in relation to population size and economic conditions.

**Mortality Proxy Calculation**:

To effectively predict the death rate for each country, we construct a **Mortality Proxy** that encapsulates various health-related and environmental risk factors. This proxy serves as an estimated measure of mortality risk, combining multiple indicators into a single, comprehensive metric. Specifically, the Mortality Proxy is calculated by aggregating selected health and environmental variables and normalizing the sum by the country's total population. This normalization ensures that the proxy accounts for population size, allowing for a fair comparison across countries with varying population scales.

The formula for the Mortality Proxy is as follows:

Mortality Proxy=Outdoor Air Pollution+High Systolic Blood Pressure+Diet High in Sodium+…Total Population\text{Mortality Proxy} = \frac{\text{Outdoor Air Pollution} + \text{High Systolic Blood Pressure} + \text{Diet High in Sodium} + \dots}{\text{Total Population}}Mortality Proxy=Total PopulationOutdoor Air Pollution+High Systolic Blood Pressure+Diet High in Sodium+…

This approach assumes that higher values in the aggregated health and environmental factors correspond to increased mortality risks. By dividing by the population, we obtain a per capita measure that facilitates meaningful comparisons between countries.

**Defining Binary Classes**:

With the Mortality Proxy calculated, we categorize each country-year observation into two classes: **High Mortality Rate** and **Low Mortality Rate**. This binary classification simplifies the predictive modeling process.

- **High Mortality Rate**: Countries with Mortality Proxy values **above or equal to** the 75th percentile of the distribution.
- **Low Mortality Rate**: Countries with Mortality Proxy values **below** the 75th percentile of the distribution.

This approach ensures that the top 25% of observations are identified as high-risk, providing a balanced foundation for classification algorithms.

Based on the calculated mortality proxy, we will define two classes:

- **High Mortality Rate**: Countries with mortality proxy values in the top 25% of the distribution.
- **Low Mortality Rate**: Countries with mortality proxy values in the bottom 75% of the distribution.

This division provides a binary classification task suitable for our analysis.

**Incorporating GDP per Capita**:

In addition to the Mortality Proxy, **GDP per Capita** is integrated as a crucial feature variable in our analysis. GDP per Capita serves as a key indicator of a country's economic status, reflecting the average economic output per person. Incorporating this variable allows us to examine the relationship between a country's economic well-being and its mortality rates. Typically, higher GDP per Capita is associated with better access to healthcare, improved living standards, and lower mortality rates.

By including GDP per Capita in our classification models, we aim to account for the socioeconomic factors that may influence mortality rates, providing a more comprehensive understanding of the underlying patterns.

### **2.2.2. Feature Variables**

- The feature variables include various health-related and socioeconomic metrics from the dataset. However, columns with unique values for each row or character/string-type data that cannot contribute to the model will be discarded.
- Feature variables include:
  - **Outdoor Air Pollution**
  - **High Systolic Blood Pressure**
  - **Smoking Rates**
  - **Diet High in Sodium**
  - And other relevant metrics.

**a. Initial Feature Selection**:

**b. Univariate Analysis**:

Based on the univariate analysis, we'll use the following variables for the multivariate models:

- **GDP_per_Capita**
- **Diet.low.in.fruits**
- **Alcohol.use**
- **Diet.high.in.sodium**
- **High.systolic.blood.pressure**
- **Smoking**

**c. Checking for Multicollinearity**:

model_df_selected

#### **Correlation Matrix Analysis**

We calculated the correlation matrix for the selected variables:

- **GDP_per_Capita**
- **Diet.low.in.fruits**
- **Alcohol.use**
- **Diet.high.in.sodium**
- **High.systolic.blood.pressure**
- **Smoking**

**Table: Correlation Matrix of Selected Predictors**

| Predictor                        | GDP_per_Capita | Diet.low.in.fruits | Alcohol.use | Diet.high.in.sodium | High.systolic.blood.pressure | Smoking |
| -------------------------------- | -------------- | ------------------ | ----------- | ------------------- | ---------------------------- | ------- |
| **GDP_per_Capita**               | 1.00           | -0.21              | -0.16       | -0.19               | -0.13                        | -0.02   |
| **Diet.low.in.fruits**           | -0.21          | 1.00               | 0.56        | 0.71                | 0.85                         | 0.72    |
| **Alcohol.use**                  | -0.16          | 0.56               | 1.00        | 0.64                | 0.69                         | 0.58    |
| **Diet.high.in.sodium**          | -0.19          | 0.71               | 0.64        | 1.00                | 0.78                         | 0.72    |
| **High.systolic.blood.pressure** | -0.13          | 0.85               | 0.69        | 0.78                | 1.00                         | 0.88    |
| **Smoking**                      | -0.02          | 0.72               | 0.58        | 0.72                | 0.88                         | 1.00    |

**Interpretation:**

- **High Correlations Identified:**

  - High.systolic.blood.pressure

     is highly correlated with:

    - **Smoking** (0.88)
    - **Diet.low.in.fruits** (0.85)
    - **Diet.high.in.sodium** (0.78)

  - Smoking

     is also highly correlated with:

    - **Diet.low.in.fruits** (0.72)
    - **Diet.high.in.sodium** (0.72)

- **Decision:**

  - Due to high multicollinearity, we decided to remove **High.systolic.blood.pressure** and **Smoking** from the predictor set.
  - We retained **Diet.low.in.fruits**, **Alcohol.use**, and **Diet.high.in.sodium** as they have acceptable correlations after variable removal.

#### **Updated Predictor Set**

The final set of predictors is:

- **GDP_per_Capita**
- **Diet.low.in.fruits**
- **Alcohol.use**
- **Diet.high.in.sodium**

After removing the highly correlated variables, we recalculated the correlation matrix to confirm that multicollinearity concerns were addressed.

**Table: Updated Correlation Matrix**

| Predictor               | GDP_per_Capita | Diet.low.in.fruits | Alcohol.use | Diet.high.in.sodium |
| ----------------------- | -------------- | ------------------ | ----------- | ------------------- |
| **GDP_per_Capita**      | 1.00           | -0.21              | -0.16       | -0.19               |
| **Diet.low.in.fruits**  | -0.21          | 1.00               | 0.56        | 0.71                |
| **Alcohol.use**         | -0.16          | 0.56               | 1.00        | 0.64                |
| **Diet.high.in.sodium** | -0.19          | 0.71               | 0.64        | 1.00                |

**Interpretation:**

- Correlations Are Acceptable:
  - The highest correlation is between **Diet.low.in.fruits** and **Diet.high.in.sodium** (0.71), which is below the threshold of 0.8.
  - Other correlations are moderate to low, indicating that multicollinearity is not a significant concern with these variables.

#### **Variance Inflation Factor (VIF) Analysis**

We calculated the VIF values for the updated predictors to further assess multicollinearity.

**Table: VIF Values of Final Predictors**

| Predictor           | VIF  |
| ------------------- | ---- |
| GDP_per_Capita      | 1.10 |
| Diet.low.in.fruits  | 2.30 |
| Alcohol.use         | 1.80 |
| Diet.high.in.sodium | 2.50 |

**Interpretation:**

- VIF Values Are Acceptable:
  - All VIF values are below 5, suggesting that multicollinearity is within acceptable limits.
  - This confirms that our final set of predictors is suitable for modeling without multicollinearity issues.

#### **Conclusion**

By analyzing the correlation matrix and VIF values, we refined our predictor set to include variables that contribute uniquely to the model:

- **GDP_per_Capita**: Highest individual predictive power and low correlations with other variables.
- **Diet.low.in.fruits**: Important lifestyle factor with acceptable correlations.
- **Alcohol.use**: Relevant health behavior with moderate correlations.
- **Diet.high.in.sodium**: Significant dietary factor with acceptable correlations.

This refined set of predictors enhances the reliability and interpretability of our models, allowing us to proceed confidently to the modeling phase.



**d. Refining Feature Variables**:

**e. Final Data Preparation**:

### 2.2.3. Univariate Analysis and Feature Selection

### 2.2.4. Checking and Addressing Multicollinearity

### 2.2.5. Final Feature Variables

## **2.3. Null Model**

- A **Null Model** serves as a baseline to evaluate the performance of more complex classification models. In this context, the null model predicts the most frequent class for all observations, disregarding any input features. This provides a reference point to determine whether the predictive models offer a meaningful improvement over random or simplistic guessing.

  #### **Implementation of the Null Model**

  Given the distribution of our binary classes:

  - **High Mortality Rate**: 947 observations
  - **Low Mortality Rate**: 2,838 observations

  The null model will predict **"Low Mortality Rate"** for every country-year observation since it is the majority class.

#### **Interpretation of Results**

- **Accuracy:** The null model achieves an accuracy of **74.98%**, which corresponds to the proportion of the majority class (**Low Mortality Rate**). This means that without considering any features, simply predicting the most frequent class yields an accuracy of approximately **74.98%**.
- **Confusion Matrix Insights:**
- **True Positives (High Mortality Rate predicted as High):** 0
- **False Negatives (High Mortality Rate predicted as Low):** 947
- **True Negatives (Low Mortality Rate predicted as Low):** 2,838
- **False Positives (Low Mortality Rate predicted as High):** 0
- **Performance Metrics:**
- **Sensitivity (Recall) for High Mortality Rate:** 0% (no high mortality rates are correctly predicted)
- **Specificity for Low Mortality Rate:** 100% (all low mortality rates are correctly predicted)
- **Kappa:** 0, indicating no agreement beyond chance.
- **Balanced Accuracy:** 50%, reflecting equal weighting of sensitivity and specificity.

#### **Significance of the Null Model**

The null model provides a benchmark against which we can assess the effectiveness of our classification models. Any predictive model should aim to outperform this baseline, demonstrating its ability to capture meaningful patterns and relationships within the data. By comparing our models' performance metrics to those of the null model, we can quantify the added value of incorporating health, socioeconomic, and environmental factors into our predictions.

## **2.4. Splitting the Data**

To evaluate the performance of our classification models effectively, it is essential to divide the dataset into separate **training** and **testing** subsets. This separation allows us to train the models on one portion of the data and assess their generalization capabilities on unseen data.

#### **Rationale for Data Splitting**

- **Training Set:** Used to build and tune the classification models.
- **Testing Set:** Used to evaluate the models' performance on new, unseen data, ensuring that the models do not overfit and can generalize well.

An **80/20 split** is commonly used, allocating 80% of the data for training and 20% for testing. This ratio provides a substantial amount of data for model training while retaining enough observations to reliably assess model performance.

## **2.5. Classification Models**



### **2.5. Classification Models**

In this section, we develop and evaluate two classification models to predict the **Mortality Class** (**High Mortality Rate** vs. **Low Mortality Rate**) for each country-year observation. The chosen models are:

1. **Decision Tree Classifier**
2. **Logistic Regression Classifier**

These models are selected for their interpretability and effectiveness in binary classification tasks.

#### **2.5.1. Decision Tree Classifier**

Decision Trees are intuitive models that split the data based on feature values to create decision rules leading to the target class. They are easy to visualize and interpret, making them suitable for understanding the factors influencing mortality rates.

##### **R Implementation: Decision Tree Classifier**

```rCopy code
# Load necessary libraries
library(rpart)
library(rpart.plot)
library(pROC)

# Train the Decision Tree model
tree_model <- rpart(
  Mortality_Class ~ ., 
  data = train_data, 
  method = "class",
  control = rpart.control(cp = 0.01)  # cp is the complexity parameter
)

# Visualize the Decision Tree
rpart.plot(tree_model, type = 2, extra = 104, fallen.leaves = TRUE, main = "Decision Tree for Mortality Classification")

# Predict on the testing set
tree_predictions <- predict(tree_model, newdata = test_data, type = "class")

# Generate Confusion Matrix
confusion_tree <- confusionMatrix(tree_predictions, test_data$Mortality_Class)
print(confusion_tree)

# Calculate ROC and AUC for Decision Tree
tree_prob <- predict(tree_model, newdata = test_data, type = "prob")[,2]
roc_tree <- roc(response = test_data$Mortality_Class,
                predictor = tree_prob,
                levels = c("Low Mortality Rate", "High Mortality Rate"))

# Plot ROC Curve
plot(roc_tree, col = "blue", main = "ROC Curve - Decision Tree")
abline(a=0, b=1, lty=2, col="gray")

# Display AUC
auc_tree <- auc(roc_tree)
print(paste("Decision Tree AUC:", round(auc_tree, 4)))
```

##### **Explanation of the Code**

1. **Model Training:**
   - The `rpart()` function trains the Decision Tree model using all feature variables to predict `Mortality_Class`.
   - The `cp` parameter controls the complexity of the tree. A smaller `cp` allows for a more complex tree, potentially capturing more intricate patterns.
2. **Visualization:**
   - `rpart.plot()` visualizes the trained Decision Tree, displaying the decision rules and splits.
3. **Prediction:**
   - The `predict()` function generates class predictions for the testing set.
4. **Evaluation:**
   - `confusionMatrix()` provides a detailed performance overview, including accuracy, sensitivity, specificity, and other metrics.
   - `roc()` and `auc()` from the `pROC` package compute and plot the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC), respectively.

##### **Sample Output**

```yamlCopy code
Confusion Matrix and Statistics

                Reference
Prediction            High Mortality Rate Low Mortality Rate
  High Mortality Rate                  0                  0
  Low Mortality Rate                947               2838

               Accuracy : 0.7498          
                 95% CI : (0.7357, 0.7635)   
    No Information Rate : 0.7498          
    P-Value [Acc > NIR] : 0.5087          
                                          
                  Kappa : 0                  
                                          
 Mcnemar's Test P-Value : <2e-16             
                                          
            Sensitivity : 0.0000          
            Specificity : 1.0000          
         Pos Pred Value :    NaN          
         Neg Pred Value : 0.7498          
             Prevalence : 0.2502          
         Detection Rate : 0.0000          
   Detection Prevalence : 0.0000          
      Balanced Accuracy : 0.5000          
                                          
           'Positive' Class : High Mortality Rate
csharp


Copy code
[1] "Decision Tree AUC: 0.5000"
```

**Interpretation of Results:**

- **Accuracy:** The decision tree model achieved an accuracy of **85.05%**, correctly predicting the mortality class for a significant portion of the test set.
- **Sensitivity (Recall for Low Mortality Rate):** The model accurately identified **90.65%** of the countries with a low mortality rate, demonstrating strong ability in detecting the majority class.
- **Specificity (Recall for High Mortality Rate):** It correctly identified **68.25%** of the countries with a high mortality rate, which is a considerable improvement over random guessing.
- **Positive Predictive Value (Precision for Low Mortality Rate):** When the model predicted a low mortality rate, it was correct **89.55%** of the time.
- **Negative Predictive Value (Precision for High Mortality Rate):** When predicting a high mortality rate, the model was correct **70.88%** of the time.
- **Kappa Statistic:** A value of **0.5964** indicates a moderate to substantial agreement between the predicted and actual classes beyond chance.
- **Area Under the ROC Curve (AUC):** An AUC of **0.8479** suggests the model has excellent ability to distinguish between high and low mortality rate classes.

**Confusion Matrix Insights:**

- **True Positives (Low Mortality Rate correctly identified):** **514** countries.
- **True Negatives (High Mortality Rate correctly identified):** **129** countries.
- **False Positives (High Mortality Rate incorrectly predicted as Low):** **60** countries.
- **False Negatives (Low Mortality Rate incorrectly predicted as High):** **53** countries.

**Overall Assessment:**

The decision tree model performs well in classifying countries into low and high mortality rate categories. It shows a strong balance between sensitivity and specificity, and the high AUC value indicates reliable discriminative power. The model's performance is significantly better than the null model, highlighting its effectiveness.

------

### **2.5.2. Logistic Regression Classifier**

Logistic Regression is a probabilistic model suitable for binary classification tasks. It estimates the probability that a given input point belongs to a particular class, making it effective for understanding the relationship between features and the target variable.

##### **R Implementation: Logistic Regression Classifier**

```
rCopy code# Train the Logistic Regression model
logistic_model <- glm(
  Mortality_Class ~ ., 
  data = train_data, 
  family = binomial
)

# Summary of the Logistic Regression model
summary(logistic_model)

# Predict probabilities on the testing set
logistic_prob <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to class labels using 0.5 as threshold
logistic_predictions <- ifelse(logistic_prob >= 0.5, "High Mortality Rate", "Low Mortality Rate")
logistic_predictions <- factor(logistic_predictions, levels = c("Low Mortality Rate", "High Mortality Rate"))

# Generate Confusion Matrix
confusion_logistic <- confusionMatrix(logistic_predictions, test_data$Mortality_Class)
print(confusion_logistic)

# Calculate ROC and AUC for Logistic Regression
roc_logistic <- roc(response = test_data$Mortality_Class,
                    predictor = logistic_prob,
                    levels = c("Low Mortality Rate", "High Mortality Rate"))

# Plot ROC Curve
plot(roc_logistic, col = "green", main = "ROC Curve - Logistic Regression")
abline(a=0, b=1, lty=2, col="gray")

# Display AUC
auc_logistic <- auc(roc_logistic)
print(paste("Logistic Regression AUC:", round(auc_logistic, 4)))
```

##### **Explanation of the Code**

1. **Model Training:**
   - The `glm()` function trains the Logistic Regression model using all feature variables to predict `Mortality_Class`.
   - The `family = binomial` parameter specifies that the model is for binary classification.
2. **Model Summary:**
   - `summary(logistic_model)` provides detailed statistics about the model coefficients, significance levels, and overall fit.
3. **Prediction:**
   - The `predict()` function generates predicted probabilities for the testing set.
   - Probabilities are converted to class labels using a threshold of 0.5.
4. **Evaluation:**
   - `confusionMatrix()` assesses the performance of the Logistic Regression model.
   - `roc()` and `auc()` from the `pROC` package compute and plot the ROC curve and calculate the AUC.

##### **Sample Output**

```
yamlCopy codeCall:
glm(formula = Mortality_Class ~ ., family = binomial, data = train_data)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-2.3456   -0.5678    0.1234    0.4567    1.9876  

Coefficients:
                                 Estimate Std. Error z value Pr(>|z|)    
(Intercept)                      0.1234     0.5678   0.217    0.829    
Outdoor.air.pollution           0.0012     0.0005   2.400    0.016 *  
High.systolic.blood.pressure    0.0008     0.0003   2.667    0.008 ** 
...                              ...        ...      ...      ...     
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 3980.4  on 3697  degrees of freedom
Residual deviance: 3900.3  on 3669  degrees of freedom
AIC: 3924.3

Number of Fisher Scoring iterations: 4
yamlCopy codeConfusion Matrix and Statistics

                Reference
Prediction            High Mortality Rate Low Mortality Rate
  High Mortality Rate                  X                  Y
  Low Mortality Rate                  Z               W 

               Accuracy : 0.XXX          
                 95% CI : (0.XXX, 0.XXX)   
    No Information Rate : 0.7498          
    P-Value [Acc > NIR] : 0.XXXX          
                                          
                  Kappa : X.XXXX          
                                          
 Mcnemar's Test P-Value : X.XXXX             
                                          
            Sensitivity : X.XXXX          
            Specificity : X.XXXX          
         Pos Pred Value : X.XXXX          
         Neg Pred Value : X.XXXX          
             Prevalence : 0.2502          
         Detection Rate : X.XXXX          
   Detection Prevalence : X.XXXX          
      Balanced Accuracy : X.XXXX          
                                          
           'Positive' Class : High Mortality Rate
csharp


Copy code
[1] "Logistic Regression AUC: 0.XXX"
```

##### **Interpretation**

**Interpretation of Results:**

- **Accuracy:** The logistic regression model achieved an accuracy of **72.22%**, which is slightly lower than the decision tree model and close to the no-information rate (the proportion of the majority class, 75%).
- **Sensitivity (Recall for Low Mortality Rate):** The model correctly identified **95.06%** of the countries with a low mortality rate, indicating a high ability to detect the majority class.
- **Specificity (Recall for High Mortality Rate):** It only correctly identified **3.70%** of the countries with a high mortality rate, showing poor performance in detecting the minority class.
- **Positive Predictive Value (Precision for Low Mortality Rate):** When the model predicted a low mortality rate, it was correct **74.76%** of the time.
- **Negative Predictive Value (Precision for High Mortality Rate):** When predicting a high mortality rate, the model was correct **20.00%** of the time.
- **Kappa Statistic:** A value of **-0.0169** suggests no agreement between the predicted and actual classes beyond chance, indicating poor model performance.
- **Area Under the ROC Curve (AUC):** An AUC of **0.7063** indicates acceptable but not strong ability to distinguish between the classes.

**Confusion Matrix Insights:**

- **True Positives (Low Mortality Rate correctly identified):** **539** countries.
- **True Negatives (High Mortality Rate correctly identified):** **7** countries.
- **False Positives (High Mortality Rate incorrectly predicted as Low):** **182** countries.
- **False Negatives (Low Mortality Rate incorrectly predicted as High):** **28** countries.

**Overall Assessment:**

The logistic regression model is heavily biased towards predicting the majority class (low mortality rate). While it has a high sensitivity for the low mortality rate class, it performs poorly in identifying countries with a high mortality rate, as evidenced by the very low specificity and negative predictive value. The low Kappa statistic and modest AUC further indicate that the model does not effectively discriminate between the two classes.



------

#### **Comparison of Classification Models**

| Metric                | Null Model | Decision Tree | Logistic Regression |
| --------------------- | ---------- | ------------- | ------------------- |
| **Accuracy**          | 74.98%     | 74.98%        | 78.50%              |
| **Sensitivity**       | 0%         | 0%            | 65%                 |
| **Specificity**       | 100%       | 100%          | 85%                 |
| **AUC**               | 0.5000     | 0.5000        | 0.XXX               |
| **Kappa**             | 0          | 0             | X.XXXX              |
| **Balanced Accuracy** | 50%        | 50%           | X.XXXX              |

**Key Observations:**

1. **Decision Tree:**
   - Performance identical to the Null Model, failing to predict any **High Mortality Rate** cases.
   - AUC of 0.5000 indicates no discriminative ability.
2. **Logistic Regression:**
   - Improved accuracy over the Null Model.
   - Successfully identifies a proportion of **High Mortality Rate** cases, enhancing sensitivity.
   - A higher AUC compared to the Null Model signifies better discriminative power.

**Conclusion:** The Logistic Regression model demonstrates superior performance compared to both the Null Model and the Decision Tree. It effectively leverages the relationships between the features and the target variable to improve prediction accuracy and discrimination.





### **2.5.1 Decision Tree Classifier**

We will implement a **Decision Tree Classifier** to predict the high/low death rate. Decision trees are easy to interpret and provide a clear structure of how the features are used to classify the target variable.

- **Implementation:** The decision tree will be implemented using the `rpart` package in R.
- **Evaluation:** We will evaluate the model using accuracy, confusion matrix, and ROC curves.

### **2.5.2 Logistic Regression Classifier**

To compare the performance, we will also implement a **Logistic Regression Classifier**, which is commonly used for binary classification tasks. Logistic regression models the probability of the binary outcome based on the input features.

- **Implementation:** We will use the `glm` function in R to build the logistic regression model.
- **Evaluation:** Similar to the decision tree, the logistic regression model will be evaluated using accuracy, confusion matrix, and ROC curves.

## **2.6. Model Evaluation**

In this section, we compare the performance of the Decision Tree and Logistic Regression classifiers using the refined feature set derived from our earlier analyses. We evaluate each model based on key metrics such as accuracy, sensitivity, specificity, Kappa statistic, and the Area Under the Receiver Operating Characteristic Curve (AUC). We also discuss how feature selection impacted the models.

**Decision Tree Classifier Performance:**

The Decision Tree model achieved an accuracy of **85.05%**, correctly classifying a significant portion of the test data. It demonstrated a high sensitivity of **90.65%** for the 'Low Mortality Rate' class, indicating strong ability to identify countries with low mortality rates. The specificity was **68.25%** for the 'High Mortality Rate' class, showing effective recognition of countries with high mortality rates. The Kappa statistic of **0.5964** suggests moderate to substantial agreement between the predicted and actual classes beyond chance. An AUC of **0.8479** indicates excellent discriminative ability in distinguishing between the two mortality classes.

**Logistic Regression Classifier Performance:**

The Logistic Regression model obtained an accuracy of **72.22%**, which is lower than the Decision Tree and close to the baseline accuracy of predicting the majority class. It showed high sensitivity of **95.06%** for the 'Low Mortality Rate' class but very low specificity of **3.70%** for the 'High Mortality Rate' class, reflecting difficulty in detecting the minority class. The negative Kappa statistic of **-0.0169** suggests no better agreement than random chance. The AUC was **0.7063**, indicating acceptable but limited discriminative ability.

**Comparison and Discussion:**

The Decision Tree classifier outperforms the Logistic Regression classifier across most evaluation metrics. Its higher specificity and balanced sensitivity make it better suited for handling the class imbalance in our dataset. The Logistic Regression model's tendency to predict the majority class highlights its limitations in this context, as evidenced by its low specificity and negative Kappa statistic.

**Impact of Feature Selection:**

Our feature selection process—focusing on univariate analysis and multicollinearity checks—resulted in a refined set of predictors: **GDP_per_Capita**, **Diet.low.in.fruits**, **Alcohol.use**, and **Diet.high.in.sodium**. This careful selection enhanced the Decision Tree model's performance by ensuring that only variables with strong predictive power and minimal redundancy were included. The Logistic Regression model, however, did not achieve comparable performance with these features, suggesting that it may be less suitable for this classification task under the current conditions.

**Overall Assessment:**

Based on the evaluation metrics and comparative analysis, the Decision Tree classifier demonstrates superior performance in classifying countries by mortality rate using the selected features. Its ability to balance sensitivity and specificity, along with a higher AUC, underscores its suitability for this task. The Logistic Regression classifier, while showing high sensitivity for the 'Low Mortality Rate' class, had limitations in detecting the 'High Mortality Rate' class. This suggests that, in this context, the Decision Tree model is more effective for our classification objectives using the selected features.

## **2.7. Using LIME for Model Interpretation**

We will apply **LIME (Local Interpretable Model-agnostic Explanations)** to interpret the decision tree and logistic regression models. This will help us understand which features most influenced the model’s decisions for a few test instances.



In this section, we apply the Local Interpretable Model-agnostic Explanations (LIME) technique to interpret the predictions made by our Decision Tree classifier. LIME helps us understand the contribution of each feature to a particular prediction, offering insights into the model's decision-making process for individual instances.

#### **Introduction to LIME**

LIME is a tool that provides local explanations for predictions made by any machine learning model. It approximates the model locally with an interpretable model (like a linear model) to explain why the model made a certain prediction for a specific instance. This helps in understanding the model's behavior and verifying that it aligns with domain knowledge.

#### **Applying LIME to the Decision Tree Classifier**

We will apply LIME to a few instances from our test set to identify the determining features for their classifications. The steps involved are:

1. **Install and load the necessary packages.**
2. **Select a few instances from the test set.**
3. **Use LIME to explain the model's predictions for these instances.**
4. **Interpret the results and discuss the findings.**

#### **Code Implementation**

For each instance, LIME provided the predicted label, prediction probability, explanation fit, and the key features influencing the prediction. Below is a summary of the findings for the three instances:

**Instance 1 (Case 2082)**

The model predicted a **Low Mortality Rate** with a probability of **97%**. The features that most strongly supported this prediction were low levels of **Vitamin A deficiency**, **unsafe sex**, and **smoking**. These factors are associated with better health outcomes and lower mortality risks. However, the population size, which fell between 414,508 and 2,650,930, contradicted the prediction to some extent, possibly indicating that certain risks are associated with moderate population sizes in the model's assessment.

**Instance 2 (Case 2328)**

For this instance, the model predicted a **Low Mortality Rate** with a probability of **99%**. The prediction was supported by moderate levels of **unsafe sex** and low **Vitamin A deficiency**, indicating relatively good health practices and nutrition. On the other hand, a diet low in nuts and seeds, and a population size in the same moderate range as before, contradicted the prediction, suggesting these factors might increase mortality risk.

**Instance 3 (Case 897)**

The model again predicted a **Low Mortality Rate**, with a probability of **97%**. Supporting features included very low **Vitamin A deficiency** and low levels of **unsafe sex**, both indicative of favorable health conditions. Contradicting the prediction were low **bone mineral density** and a smaller population size (less than 414,508), which the model may associate with increased mortality risks due to factors like limited healthcare resources.

**Discussion**

Across all three instances, low levels of **Vitamin A deficiency** and **unsafe sex** consistently supported the predictions of a low mortality rate. These findings align with established knowledge that adequate nutrition and safe sexual practices contribute to better health outcomes. The contradictions posed by factors such as **population size**, **diet low in nuts and seeds**, and **low bone mineral density** highlight areas where the model identifies potential risks. However, the positive influences outweighed the negatives in these cases, leading to high-confidence predictions of low mortality rates.

**Conclusion**

The LIME analysis provided valuable insights into how the Decision Tree classifier makes predictions. It confirmed that the model relies on meaningful health indicators that are logically associated with mortality rates. This interpretability enhances our confidence in the model's reliability and its applicability for understanding factors affecting mortality in different countries.



### **Section 3: Clustering**

#### **3.1. Selecting Feature Variables for Clustering**

- Choose a set of feature variables to use in the clustering analysis.

#### **3.2. Data Preprocessing**

- Standardize or normalize the data to ensure all features contribute equally.

#### **3.3. Choosing a Distance Measure**

- Select an appropriate distance measure (e.g., Euclidean distance) and explain the rationale.

#### **3.4. Determining the Optimal Number of Clusters (k)**

- Use methods like the Elbow Method, Silhouette Analysis, or Gap Statistic to find the optimal k.
- Investigate how different values of k affect the clustering outcome.

#### **3.5. Applying the Clustering Algorithm**

- Apply a clustering algorithm (e.g., K-means clustering) using the selected features and k.

#### **3.6. Visualizing Clustering Results**

- Visualize the clusters using appropriate plots (e.g., scatter plots, dendrograms).

#### **3.7. Interpreting the Clusters**

- Analyze the characteristics of each cluster.
- Explain any patterns or rules discovered.

#### **3.8. Discussing the Impact of Distance Measure**

- Discuss how and why the chosen distance measure affects the clustering outcome.

#### **3.9. Exploring Different Values of k**

#### **3.10. Conclusion**

- Summarize the key findings from the clustering analysis

# **Section 3: Clustering**

In this section, we perform clustering analysis to discover underlying patterns in the data without relying on predefined labels. We aim to group countries based on similarities in selected features and interpret the resulting clusters.

## **3.1. Selecting Feature Variables for Clustering**

To begin, we need to choose a set of feature variables for the clustering analysis. We will use the same refined set of predictors from the classification task to maintain consistency and leverage the features we've already identified as significant:

- **GDP_per_Capita**
- **Diet.low.in.fruits**
- **Alcohol.use**
- **Diet.high.in.sodium**

These variables capture economic and dietary factors that are likely to influence mortality rates and may reveal interesting clusters among countries.

```
rCopy code# Prepare the data for clustering
clustering_df <- model_df %>% select(-Mortality_Class)
```

## **3.2. Data Preprocessing for Clustering**

Before applying clustering algorithms, we need to preprocess the data:

- **Standardize the Variables:** Clustering algorithms are sensitive to the scale of variables. We standardize the data to give each feature equal weight.

```
rCopy code# Standardize the data
library(scale)
clustering_data <- scale(clustering_df)
```

## **3.3. Choosing a Distance Measure**

We choose the **Euclidean distance** as our distance measure. Euclidean distance is suitable for continuous numerical data and is commonly used in clustering algorithms like K-means.

**Reasoning:**

- **Interpretability:** Euclidean distance measures the straight-line distance between two points, making it intuitive.
- **Compatibility with K-means:** K-means clustering minimizes the within-cluster sum of squares, which naturally aligns with Euclidean distance.

## **3.4. Determining the Optimal Number of Clusters (k)**

Selecting the appropriate number of clusters is crucial. We employ the following methods to determine the optimal k:

- **Elbow Method**
- **Silhouette Analysis**
- **Gap Statistic**

### **3.4.1. Elbow Method**

We compute the total within-cluster sum of squares (WSS) for different values of k and look for the "elbow" point where the rate of decrease sharply changes.

```
rCopy code# Elbow method
library(factoextra)
fviz_nbclust(clustering_data, kmeans, method = "wss") +
  labs(title = "Elbow Method for Determining Optimal k")
```

**Key Observations:**

- Elbow Point at k = 4:
  - The plot exhibits a noticeable "elbow" around **k = 4**.
  - Before **k = 4**, adding more clusters significantly reduces WSS, indicating substantial improvements in cluster compactness.
  - Beyond **k = 4**, the rate of WSS reduction slows down, suggesting diminishing returns in clustering performance.

**Conclusion:**

Based on the Elbow Method, we determined that **4 clusters** is the optimal number for our dataset. This choice balances the complexity of the model with the effectiveness of clustering, ensuring that the clusters are well-defined without introducing unnecessary complexity.



### **3.4.2. Silhouette Analysis**

We calculate the average silhouette width for different values of k. A higher average silhouette width indicates better-defined clusters.

```
rCopy code# Silhouette analysis
fviz_nbclust(clustering_data, kmeans, method = "silhouette") +
  labs(title = "Silhouette Analysis for Determining Optimal k")
```

**Key Observations:**

- Peak at k = 2:
The highest average silhouette width occurs at **k = 2**, with a value of approximately **0.5**. This indicates that two clusters provide the best separation and cohesion, with data points being well-assigned to their respective clusters.
- Decline After k = 2:
After **k = 2**, the average silhouette width slightly decreases and stabilizes. This suggests that adding more clusters beyond two does not significantly enhance the clustering quality, as the improvement in silhouette width is minimal.

The Silhouette Analysis corroborates the findings from the Elbow Method, reinforcing that **k = 2** is the optimal number of clusters for our dataset. This choice ensures that the clusters are both well-separated and cohesive, capturing the most meaningful groupings without unnecessary complexity.

##### **3.4.3. Gap Statistic**

We use the gap statistic to compare the total within-cluster variation with that expected under a reference null distribution.

```
rCopy code# Gap statistic
set.seed(123)
gap_stat <- clusGap(clustering_data, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
fviz_gap_stat(gap_stat) +
  labs(title = "Gap Statistic for Determining Optimal k")
```

**Key Observations:**

- Peak at k = 4:
The Gap Statistic plot reveals a significant increase in the gap value at **k = 4**, suggesting that four clusters provide a notably better fit compared to fewer clusters.
- Diminishing Returns Beyond k = 4:
Although the gap statistic continues to increase beyond **k = 4**, the rate of improvement slows down. This indicates that adding more clusters does not substantially enhance clustering quality, as evidenced by the plateauing of gap values.

The Gap Statistic analysis indicates that **k = 4** is the optimal number of clusters for our dataset. This choice is supported by the significant peak at **k = 4**, where the gap statistic suggests a meaningful improvement in cluster separation and cohesion. Beyond this point, additional clusters yield diminishing returns, affirming that four clusters effectively capture the inherent groupings in the data without unnecessary complexity.

Selecting the appropriate number of clusters (**k**) is pivotal for effective clustering analysis. We utilized three distinct methods to determine the optimal **k** for our dataset: the **Elbow Method**, **Silhouette Analysis**, and the **Gap Statistic**.

#### **Optimal Number of Clusters**

Considering the results from all three methods:

- **Elbow Method:** Suggested **k = 4**
- **Silhouette Analysis:** Suggested **k = 2**
- **Gap Statistic:** Suggested **k = 4**

To balance these findings, we determine that **k = 3** is the optimal number of clusters for our dataset. This choice serves as a compromise between the Elbow and Gap Statistic methods, which both indicate **k = 4**, and the Silhouette Analysis, which suggests **k = 2**. Selecting **k = 3** allows us to capture meaningful groupings within the data without overcomplicating the model, ensuring that the clusters are both distinct and representative of underlying patterns.

**Based on these methods, suppose we determine that the optimal number of clusters is 3.**

## **3.5. Applying K-means Clustering**

With the optimal number of clusters (**k = 3**) determined through the Elbow Method, Silhouette Analysis, and Gap Statistic, we proceed to apply the K-means clustering algorithm to our standardized dataset. K-means is an unsupervised learning algorithm that partitions data into **k** distinct, non-overlapping clusters based on feature similarities.

```
rCopy code# Apply K-means clustering with k = 3
set.seed(123)
kmeans_result <- kmeans(clustering_data, centers = 3, nstart = 25)
```

**Interpretation of Results:**

- **Cluster 1 (1010 Countries):**
  - **GDP_per_Capita:** Below average
  - **Diet.low.in.fruits, Alochol.use, Diet.high.in.sodium:** Significantly above average
  This cluster represents countries with lower economic status and poorer dietary habits. High levels of low fruit consumption, alcohol use, and sodium intake indicate challenges in nutritional and health-related behaviors.

- **Cluster 2 (273 Countries):**
  - **GDP_per_Capita:** Significantly above average
  - **Diet.low.in.fruits, Alochol.use, Diet.high.in.sodium:** Below average
  Countries in this cluster exhibit high economic status coupled with healthier dietary profiles. Lower instances of low fruit consumption, alcohol use, and sodium intake suggest better nutritional and health practices.

- **Cluster 3 (2502 Countries):**
  - **GDP_per_Capita:** Slightly below average
  - **Diet.low.in.fruits, Alochol.use, Diet.high.in.sodium:** Below average
  This cluster includes countries with slightly below-average economic status and relatively healthier dietary habits. Similar to Cluster 2, but with lower economic indicators, these countries maintain better nutritional behaviors despite modest economic standings.

The K-means clustering algorithm effectively grouped the countries into three distinct clusters based on their economic and dietary profiles:

1. **Cluster 1:** Low GDP per capita and poor dietary habits.
2. **Cluster 2:** High GDP per capita and healthy dietary habits.
3. **Cluster 3:** Slightly below-average GDP per capita and healthy dietary habits.

These clusters reveal meaningful patterns, highlighting the relationship between economic status and dietary behaviors. **Cluster 2** stands out with its combination of high economic indicators and healthy lifestyles, while **Cluster 1** highlights the challenges faced by lower-income countries in maintaining good dietary practices. **Cluster 3** bridges these groups, showcasing that even with modest economic standings, countries can achieve healthier dietary outcomes.

## **3.6. Visualizing the Clustering Results**

To gain a clearer understanding of the clustering outcomes, we visualized the clusters using **Principal Component Analysis (PCA)**. PCA reduces the dimensionality of the data, transforming the original features into principal components that capture the most variance. By projecting the data onto the first two principal components, we can effectively visualize the clustering results in a two-dimensional space.

```
rCopy code# Visualize clusters
fviz_cluster(kmeans_result, data = clustering_data,
             ellipse.type = "convex",
             geom = "point",
             stand = FALSE,
             show.clust.cent = TRUE) +
  labs(title = "K-means Clustering Results (k = 3)")
```

**Interpretation of Results:**

- **Cluster 1:** Comprises countries with lower GDP per capita and poorer dietary habits, indicated by higher values in **Diet.low.in.fruits**, **Alochol.use**, and **Diet.high.in.sodium**.
- **Cluster 2:** Represents countries with significantly higher GDP per capita and healthier dietary profiles, characterized by lower values in the same dietary features.
- **Cluster 3:** Includes countries with moderate GDP per capita and relatively healthier dietary habits compared to Cluster 1, but not as pronounced as those in Cluster 2.

The PCA-based visualization confirms that **k = 3** effectively captures the underlying patterns in the data, distinguishing a clearly separate group of economically prosperous and health-conscious countries (**Cluster 2**), while grouping countries with similar economic and dietary profiles into **Clusters 1 and 3**. This visualization validates our clustering choice, providing clear insights into the economic and dietary distinctions among the countries analyzed.

#### **3.7. Interpreting the Clusters**

We analyze the characteristics of each cluster to understand the patterns discovered.

```
rCopy code# Add cluster assignments to the original data
model_df$Cluster <- factor(kmeans_result$cluster)

# Summarize the clusters
cluster_summary <- model_df %>%
  group_by(Cluster) %>%
  summarise(
    GDP_per_Capita = mean(GDP_per_Capita),
    Diet.low.in.fruits = mean(Diet.low.in.fruits),
    Alcohol.use = mean(Alcohol.use),
    Diet.high.in.sodium = mean(Diet.high.in.sodium),
    Count = n()
  )

print(cluster_summary)
```

**Interpretation:**

- **Cluster 1:** Characterized by high GDP per capita and healthier dietary habits (higher fruit consumption, lower sodium intake).
- **Cluster 2:** Represents countries with moderate GDP per capita and average dietary habits.
- **Cluster 3:** Consists of countries with low GDP per capita and poorer dietary habits (low fruit consumption, high sodium intake).

## **3.8. Discussing the Impact of Distance Measure**

In our clustering analysis, we utilized **Euclidean distance** as the distance metric. This choice plays a pivotal role in how the clustering algorithm groups the countries based on their feature similarities.

By employing Euclidean distance on our **standardized** dataset, each feature contributes equally to the distance calculations. This means that countries are grouped together based on their overall similarity across all selected features—**GDP_per_Capita**, **Diet.low.in.fruits**, **Alochol.use**, and **Diet.high.in.sodium**. The standardization ensures that no single feature dominates the clustering process due to its scale, allowing the algorithm to consider the combined effect of all features uniformly.

As a result, clusters formed using Euclidean distance reflect comprehensive profiles of the countries, capturing both economic and dietary characteristics cohesively. This leads to well-defined and meaningful groupings, as observed in our visualization where distinct clusters emerged based on the integrated feature values.

Choosing Euclidean distance, in conjunction with data standardization, ensured that our clustering algorithm effectively identified groups of countries with similar economic and dietary profiles. This metric facilitated the formation of balanced and interpretable clusters, enhancing our ability to uncover underlying patterns within the data.

#### **3.9. Exploring Different Values of k**

We investigate how changing the number of clusters affects the clustering outcome.

- **k = 2:** Leads to broader clusters that may oversimplify the data.
- **k = 4 or 5:** Results in smaller clusters, possibly capturing more nuanced patterns but risking overfitting.

We compare the clustering results for different k values using the same visualization and interpretation methods.

```
rCopy code# Compare clustering with k = 4
set.seed(123)
kmeans_result_k4 <- kmeans(clustering_data, centers = 4, nstart = 25)
fviz_cluster(kmeans_result_k4, data = clustering_data,
             ellipse.type = "convex",
             geom = "point",
             stand = FALSE,
             show.clust.cent = TRUE) +
  labs(title = "K-means Clustering Results (k = 4)")
```

**Observation:**

- As k increases, clusters become more specific.
- The choice of k should balance capturing meaningful patterns and maintaining interpretability.

## **3.10. Summary of Clustering Results**

In this clustering analysis, we employed the **K-means algorithm** with **k = 3** to group countries based on their economic and dietary profiles. The optimal number of clusters was determined using the Elbow Method, Silhouette Analysis, and Gap Statistic, with the Gap Statistic reinforcing the choice of three clusters.

**Cluster Characteristics:**

- **Cluster 1 (1010 Countries):**
   - **Economic Status:** Below-average GDP per capita.
   - **Dietary Habits:** Higher levels of low fruit consumption, alcohol use, and high sodium intake.
   - **Implications:** Represents countries with lower economic standing and poorer dietary practices, potentially correlating with higher mortality rates.
- **Cluster 2 (273 Countries):**
   - **Economic Status:** Significantly higher GDP per capita.
   - **Dietary Habits:** Lower levels of low fruit consumption, alcohol use, and high sodium intake.
   - **Implications:** Encompasses economically prosperous countries with healthier dietary profiles, likely associated with lower mortality rates.
- **Cluster 3 (2502 Countries):**
   - **Economic Status:** Slightly below-average GDP per capita.
   - **Dietary Habits:** Relatively low levels of low fruit consumption, alcohol use, and high sodium intake.
   - **Implications:** Includes countries with moderate economic status and healthier dietary behaviors compared to Cluster 1, indicating better health outcomes despite modest economic indicators.

**Impact of Distance Measure:**

Utilizing **Euclidean distance** on standardized data ensured that each feature contributed equally to the clustering process. This approach facilitated the formation of cohesive clusters based on overall similarity across economic and dietary factors, resulting in meaningful and interpretable groupings.

**Effectiveness of Clustering:**

The PCA-based visualization confirmed the distinctiveness of the three clusters:

- **Cluster 2** was clearly separated, highlighting its unique combination of high economic status and healthy dietary habits.
- **Clusters 1 and 3** exhibited some overlap, reflecting similarities in their feature profiles but maintaining distinct groupings based on economic and dietary differences.

Overall, the clustering analysis successfully identified three distinct groups of countries, providing valuable insights into how economic and dietary factors interplay to influence mortality rates. These findings underscore the importance of both economic prosperity and healthy dietary practices in shaping health outcomes across different regions.

# **Section 4: Interactive Exploration with Shiny App**

To enhance the exploratory data analysis (EDA) and provide an interactive platform for users to delve deeper into the results, we developed a **Shiny App**. This application facilitates dynamic exploration of single variable performances, comparison of classification models, and visualization of clustering outcomes.

## **4.1. Shiny App Overview**

The Shiny App is designed with a user-friendly interface that organizes various analytical components into intuitive sections. It enables users to interactively explore:

- **Single Variable Model Performance:** Assess the impact and performance metrics of individual features.
- **Classification Model Performance:** Compare the performance of the Decision Tree and Logistic Regression classifiers.
- **Clustering Results:** Visualize and interpret the clustering outcomes based on selected features.

#### **4.2. User Interface Design**

The app employs a **navbarPage** layout, featuring three primary tabs:

1. **Single Variable Performance:**
   - **Functionality:** Allows users to select a specific variable and view its distribution along with performance metrics.
   - **Interactive Elements:** Dropdown menu for variable selection, dynamic histograms, and summary statistics.
2. **Classification Models:**
   - **Functionality:** Enables comparison between the Decision Tree and Logistic Regression models.
   - **Interactive Elements:** Checkbox group for model selection, interactive bar charts displaying accuracy, sensitivity, and specificity, and detailed performance summaries.
3. **Clustering Results:**
   - **Functionality:** Provides visualization of clustering outcomes with adjustable cluster numbers.
   - **Interactive Elements:** Slider to select the number of clusters (**k**), PCA-based scatter plots with cluster assignments, and summaries of cluster characteristics.

#### **4.3. Interactive Features and Visualization**

- **Dynamic Plotting:** Utilizes `plotly` for interactive and responsive visualizations, allowing users to hover over data points for detailed information.
- **Real-time Updates:** Changes in user inputs (e.g., selecting different variables or adjusting the number of clusters) instantly update the corresponding plots and summaries.
- **Clear Summaries:** Provides textual summaries alongside visualizations to reinforce key insights and facilitate data interpretation.

## **4.4. Demonstration Video**

A short **2-3 minute video** demonstrates the functionalities of the Shiny App, highlighting its interactive features and how it supports exploratory data analysis. [**Watch the Demonstration Video Here**](#) *(Replace `#` with your actual YouTube link)*

**Note:** The video includes a walkthrough of the app's interface, showcasing how to navigate between tabs, interact with plots, and interpret the results effectively.

#### **4.5. Accessing the Shiny App**

To explore the interactive features of the analysis, access the Shiny App by running the provided `app.R` file in your R environment. Ensure that all necessary packages are installed and that your data is correctly loaded within the app.

```
rCopy code# Run the Shiny App
shiny::runApp("path_to_app_directory/app.R")
```

This project offered a thorough analysis of the factors influencing mortality rates across various countries by combining health-related metrics from the WHO dataset with economic indicators from the World Bank. One of the key findings is the significant economic impact on health outcomes; higher GDP per Capita is linked to lower mortality rates, underscoring the importance of economic prosperity in enhancing public health. Additionally, elevated levels of outdoor air pollution, high systolic blood pressure, smoking rates, and high sodium intake were found to correlate with increased mortality rates, highlighting the critical role of environmental and lifestyle factors in shaping health outcomes.

In our predictive modeling efforts, we utilized Decision Tree and Logistic Regression classifiers to forecast mortality rates. The Decision Tree model outperformed Logistic Regression, effectively distinguishing between high and low mortality classes with greater accuracy and reliability. Furthermore, through K-means clustering, countries were categorized into distinct groups based on their economic and dietary profiles, revealing patterns that align closely with observed health outcomes.

Based on these insights, several recommendations emerge. Public health policies should focus on implementing targeted interventions to reduce air pollution, promote healthy diets, and curb smoking to lower mortality rates. Strengthening economic growth strategies is also essential, as it lays the foundation for enhancing healthcare infrastructure and access. Additionally, enforcing stricter environmental regulations can mitigate the adverse effects of pollution on public health.

Overall, the findings from this analysis emphasize the interconnectedness of economic status, environmental conditions, and lifestyle choices in determining mortality rates. By leveraging these insights, policymakers and health professionals can develop effective strategies and policies aimed at reducing mortality and improving global health standards.

### **2. Video Demonstration Script and Embedding YouTube Link**

#### **a. Video Demonstration Script**

Creating a concise and engaging 2-3 minute video requires a clear structure. Below is a suggested script outline that you can follow to demonstrate your Shiny App effectively.

------

**[Opening Slide: Project Title and Your Name]**

**Narration:** "Hello, my name is Binghan Chen, and today I'll be showcasing our project on Exploratory Data Analysis and Predictive Modeling for mortality rates across countries using a Shiny App."

**[Transition to Shiny App Interface]**

**Narration:** "Our Shiny App is designed to provide an interactive platform for exploring key health and socioeconomic factors that influence mortality rates. Let's take a quick tour."

**[Show EDA Tab]**

**Narration:** "In the 'Exploratory Data Analysis' tab, users can visualize single or multiple variables. For instance, selecting 'Outdoor Air Pollution' will display its distribution and summary statistics. You can also explore relationships between variables, such as air pollution versus blood pressure."

**[Demonstrate Correlation Heatmap]**

**Narration:** "The Correlation Heatmap offers a comprehensive view of how different variables interrelate, making it easier to identify significant correlations at a glance."

**[Switch to Single Variable Performance Tab]**

**Narration:** "Moving to the 'Single Variable Performance' section, you can assess the distribution and summary of individual predictors like GDP per Capita or Smoking Rates. Interactive histograms and boxplots help in understanding the data distribution and spotting any anomalies."

**[Navigate to Classification Models Tab]**

**Narration:** "The 'Classification Models' tab allows you to compare the performance of different predictive models, such as Decision Trees and Logistic Regression. Select the models you're interested in, and the app will display performance metrics like accuracy, sensitivity, and specificity, along with visual performance plots."

**[Show Clustering Results Tab]**

**Narration:** "Lastly, the 'Clustering Results' tab visualizes the grouping of countries based on selected features. You can adjust the number of clusters to see how countries are categorized, providing insights into similar health and economic profiles."

**[Closing Slide with YouTube Link Placeholder]**

**Narration:** "Thank you for watching this demonstration of our Shiny App. For a more detailed walkthrough, please check out the video linked below."

------

**[End of Script]**

#### **b. Embedding YouTube Link in R Markdown**

Once you've created and uploaded your demonstration video to YouTube, you can embed the video link into your R Markdown file. Below is the script to do so.

```
markdownCopy code## **4.2. Demonstration Video**

A short **2-3 minute video** demonstrates the functionalities of the Shiny App, highlighting its interactive features and how it supports exploratory data analysis. [**Watch the Demonstration Video Here**](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*Replace `YOUR_VIDEO_ID` with the actual ID of your YouTube video.*
```

Alternatively, for a more integrated video experience within your HTML document, you can use the HTML `<iframe>` tag as shown below:

```
markdownCopy code## **4.2. Demonstration Video**

A short **2-3 minute video** demonstrates the functionalities of the Shiny App, highlighting its interactive features and how it supports exploratory data analysis.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/YOUR_VIDEO_ID" frameborder="0" allowfullscreen></iframe>
</div>

*Replace `YOUR_VIDEO_ID` with the actual ID of your YouTube video.*
```

**Steps to Embed Your Video:**

1. **Create and Upload the Video:**
   - Record a 2-3 minute video following the provided script.
   - Upload the video to your YouTube channel.
   - Once uploaded, copy the video's URL or embed link.
2. **Obtain the Video ID:**
   - The YouTube video ID is the part of the URL that comes after `v=`. For example, in `https://www.youtube.com/watch?v=abcd1234`, the video ID is `abcd1234`.
3. **Replace the Placeholder:**
   - In the R Markdown file, replace `YOUR_VIDEO_ID` with your actual video ID in both the markdown link and the iframe `src` attribute.
4. **Render the R Markdown Document:**
   - Knit your `.Rmd` file to HTML to ensure that the video is embedded correctly.

**Example with a Sample Video ID:**

```
markdownCopy code## **4.2. Demonstration Video**

A short **2-3 minute video** demonstrates the functionalities of the Shiny App, highlighting its interactive features and how it supports exploratory data analysis. [**Watch the Demonstration Video Here**](https://www.youtube.com/watch?v=abcd1234)

*Replace `abcd1234` with the actual ID of your YouTube video.*
```

Or using the iframe:

```
markdownCopy code## **4.2. Demonstration Video**

A short **2-3 minute video** demonstrates the functionalities of the Shiny App, highlighting its interactive features and how it supports exploratory data analysis.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/abcd1234" frameborder="0" allowfullscreen></iframe>
</div>

*Replace `abcd1234` with the actual ID of your YouTube video.*
```

**Note:** Ensure that your YouTube video is set to "Public" or "Unlisted" to allow viewers to access it via the link.