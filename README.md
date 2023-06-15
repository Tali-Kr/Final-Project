<h1 align="center"> Prediction of the Optimal Day for a Soccer Game in the Israeli Premier League "Ligat Ha’Al" Using Machine Learning </h1>
The project is for the final project to graduate with a bachelor's in Industrial engineering at the SCE college.

## Project Overview
As part of the project, we conducted a comprehensive study on the influence of match days on both Israeli soccer teams and their respective audiences.

To conduct this research, we relied on data sourced from the "Ligat Ha'Al", and applied advanced models joined with machine learning with Python. These models enabled an in-depth data analysis of the various teams and thus we received recommendations for optimal days.

We built our own dataset by using the “Web scraping" technique to extract the required data from the "Transfermrkt" website.

## Data
Despite the existence of numerous data resources for Ligat Ha'Al, we were unable to find any that offered the option to download the data to personal devices. Therefore, we decided to build our own dataset.

We utilized the technique of "web scraping" to extract the required data from the "Transfermrkt" website. Web scraping involves automatically accessing a website's HTML code using a software program or tool to locate and extract relevant data.

After successfully extracting all the necessary data, we adjusted it and combined it into a single dataset, which we then carefully cleaned and organized.

## Data Preparation
The data preparation process involved several stages, each contributing to the refinement and enrichment of the dataset. These stages are as follows:

1. **Preprocessing**: In this initial stage, we handled missing values and performed preliminary data processing as needed.

2. **Feature Extraction**: The feature extraction stage encompassed extracting and encoding relevant features using various techniques. We also dealt with outlier data points to reduce bias and constructed new features. This stage required the most extensive manipulation of the data.

3. **Feature Selection**: The final data preparation stage involved feature scaling and selection. We performed data normalization and made an initial selection of features to be included in the models.

### Data Infographics
**Teams point of view**
<p align="center">
    <img width="600" alt="Teams" src="https://github.com/Tali-Kr/Final-Project/assets/126663704/ae14d813-f284-4f19-ba96-3d5c2ed7b21d">
  </p>

**Audiance point of view**
<p align="center">
    <img width="700" alt="Audiance" src="https://github.com/Tali-Kr/Final-Project/assets/126663704/ebae101e-f406-4b97-ace9-814f04269b2d"> 
  </p>

## Model Building

Once the data was prepared, we proceeded with building the models. The model building process consisted of the following steps:

1. **Model Selection**: We conducted a comprehensive evaluation of different machine learning models suitable for our task. After careful consideration, we selected the XGBoost model as our primary choice. We utilized two variations of the XGBoost model: the XGBRegressor for predicting the optimal number of spectators and the XGBClassifier for predicting the likelihood of a home team win or an away team win.

2. **Model Tuning**: We fine-tuned the selected models by adjusting their hyperparameters and modifying the composition of features included in the models. This iterative process was repeated several times until we achieved impressive accuracy percentages given the size of our dataset. The accuracy of the model for predicting the number of spectators stands at **67%**, while the accuracy for predicting the home team win and away team win is **69%** and **74%**, respectively.

## Optimal Match Days

We demonstrated that our models effectively predict the number of spectators and the probability of a home team win or an away team win. After obtaining the final models, we proceeded to determine the optimal match days for each of the models. To accomplish this, we gathered additional data for previously unencountered matches. Each match was evaluated six times, considering different days of the week (excluding Fridays). This process allowed us to identify the optimal match day for football fans and the optimal match day for the home and away teams. Notably, we found that the day of the week indeed has an impact on the match outcome, with certain days being more favorable for a team's victory.

### Evaluation of Overlapping Optimal Days

Finally, we examined whether there is an overlap between the optimal match day for football fans and the optimal match day for the home or away teams. This evaluation was conducted by cross-referencing the model predictions. 
Our analysis revealed that in approximately 26% of the matches, there is a discrepancy between the optimal match day for football fans and the optimal match day for the home team. Moreover, 75% of the matches were played on Saturdays, indicating a significant overlap. 
<p align="center">
    <img width="255" alt="overlap between the optimal match day for football fans and the optimal match day for the home team" src="https://github.com/Tali-Kr/Final-Project/assets/126663704/336bcc1d-774d-448e-96c3-040681b643ca">
    <img width="400" src="https://github.com/Tali-Kr/Final-Project/assets/126663704/1be56856-a0eb-4663-948b-86b8e5f894e5">
  </p>
In contrast, the overlap between the optimal match day for football fans and the optimal match day for the away team was smaller, accounting for only 12%. Additionally, the majority of these matches (89%) were also played on Saturdays.
<p align="center">
    <img width="319" src="https://github.com/Tali-Kr/Final-Project/assets/126663704/ef0af0ed-ea01-4151-8d4c-b7ca9b200978">
    <img width="400" src="https://github.com/Tali-Kr/Final-Project/assets/126663704/316c3c9f-6216-4444-bd19-3beef5151284">
  </p>

These results demonstrate that home advantage does indeed influence the match outcome, implying that the presence of spectators in the stadium affects the chances of a home team win. The optimal match day for football fans and the home team is Saturday.
