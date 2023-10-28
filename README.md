# NeuroMetrics
NeuroMetrics is an AI-driven platform designed to predict the UPDRS 3 score (Motor abilitity deteroriation), a crucial metric in the Unified Parkinson's Disease Rating Scale. The application collects various inputs from users, including UPDRS scores for different stages (except UPDRS 3), time since diagnosis, medication state at evaluation, a selection from UniProt proteins evaluated, and proteins expression results. Using this data, the platform predicts the UPDRS 3 score, which assesses the progression and severity of Parkinson's disease in motor ability. This capability makes NeuroMetrics particularly valuable for clinicians, researchers, and potentially patients, as predicting UPDRS 3 provides insights into the current state of the disease, aiding in treatment planning and interventions. This repository includes preprocessing scripts, ML models, and a web application to facilitate these predictions and guide clinical decisions.

## 1. Problem Definition
- **Objective**: Develop an AI product to predict the UPDRS 3 score, which assesses the progression and severity of Parkinson's disease using the AMP®PD dataset.
- **Key Metrics**: UPDRS 1, UPDRS 2, UPDRS 4 scores, time since diagnosis, NPX, peptide abundances, UniProt proteins, and associated clinical data.

## 2. Data Pre-processing
- **Data Cleaning**: Handle missing values, outliers, and duplicates.
- **Data Integration**: Merge datasets using unique identifiers (`visit_id`, `patient_id`).
- **Feature Engineering**: Create features like duration since diagnosis and rate of UPDRS score change.

## 3. Exploratory Data Analysis
- **Distribution Analysis**: Examine UPDRS scores, NPX, and peptide abundances.
- **Temporal Analysis**: Study UPDRS scores over time for different treatments.
- **Correlation Analysis**: Identify proteins or peptides correlated with UPDRS scores.

## 4. Hypothesis Testing
- **Treatment Effect**: Test for significant differences in UPDRS scores among treatments.
- **Protein & Peptide Influence**: Test if specific proteins or peptides influence the UPDRS 3 outcomes.

## 5. Model Development
- **Feature Selection**: Choose relevant features based on importance.
- **Model Selection**: Opt for the most suitable model based on performance.
- **Model Validation**: Use cross-validation to assess the model's performance.

## 6. Interpretation & Visualization
- **Feature Importance**: Identify influential features.
- **Visualization**: Develop dashboards for patient progression and treatment efficacy.

## 7. Deployment & Monitoring
- **API Development**: Convert the model into an API for healthcare systems. The web application for data input and prediction can be found at [NeuroMetrics Web Application](https://azizmoussa.com/api.html).
  ![Architecture Diagram](https://s3.eu-west-3.amazonaws.com/azizmoussa.com/neurometrics2.png)
- **Real-time Monitoring**: Monitor and recalibrate the model based on incoming data.

## 8. Feedback Loop
- **Clinical Feedback**: Gather feedback from healthcare professionals on predictions.
- **Continuous Learning**: Update the model with new data and insights.

## 9. Ethical Considerations
- **Data Privacy**: Ensure patient data privacy and adhere to relevant regulations.
- **Bias Mitigation**: Continuously check for and rectify biases in the model and data.

## 10. Documentation & Reporting
- **Model Documentation**: Thoroughly document model details, inputs, outputs, and performance metrics.
- **Reporting**: Create periodic performance and usage reports.
