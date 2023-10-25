# NeuroMetrics
NeuroMetrics is an AI-driven platform designed to predict the UPDRS 3 score (Motor abilitity deteroriation), a crucial metric in the Unified Parkinson's Disease Rating Scale. The application collects various inputs from users, including UPDRS scores for different stages (except UPDRS 3), time since diagnosis, medication numeric value, a selection from UniProt proteins, and NPX numeric value. Using this data, the platform predicts the UPDRS 3 score, which assesses the progression and severity of Parkinson's disease in motor ability. This capability makes NeuroMetrics particularly valuable for clinicians, researchers, and potentially patients, as predicting UPDRS 3 provides insights into the current state of the disease, aiding in treatment planning and interventions. This repository includes preprocessing scripts, ML models, and a web application to facilitate these predictions and guide clinical decisions

## 1. Problem Definition
- **Objective**: Develop an AI product to assess the efficacy of Parkinson's disease treatments using the AMP®PD dataset.
- **Key Metrics**: MDS-UPDR scores, NPX, peptide abundances, and associated clinical data.

## 2. Data Pre-processing
- **Data Cleaning**: Handle missing values, outliers, and duplicates.
- **Data Integration**: Merge datasets using unique identifiers (`visit_id`, `patient_id`).
- **Feature Engineering**: Create features like duration since diagnosis and rate of MDS-UPDR score change.

## 3. Exploratory Data Analysis
- **Distribution Analysis**: Examine MDS-UPDR scores, NPX, and peptide abundances.
- **Temporal Analysis**: Study MDS-UPDR scores over time for different treatments.
- **Correlation Analysis**: Identify proteins or peptides correlated with MDS-UPDR scores.

## 4. Hypothesis Testing
- **Treatment Effect**: Test for significant differences in MDS-UPDR scores among treatments.
- **Protein & Peptide Influence**: Test if specific proteins or peptides influence outcomes.

## 5. Model Development
- **Feature Selection**: Choose relevant features based on importance.
- **Model Selection**: 
- **Model Validation**: Use cross-validation for performance.

## 6. Interpretation & Visualization
- **Feature Importance**: Identify influential features.
- **Visualization**: Develop dashboards for patient progression and treatment efficacy.

## 7. Deployment & Monitoring
- **API Development**: Convert the model into an API for healthcare systems.
- https://azizmoussa.com/api.html
  ![Architecture Diagram](https://s3.eu-west-3.amazonaws.com/azizmoussa.com/neurometrics.png)

- **Real-time Monitoring**: Monitor and recalibrate the model.

## 8. Feedback Loop
- **Clinical Feedback**: Get feedback from healthcare professionals.
- **Continuous Learning**: Update the model with new data.

## 9. Ethical Considerations
- **Data Privacy**: Ensure patient data privacy.
- **Bias Mitigation**: Check for biases in the model.

## 10. Documentation & Reporting
- **Model Documentation**: Document model details.
- **Reporting**: Create periodic performance reports.
