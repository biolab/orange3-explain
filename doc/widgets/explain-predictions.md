Explain Predictions
===================

Explains which features contribute the most to the predictions for the selected instances based on the model and how they contribute.

**Inputs**

- Model: model whose predictions are explained by the widget
- Background data: dataset needed to compute explanations
- Data: dataset whose predictions are explained by the widget

**Outputs**

- Selected Data: instances selected from the plot
- Data: original dataset with an additional column showing whether the instance is selected
- Scores: SHAP values for each feature. Features that contribute more to prediction have a higher score deviation from 0.

**Explain Predictions** widget explains classification or regression model's predictions for the provided data instances.
