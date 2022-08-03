ICE
===

Displays one line per instance that shows how the instance’s prediction changes when a feature changes.

**Inputs**

- Model: model
- Data: dataset

The **ICE** (Individual Conditional Expectation) widget visualizes the dependence of the prediction on a feature for each instance separately, resulting in one line per instance, compared to one line overall in partial dependence plots.


![](images/ICE.png)

1. Select a target class.
2. Select a feature.
3. Order features by importance (partial dependence averaged across all the samples).
4. Apply the color of a discrete feature.
5. If **Centered** is ticked, the plot lines will start at the origin of the y-axis.
5. If **Show mean** is ticked, the average across all the samples in the dataset is shown. 
6. If **Send Automatically** is ticked, the output is sent automatically after any change.
   Alternatively, click **Send**.
7. Get help, save the plot, make the report, set plot properties, or observe the size of input and output data.
8. Plot shows a line for each instance in the input dataset.

Example
-------

In the flowing example, we use the ICE widget to explain Random Forest model. In the File widget, we open the Housing dataset. We connect it to the Random Forest widget, which trains the model. The ICE widget accepts the model and data which are used to explain the model.

By selecting some arbitrary lines, the selected instances of the input dataset appear on the output of the ICE widget.

![](images/ICE-Example.png)
