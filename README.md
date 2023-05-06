Download Link: https://assignmentchef.com/product/solved-ee219-project-4-regression-analysis
<br>
<h1>Regression Analysis</h1>

Regression analysis is a statistical procedure for estimating the relationship between a target variable and a set of potentially relevant variables. In this project, we explore basic regression models on a given dataset, along with basic techniques to handle overfitting; namely cross-validation, and regularization. With cross-validation, we test for over-fitting, while with regularization we penalize overly complex models.

<h2>Dataset</h2>

We use a Network backup Dataset, which is comprised of simulated traffic data on a backup system over a network. The system monitors the files residing in a destination machine and copies their changes in four hour cycles. At the end of each backup process, the size of the data moved to the destination as well as the duration it took are logged, to be used for developing prediction models. We define a workflow as a task that backs up data from a group of files, which have similar patterns of change in terms of size over time. The dataset has around 18000 data points with the following columns/variables:

<ul>

 <li>Week index</li>

 <li>Day of the week at which the file back up has started</li>

 <li>Backup start time: Hour of the day</li>

 <li>Workflow ID</li>

 <li>File name</li>

 <li>Backup size: the size of the file that is backed up in that cycle in GB</li>

 <li>Backup time: the duration of the backup procedure in hour</li>

</ul>

<h2>Problem Statement</h2>

<ol>

 <li><strong>Load the dataset</strong>. You can download the dataset from this <a href="https://drive.google.com/open?id=1MAxRbYjOjeaHRDT-fIr10UeUWItlkAoc">link</a><a href="https://drive.google.com/open?id=1MAxRbYjOjeaHRDT-fIr10UeUWItlkAoc">.</a> To get an idea on the type of relationships in your dataset:</li>

</ol>

(a) For a twenty-day period (X-axis unit is day number) plot the backup sizes for all workflows (color coded on the Y-axis), (b) Do the same plot for the first 105-day period. Can you identify any repeating patterns?

<ol start="2">

 <li><strong>Predict </strong>the backup size of a file given the other attributes. We use all attributes, except Backup time, as candidate features for the prediction of backup size.</li>

</ol>

We will try different feature sets, as well as different encoding schemes for each feature and in combination. For example, each of the five features is a categorical variable: Day of the week, hour of the day, work-flow number, file-type, and week number.

For each categorical variable, we could convert it into a one dimensional numerical value. For example, Day of the Week variable could take on values 1<em>,</em>··· <em>,</em>7 corresponding to Monday through Friday. Similarly, the Hour of the Day could be encoded as 1···24. We will refer to this as a scalar encoding.

For each categorical variable that takes one of <em>M </em>values we can also encode it as an <em>M </em>dimensional vector, where only one entry is 1 and the rest are 0’s. Thus for the Day of the Week, Monday could be encoded as [1<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0] and Friday as [0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>1]. We will refer to this encoding as One-Hot-Encoding.

Now for the five variables, when looked at as a set, we have 32 (= 2<sup>5</sup>) possible combinations, where in each combination only a subset of the features are encoded using One-Hot-Encoding and the rest of the features are encoded using a scalar encoding.

For part a-e, for each model you need to <strong>report training and test RMSE from 10-fold cross validation </strong>as basic evaluation of the performance. That is, for each fold you get two numbers: Training RMSE and Test RMSE. In addition, you need to: (i) Plot fitted values against true values scattered over the number of data points and (ii) Plot residuals versus fitted values scattered over the number of data points using the whole dataset for each model with the best parameters you have found. It visualizes how well your model fits the data.

<ul>

 <li>Fit a <strong>linear regression model</strong>. We use ordinary least square as the penalty function. mink<em>Y </em>− <em>Xβ</em>k<sup>2</sup></li>

</ul>

<em>β</em>

, where the minimization is on the coefficient vector <em>β</em>.

<ol>

 <li>First convert each categorical feature into one dimensional numerical values using scalar encoding (e.g. Monday to Sunday can be mapped to 1-7), and then directly use them to fit a basic linear regression model.</li>

 <li><strong>Data Preprocessing: </strong>Standardize (see the Useful Functions Section) all these numerical features, then fit and test the model. How does the fitting result change as shown in the plots?</li>

</ol>

<ul>

 <li><strong>Feature Selection: </strong>Use f regression and mutual information regression measure to select three most important variables respectively. Report the three most important variables you find. Use those three most important variables to train a new linear model, does the performance improve?</li>

</ul>

<ol>

 <li><strong>Feature Encoding: </strong>As explained in the preceding discussions, there are 32 possible combinations of encoding the five categorical variables. Plot the average training RMSE and test RMSE for each combination (in range 1 to 32). Which combinations achieve best performance? Can you provide an intuitive explanation?</li>

 <li><strong>Controlling ill-conditioning and over-fiting: </strong>You should have found obvious increases in test RMSE compared to training RMSE in some combinations, can you explain why this happens? Observe those fitted coefficients. To solve this problem, you can try the following regularizations with suitable parameters. 1. Ridge Regularizer: min

  <ol start="2">

   <li>Lasso Regularizer: mink<em>Y </em>− <em>Xβ</em>k<sup>2 </sup>+ <em>α</em>k<em>β</em>k<sub>1</sub></li>

  </ol></li>

</ol>

<em>β</em>

<ol start="3">

 <li>Elastic Net Regularizer: min (optional)</li>

</ol>

<em>β</em>

For any choice of the hyper-parameter(s) (i.e., <em>α,λ</em><sub>1</sub><em>,λ</em><sub>2</sub>) you will find one of the 32 models with the lowest Test-RMSE. Optimize over choices of <em>α,λ</em><sub>1</sub><em>,lambda</em><sub>2 </sub>to pick one good model. Compare the values of the estimated coefficients for these regularized good models, with the un-regularized best model.

<ul>

 <li>Use a random forest regression model for this same task.</li>

</ul>

<strong>Feature importance in random forest algorithm</strong>: During the training process, for each node, a branching decision is made based on only one feature that minimized a chosen measure of impurity. For classification, it is typically Gini impurity or information gain(entropy) and for regression task, it is variance (see lecture notes). The importance of each feature will be the averaged decreased variance for each node split with this feature in the forest and weighted by the number of samples it splits.

<strong>Out of bag error</strong>: In the random forest regression, since we use bootstrapping, it’s easier and faster to evaluate the generalization ability. For each tree, only a subset of the data set is used to build it (because of sampling) so the data points that are left out can be used as the test set. One can then define predictionRMSE for each tree and then average over all trees. In sklearn random forest regression, oob score will return out of bag <em>R</em><sup>2 </sup>score, so you can calcalute 1- oob score as Out Of Bag error.

Set the parameters of your model with the following initial values.

<ul>

 <li>Number of trees: 20</li>

 <li>Depth of each tree: 4</li>

 <li>Bootstrap: True</li>

 <li>Maximum number of features: 5</li>

</ul>

Recall that a Random Forest model can handle categorical variables without having to use one-hot or scalar encodings.

<ol>

 <li>Report Training and average Test RMSE from 10 fold cross validation (sum up each fold’s square error, divide by total number of data then take square root) and Out Of Bag error you get from this initial model.</li>

 <li>Sweep over number of trees from 1 to 200 and maximum number of features from 1 to 5, plot figure 1 for out of bag error(y axis) against number of trees(x axis), figure 2 for average Test-RMSE(y axis) against number of trees(x axis).</li>

</ol>

<ul>

 <li>Pick another parameter you want to experiment on. Plot similar figure 1 and figure 2 as above. What parameters would you pick to achieve the best performance?</li>

</ul>

<ol>

 <li>Report the feature importances you got from the best random forest regression you find.</li>

 <li>Visualize your decision trees. Pick any tree (estimator) in best random forest (with max depth=4) and plot its structure, which is the root node in this decision tree? Is it the most important feature according to the feature importance reported by the regressor?</li>

</ol>

<ul>

 <li>Now use a neural network regression model (one hidden layer) with all features one-hot encoded. Parameters:

  <ul>

   <li>Number of hidden units</li>

   <li>Activity Function(relu, logistic, tanh)</li>

  </ul></li>

</ul>

Plot Test-RMSE as a function of the number of hidden units for different activity functions. Report the best combination.

<ul>

 <li>Predict the Backup size for each of the workflows separately.

  <ol>

   <li>Using linear regression model. Explain if the fit is improved?</li>

   <li>Try fitting a more complex regression function to your data. You can try a polynomial function of your variables. Try increasing the degree of the polynomial to improve your fit. Again, use a 10 fold cross validation to evaluate your results. Plot the average train and test RMSE of the trained model against the degree of the polynomial you use. Can you find a threshold on the degree of the fitted polynomial beyond which the generalization error of your model gets worse? Can you explain how cross validation helps controlling the complexity of your model?</li>

  </ol></li>

 <li>Use <em>k</em>-nearest neighbor regression and find the best parameter.</li>

</ul>

<ol start="3">

 <li>Compare these regression models you have used and write some comments, such as which model is best at handling categorical features, which model is good at handling sparse features or not? which model overall generates the best results?</li>

 <li>Useful functions

  <ul>

   <li>Linear Regression Model: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">http://scikit-learn.org/stable/modules/gener</a>ated/ <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">linear_model.LinearRegression.html</a></li>

   <li>OneHotEncoder: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">http://scikit-learn.org/stable/modules/generated/sk</a> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">preprocessing.OneHotEncoder.html</a></li>

   <li>Random Forest Model: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">http://scikit-learn.org/stable/modules/generat</a>ed/ <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">ensemble.RandomForestRegressor.html</a></li>

   <li>Neural Network Models: <a href="http://scikit-learn.org/stable/modules/neural_networks_supervised.html">http://scikit-learn.org/stable/modules/neural</a>_ <a href="http://scikit-learn.org/stable/modules/neural_networks_supervised.html">html</a></li>

   <li>Polynomial Transformation: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">http://scikit-learn.org/stable/modules/gen</a>erated/ <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">preprocessing.PolynomialFeatures.html</a></li>

   <li>KNN Regressor: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">http://scikit-learn.org/stable/modules/generated/skl</a> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">neighbors.KNeighborsRegressor.html</a></li>

   <li>Standardization: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler">http://scikit-learn.org/stable/modules/generated/sk</a></li>

  </ul></li>

</ol>

<a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler">preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler</a>