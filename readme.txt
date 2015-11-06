###Major functions explained###
1.separatefile(num_seg)
This function partition the original dataset into num_seg number of files for cross validation

2.findbestmodel(num_seg)
This function tests different AR models and chooses the best one.
Each model is represented using a four-tuple
[porder, interval, pcaorder,kernalorder]
(1)porder decides how many data point we look back
(2)interval decides the sampling interval we take the data points when looking back
(3)kernal order decides the order of features we select, if it is 1, we only look at linear features, if it is 2, we also look at quadratic features
(4)pcaorder decides the number of features we select after a dimension reduction.

For example, when porder=5, interval=10, pcaorder=2. kernalorder=2, we have
Return(t) = [1 pricex(t) pricex(t-10)... pricex(t-50) pricey(t) pricey(t-10)...pricey(t-50) 
		  pricex(t)^2 pricex(t-10)^2... pricex(t-50)^2 pricey(t)^2 pricey(t-10)^2...pricey(t-50)^2]*V +epsilon_t
where V is the linear parameters we need to measure and epsilon_t is the noise term.
Since there are too many features, and our data is weak, we use PCA to reduce the dimensionality of the feature sets.
Therefore, "pcaorder" decides the number of PCAs after reduction.

The best model we find is [20,120,2,1], that is porder=20, interval=120, pcaorder=2,kernalorder=1


3.parameters = modelEstimate(trainfile)
This function trains the model using the trainfile

4.predictions = modelForecast(testfile,parameters) 
This function outputs the predicton on the testfile using the parameters 

###Usage###
python gsacapitcal.py

which does the follows
1.run the function separatefile to partition data
2.run the function findbestmodel to find the best model
3.run the function modelEstimate to obtain the model parameters
4.run the function modelForecast for prediction

