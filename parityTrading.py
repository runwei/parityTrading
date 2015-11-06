__author__ = 'runwei_zhang'
import numpy as np

### Principal component analysis###
def pca(data, pc_count = None):
    normdata = (data- np.mean(data, 0))/np.std(data, 0)
    C = normdata.transpose().dot(normdata)
    E, V = np.linalg.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    data = data.dot(V)
    return data, V

### Add quadratic terms into the feature set
def quadterms(vecx):
    ret = np.append(vecx,np.multiply(vecx,vecx))
    return ret

### Merge and preprocess data from a list of files for training
def preprocesslist(filenamelist,porder,interval):
    data = np.empty((0,4),float)
    for filename in filenamelist:
        tmpdata = np.genfromtxt(filename, delimiter=',',skip_header=1)
        data = np.append(data,tmpdata,0)
    pricex = data[:,1]
    pricey = data[:,2]
    Returns = data[:,3]
    length = data.shape[0]
    mA=np.empty((0,length-interval*porder),dtype = float)
    for i in xrange(0,porder+1):
        mA = np.append(mA,np.array([pricex[i*interval:length-porder*interval+i*interval]]),0)
        mA = np.append(mA,np.array([pricey[i*interval:length-porder*interval+i*interval]]),0)
    vb = Returns[i*interval:length]
    mA = mA.transpose()
    ret = {
        'pricex': pricex,
        'pricey': pricey,
        'Returns': Returns,
        'length': length,
        'mA' : mA,
        'vb' : vb
    }
    return ret

### Preprocess data from a file for training or testing
def preprocess(filename,porder,interval):
    data = np.genfromtxt(filename, delimiter=',',skip_header=1)
    pricex = data[:,1]
    pricey = data[:,2]
    Returns = data[:,3]
    length = data.shape[0]
    mA=np.empty((0,length-interval*porder),dtype = float)
    for i in xrange(0,porder+1):
        mA = np.append(mA,np.array([pricex[i*interval:length-porder*interval+i*interval]]),0)
        mA = np.append(mA,np.array([pricey[i*interval:length-porder*interval+i*interval]]),0)
    vb = Returns[i*interval:length]
    mA = mA.transpose()
    ret = {
        'pricex': pricex,
        'pricey': pricey,
        'Returns': Returns,
        'length': length,
        'mA' : mA,
        'vb' : vb
    }
    return ret

### Estimate the model parameters using the training data
def modelEstimateHelper(train_data,porder,interval,pcaorder,kernalorder=1):
    mA = train_data['mA']
    if kernalorder ==2:
        mA = np.apply_along_axis(quadterms, axis=1, arr=mA)
    mA, RdMat = pca(mA,pcaorder)
    arr = np.ones((mA.shape[0],1),dtype = float)
    mA = np.append(arr,mA,1)
    coeff = np.linalg.lstsq(mA, train_data['vb'])[0]
    Returns_train = mA.dot(coeff)
    Returns_train = np.append(np.zeros(porder*interval),Returns_train)
    print "R^2:",np.inner(Returns_train,Returns_train)/np.inner(train_data['Returns'],train_data['Returns'])
    parameters = {
        'coeff' : coeff,
        'RdMat': RdMat,
        'porder': porder,
        'interval': interval,
        'kernalorder':kernalorder,
        'pcaorder': pcaorder
    }
    return parameters

### The entry point for training the model
def modelEstimate(filename_train):
    model = np.load('data/model.npz')
    porder = model['porder']
    interval = model['interval']
    pcaorder = model['pcaorder']
    kernalorder = model['kernalorder']
    print "The model we use: porder=",porder,"interval=",interval,"pcaorder=",pcaorder,"kernalorder=",kernalorder
    train_data = preprocess(filename_train,porder,interval)
    return modelEstimateHelper(train_data,porder,interval,pcaorder,kernalorder)

### Test the model parameters using a test dataset, outputMSE controls the number of outputs
def modelForecast(filename_test, parameters,outputMSE=0):
    porder = parameters['porder']
    interval = parameters['interval']
    kernalorder =parameters['kernalorder']
    coeff = parameters['coeff']
    RdMat = parameters['RdMat']
    test_data = preprocess(filename_test,porder,interval)
    mA = test_data['mA']
    if kernalorder ==2:
        mA = np.apply_along_axis(quadterms, axis=1, arr=mA)
    mA = mA.dot(RdMat)
    arr = np.ones((mA.shape[0],1),dtype = float)
    mA = np.append(arr,mA,1)
    Returns_predict = mA.dot(coeff)
    Returns_predict = np.append(np.zeros(porder*interval),Returns_predict)
    print "Correlation coefficients:\n",np.corrcoef(test_data['Returns'],Returns_predict)
    err = (Returns_predict-test_data['Returns'])
    print "Prediction MSE:", np.inner(err,err)/len(err)
    if outputMSE==0:
        return Returns_predict
    else:
        return Returns_predict,np.inner(err,err)/len(err)

### Partition the original file into several segments for cross validation
def separatefile(num_seg):
    my_data = np.genfromtxt('data/data.csv', delimiter=',',skip_header=1)
    length = my_data.shape[0]
    for i in xrange(0,num_seg):
        data_seg = my_data[length/num_seg*i:length/num_seg*(i+1),:]
        np.savetxt("data/data_seg%s.csv"%i, data_seg, delimiter=",",header = "data segment%s"%i)

### Simply partition the data into a train set and a test set
def separatetrainfile():
    my_data = np.genfromtxt('data/data.csv', delimiter=',',skip_header=1)
    length = my_data.shape[0]
    traindata = my_data[0:length*4/5,:]
    testdata = my_data[length*4/5:length,:]
    np.savetxt("data/train.csv", traindata, delimiter=",",header = "train data")
    np.savetxt("data/test.csv", testdata, delimiter=",",header = "test data")

### Perform cross validation of a single model
def cross_validation(model,num_seg):
    porder = model[0]
    interval = model[1]
    pcaorder =model[2]
    kernalorder =model[3]
    my_data = np.genfromtxt('data/data.csv', delimiter=',',skip_header=1)
    MSEavg = 0
    for i in xrange(0,num_seg):
        trainnames =[] # the list of training data segment, [0,...i-1,i+1,...num_seg-1]
        for j in xrange(0,num_seg):
            if j!=i:
                trainnames.append("data/data_seg%s.csv"%j)
        traindata = preprocesslist(trainnames,porder,interval)
        parameters = modelEstimateHelper(traindata,porder,interval,pcaorder,kernalorder)
        predictions,MSE = modelForecast("data/data_seg%s.csv"%i,parameters,1)
        MSEavg = MSEavg +MSE
    return MSEavg/num_seg

### Find the best model
def findbestmodel(num_seg):
    models =np.array([
        [0,0,1,1],
        [10,60,1,1],
        [10,60,1,2],
        [20,10,1,1],
        [20,10,1,2],
        [20,30,1,1],
        [20,30,1,2],
        [20,60,1,1],
        [20,60,1,2],
        [20,120,1,1],
        [20,120,1,2],
        [20,240,1,1],
        [0,0,1,2],
        [10,60,2,1],
        [10,60,2,2],
        [20,10,2,1],
        [20,10,2,2],
        [20,30,2,1],
        [20,30,2,2],
        [20,60,2,1],
        [20,60,2,2],
        [20,120,2,1],
        [20,120,2,2],
        [20,240,2,1],
    ])
    MSE = np.zeros((len(models),1),float)
    for i in xrange(0,len(models)):
        model  =models[i]
        print "porder=", model[0], "interval=", model[1], "pcaorder=", model[2], "kernalorder=",model[3]
        MSE[i] = cross_validation(model,num_seg)
        print "MSE=",MSE[i]
    bestind = np.argmin(MSE)
    bestmodel = models[bestind]
    print "bestmodel:",bestmodel
    np.savez('data/model.npz',porder=bestmodel[0], interval = bestmodel[1], pcaorder =bestmodel[2], kernalorder=bestmodel[3])
    return models[bestind]

### The entry point of the program
if __name__ == "__main__":
    num_seg=5 # number of segments we partition the data
    # separatefile(num_seg)
    # findbestmodel(num_seg)
    parameters = modelEstimate("data/train.csv")
    predictions = modelForecast("data/test.csv",parameters)
