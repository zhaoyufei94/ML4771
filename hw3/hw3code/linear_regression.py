import scipy.io as sio
from sklearn import linear_model

mat = sio.loadmat("MSData.mat")

testx, trainx, trainy = mmat["testx"], mat["trainx"], mat["trainy"]

regr = linear_model.LinearRegression()
regr.fit(trainx, trainy)
py = regr.predict(testx)

with open("prediction.csv", "w") as f:
	f.write("dataid,prediction\n")
    for i in range(51630):
        f.write("{},{}\n".format(i+1, int(py[i, 0])))