from os.path import dirname, abspath
import csv

d = dirname(abspath(__file__))
d = d.replace('\\', '/')

train_path = d + "/reviews_tr.csv"
test_path = d + "/reviews_te.csv"
result_train_path = d+"/reviews_tr_2.csv"
result_test_path = d+"/reviews_te_2.csv"

with open(train_path,"r") as file:
	lines = file.readlines()
#with open(result_train_path,"w") as result:
with open(result_train_path, 'w') as csvfile:
#	csvwriter = csv.writer(csvfile, lineterminator='\n')
	count = 0
	for item in lines:
		count += 1
		if count in range (1,500000):
			csvfile.write(item)
		else: pass