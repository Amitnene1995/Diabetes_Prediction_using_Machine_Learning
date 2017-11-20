import csv
import numpy
filename = 'diabetes.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')

inp = data[:,0:7]
tar = data[:,8]
print(data.shape)
print tar
print inp
