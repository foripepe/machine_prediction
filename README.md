# Machine Learning with Keras and Deep Learning for Time Series Prediction

## Data

### Input
Put input data in ./data directory in CSV format.

#### Example dataset for using only one column
OrderDate	Orderlines
1/1/2010	2381

Name: *NL Pb till Dec 2016 small.csv*

#### Example dataset for many columns
VendorId	OrderDate	DayOfTheMonth	Month	Year	Week	Weekday	Orderlines	IsHoliday
1204	1/1/2010	1	1	2010	53	5	2381	1

Name: *NL Pb till Dec 2016.csv*

### Output
Prediction.

## Code
Code is available in python and jupyter notebook format.
https://ipython.org/notebook.html

* lstm_one_column : Simple LSTM with one column.
  Currenly it produces the best performance.
* deep_neural_learning : Deep naural network
* lstm_many_colums_version_1 : LSTM with many columns version 1
  There isn't any prediction in this code.
* lstm_many_colums_version_2 : LSTM with many columns version 2
* deep_neural_learning_2 : Deep naural network
  Deep learning with more data.

## Related repositories
* https://github.com/sajnikanth/prediction

## Documentation
* http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
* http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
* http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
* http://stackoverflow.com/questions/39674713/neural-network-lstm-keras-multiple-inputs
