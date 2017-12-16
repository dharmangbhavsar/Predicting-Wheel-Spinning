Python Version - 3.6.3
Files present -
-Prediting Wheel Spinning.zip:-
	->script.py
	->cleaned.csv
	->bkt_model.py
	->lr_model.py
-Predicting Wheel Spinning in Students Report.pdf
-original_data.csv

Save the data file "original_data.csv" and the extracted files from the zip file in a single directory.
To run the project, run the following command

python script.py


It runs the other two scipts in the order mentioned below.
	bkt_model.py  - Runtime ~ 18 minutes
	lr_model.py   - Runtime ~ 7 minutes

bkt_model.py -
	Preprocesses the data after reading a file called "original_data.csv"
	Runs a BKT model on the dataset and sets the prediction column as 0 or 1 based on threshold
		- Threshold is set to 0.99
	Stores the data in a file named "cleaned.csv"

lr_model.py -
	Consumes "cleaned.csv" and applies Logistic regression for different n values and attempts.
	Generates 4 png files 
		- Recall
		- Precision
		- Accuracy
		- F1


Changing Parameters?
To change the parameters for BKT or the BKT threshold, open the bkt_model.py file
parameters are located on line 16 to 25. Change accordingly. Then run the file script.py as mentioned above.
To change the Linear Regression Model to LinearSVC, comment line 122 in lr_model.py and uncomment line 121.

Libraries Used -
sklearn
pandas
numpy
matplotlib
collections

