
from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd 
import numpy as np
import string
def filter_it(x):
  
  STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
  new_x = []
  
  for email in x:
    new_email = email.lower().strip().split()                                     # converting all to lowecase
    new_email = ' '.join([i for i in new_email if i not in STOP_WORDS])           #removing stop_words
    new_email = new_email.translate(str.maketrans('', '', string.punctuation))    # removing puncuations
    new_x.append(new_email)
  return new_x

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import pylab as plt
import seaborn as sns
import matplotlib.pyplot as plt  

#Step-1: Loading csv file in which emails and their corresponding label are stored

data = pd.read_csv('/content/gdrive/MyDrive/spam2.csv')
print("The size of csv file in which emails and their corresponding label are stored is",len(data))
#Step-2: Spliting data into training and test Data

e = data["EmailText"]
l = data["Label"]

train_e,train_l = e[0:6000],l[0:6000]
test_e,test_l = e[6000:],l[6000:]
train_x= filter_it(train_e)
test_x= filter_it(test_e)

#Step-3: Extracting features
countv = CountVectorizer()  
features = countv.fit_transform(train_x)

#Step-4: Building model
model1 = svm.SVC(kernel = 'linear', C=1)
model1.fit(features,train_l)
print("Accuracy of SVM model with linear kernel in percentage is: ",(100*model1.score(countv.transform(test_x),test_l)))
#Step-5: Save the model
filename = "my_model"
pickle.dump(model1, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

filename1 = "my_cv"
pickle.dump(countv, open(filename1, 'wb'))
loaded_cv = pickle.load(open(filename1, 'rb'))

print("Accuracy of our saved SVM model in percentage is: ",(100*loaded_model.score(loaded_cv.transform(test_x),test_l)))

from sklearn.metrics import confusion_matrix
output=loaded_model.predict(loaded_cv.transform(test_x))
labels = ['HAM', 'SPAM']

cm = confusion_matrix(list(test_l), list(output))   
print("Confusion matrix for our model accuracy is : \n ",cm)
ax1= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax1); #annot=True to annotate cells

# labels, title and ticks
ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax1.xaxis.set_ticklabels(labels); ax1.yaxis.set_ticklabels(labels);

model2 = svm.SVC(kernel = 'rbf')
model2.fit(features,train_l)
print("Accuracy of SVM model with rbd kernel in percentage is: ",(100*model2.score(countv.transform(test_x),test_l)))

model3 = svm.SVC(kernel = 'sigmoid')
model3.fit(features,train_l)
print("Accuracy of SVM model with sigmoid kernel in percentage is: ",(100*model3.score(countv.transform(test_x),test_l)))

model4 = svm.SVC(kernel = 'poly')
model4.fit(features,train_l)
print("Accuracy of SVM model with poly kernel in percentage is: ",(100*model4.score(countv.transform(test_x),test_l)))

#Training SVM model again using spam2.csv data for spam classification
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pickle

#Step-1: Loading csv file in which emails and their corresponding label are stored

data = pd.read_csv('/content/gdrive/MyDrive/spam2.csv')

train_e = data["EmailText"]
train_l = data["Label"]


train_x= filter_it(train_e)

#Step-3: Extracting features
countv = CountVectorizer()  
features = countv.fit_transform(train_x)

#Step-4: Building model
model1 = svm.SVC(kernel = 'linear', C=1)
model1.fit(features,train_l)

#Step-5: Save the model
filename = "my_model"
pickle.dump(model1, open(filename, 'wb'))
loaded_model1 = pickle.load(open(filename, 'rb'))

filename1 = "my_cv"
pickle.dump(countv, open(filename1, 'wb'))
loaded_cv1 = pickle.load(open(filename1, 'rb'))

# Reading set of emails stored in folder named test which is stored in current directory
import glob
import os

os.chdir(r'./test')
myFiles = glob.glob('*.txt')
output_list= []
for files in myFiles:
  content1=""
  with open(files,'r') as file1:
    countriesStr = file1.read()
  output_list.append(countriesStr)
  output_list1=filter_it(output_list)
output=loaded_model1.predict(loaded_cv1.transform(output_list1))
print("predicted values are: "output)
# Storing predicted values in output.txt
with open("./output.txt", "w") as txt_file:
        txt_file.write(str(output))
