
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

np.random.seed(0)

FILENAME = 'movie_metadata_filtered_aftercsv.csv'
THRESHOLD_PREDICTION = 1

def _make_in_format(filename):
    datadf = pd.read_csv(filename)
    #separate classes and stuffs
    y = np.array(datadf['imdb_score'])
    datadf = datadf.drop(datadf.columns[[0,9]],axis=1)
    #normalize
    datadf = (datadf-datadf.mean())/(datadf.max()-datadf.min())
    X = np.array(datadf)

    return X,y

def _pickle_it(model,filename):
    a = pickle.dumps(model)
    write_file = open('models/'+filename,'w')
    write_file.write((str)(a))

def accuracy_score(y_test,predictions):
        correct = 0
        for i in range(len(y_test)):
            if y_test[i]>=predictions[i]-THRESHOLD_PREDICTION and y_test[i]<=predictions[i]+THRESHOLD_PREDICTION:
            	correct+=1
        accuracy = correct*1.0/len(y_test)
        return accuracy


def naiveBayesG():
    X,y = _make_in_format(FILENAME)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1	)
    model_guass = GaussianNB()
    model_guass.fit(X_train,y_train)
    predictions_gauss = model_guass.predict(X_test)
    _pickle_it(model_guass,"guass_thre1")
    print("naive bayes using gaussian ",accuracy_score(y_test,predictions_gauss)*100)




def main():
    naiveBayesG()

if __name__ == '__main__':
    main()
