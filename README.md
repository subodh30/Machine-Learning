# Machine-Learning
Movie Prediction Algorithm and Dataset

## Steps
* After preprocessing/cleaning the data there were around 2000 data points.    
* The main task was to predict the IMDB rating of a movie.  
* This was considered as a classification problem by taking 10 classes 1-10 i.e the rating.  
* There were initially many features which was then reduced using the domain knowledge finally only 9 features was taken into consideration, the filtered and processed data is saved in the after csv.csv file.  
* All the models are pickled in the models folder.  
## Results
* Error=0
| Model  	            | Accuracy (%) 	|  
|--------------------	|--------------	|  
| K Nearest Neighbours| 37.52       	|  
| Logitic Regression  | 40.9         	|  
| SVC                 | 36.35        	|  


* Error= +/- 1

| Model  	            | Accuracy (%) 	|  
|--------------------	|--------------	|  
| K Nearest Neighbours| 80.90       	|  
| Logitic Regression  | 85.09         |  
| SVC                 | 83.91        	|  
