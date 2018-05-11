
import numpy as np
import pandas as pd


def process(filename):
    datadf = pd.read_csv(filename)
    #color, duration, actor_3_fb_likes etc.
    datadf = datadf.drop(datadf.columns[[0,1,3,5,6,8,10,11,14,15,16,17,19,20,21,23,26]],axis=1)
    datadf = datadf.replace(0,float("NaN"))
    datadf = datadf.dropna(axis=0,how='any')
    # datadf.to_csv('movie_metadata_filtered.csv')
    #remove genres
    datadf = datadf.drop(datadf.columns[[3]],axis=1)
    #label classes
    datadf = datadf.astype(int)
    datadf.to_csv('movie_metadata_filtered_aftercsv.csv')


def main():
    process('movie_metadata.csv')

if __name__ == '__main__':
    main()
