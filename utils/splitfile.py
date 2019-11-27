import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path


def data_split(filename,split_size,seed):
    data=pd.read_csv(filename)
    data=shuffle(data)
    i=0
    break_condition = False
    while(True):
        print(data.shape)
        if data.shape[0] < split_size:
            split_size = data.shape[0]
            break_condition=True
        data_split=data.sample(n=split_size,replace=False,random_state=seed)
        data_split.to_csv(Path(filename).stem+"_" + str(i) + ".csv",index=False)
        data = data.drop(data_split.index)
        i+=1
        if (break_condition) | (data.shape[0]==0):
            break





if __name__ == "__main__":
    seed=123
    # Split given csv file to different csv each of size 1500
    data_split(filename="Tweets_final_part3.csv",split_size=1500,seed=seed)
    # Split 1500 tweet data csv to 10 different files each of size 150
    data_split(filename="Tweets_final_part3_0.csv",split_size=150,seed=seed)
