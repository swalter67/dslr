import pandas as pd
import numpy as np
import describe as ds


def standardize(arr: np.ndarray) -> np.ndarray:
    """Standardization with Z-score"""
    mean = ds.custom_mean(arr)
    std = ds.custom_std(arr)
    return (arr - mean) / std


def standard(p_data):
    df2 = pd.DataFrame
    for col in p_data:
        mean = ds.custom_mean(p_data[col])
        std = ds.custom_std(p_data[col])
        df2 = df2.append( (p_data[col] - mean) / std)
    return df2

def sigmoid(arr:np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-arr))


#def loss(h, y):
#    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient



def fit(X, y):
    """
    Logistic Regression algo
    x : predictor variables
    y :variables response : Houses
    """
    weight = []
    max_iteration = 10000
    rate = 0.01
    X = np.insert(X, 0,1,axis=1)
    for house in np.unique(y):
        current = np.where(y ==house, 1, 0)
        r = np.ones(X.shape[1])
        for _ in range(max_iteration):
            o = np.dot(X, r) 
            er = current - sigmoid(o)
            grad = np.dot(X.T, er)
            r += rate * grad


    weight.append(r, house)
    

    return weight



def main():
   

   file_path = 'dataset_train.csv'
   df = pd.read_csv(file_path)
   t_data = df["Hogwarts House"]
   p_data = df[["Herbology", "Divination", "Ancient Runes"]]
  
   #standardixzation 
   p_data = standard(p_data)
   print(p_data) 
    
   
   trained = fit(p_data, t_data)

   np.save("pred", np.array(trained, dtype='object'))
   
if __name__ == "__main__":
    main()

