from onad.transform.pca import IncrementalPCA
import numpy as np

if __name__ == "__main__":
    ipca = IncrementalPCA(n_components=2)
    
    # Simulate streaming data (big)
    '''
    ipca = IncrementalPCA(n_components=2)
    import test_data
    data = test_data.data
    '''

    # small data Test
    ipca = IncrementalPCA(n_components=2, n0=3)
    data = data = np.array([[1, 2, 2.5, 5, 5], 
                 [10, 10.5, 11, 8, 4], 
                 [3, 3.5, 7, 10, 9]])
    data_stream = [{f'feature_{i}': val for i, val in enumerate(dp)} for dp in data]
    x = {f'feature_{i}': val for i, val in enumerate([2, 3, 3.5, 11, 5])}
    y = {f'feature_{i}': val for i, val in enumerate([4, 3.4, 9.5, 1, 1])}


    data_stream = [{f'feature_{i}': val for i, val in enumerate(dp)} for dp in data]
    for data_point in data_stream:
        ipca.learn_one(data_point)
    print('init')
    print(ipca.values)
    print(ipca.vectors.T)
    
    print('learn x')
    ipca.learn_one(x)
    print(ipca.values)
    print(ipca.vectors.T)

    print('learn y')
    ipca.learn_one(y)
    print(ipca.values)
    print(ipca.vectors.T)

    # Testing 
    print("\n Transforming [2, 3, 3.5, 11, 5]:")
    print(ipca.transform_one(x))