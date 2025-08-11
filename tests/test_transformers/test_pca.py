from onad.transform.pca import IncrementalPCA
import numpy as np
import unittest


class TestIncPCA(unittest.TestCase):
    data = data = np.array([[1, 2, 2.5, 5, 5], 
                 [10, 10.5, 11, 8, 4], 
                 [3, 3.5, 7, 10, 9]])
    x = {f'feature_{i}': val for i, val in enumerate([2, 3, 3.5, 11, 5])}
    y = {f'feature_{i}': val for i, val in enumerate([4, 3.4, 9.5, 1, 1])}
    
    def test_ipca_transform_before_learn(self):
        x = {f'feature_{i}': val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        ipca = IncrementalPCA(2)
        self.assertEqual(ipca.transform_one(x), {'component_0': 0.0, 'component_1': 0.0})

    def test_q_greater_d_init(self):
        with self.assertRaises(ValueError):
            ipca = IncrementalPCA(7, keys=['key_01', 'key_02', 'key_03', 'key_04', 'key_05'])

    def test_q_greater_d_dict(self):
        x = {f'feature_{i}': val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        ipca = IncrementalPCA(7)
        with self.assertRaises(ValueError):
            ipca.learn_one(x)

    def test_learn_one(self):
        # no keys
        x = {f'feature_{i}': val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        ipca1 = IncrementalPCA(3)
        ipca1.learn_one(x)
        self.assertListEqual(list(ipca1.window[0]), [2, 3, 3.5, 11, 5])
        # keys match
        ipca2 = IncrementalPCA(3, keys=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'])
        ipca2.learn_one(x)
        self.assertListEqual(list(ipca1.window[0]), [2, 3, 3.5, 11, 5])
        #keys do not match
        ipca3 = IncrementalPCA(3, keys=['key_01', 'key_02', 'key_03', 'key_04', 'key_05'])
        with self.assertRaises(KeyError):
            ipca3.learn_one(x)
    
    def test_initialization(self):  # compare to incRpca from R (onlinePCA package)
        data = np.array([[1, 2, 2.5, 5, 5], 
                 [10, 10.5, 11, 8, 4], 
                 [3, 3.5, 7, 10, 9]])
        data_stream = [{f'feature_{i}': val for i, val in enumerate(dp)} for dp in data]
        ipca = IncrementalPCA(2,center=False, n0=3)
        for data_point in data_stream:
            ipca.learn_one(data_point)
        #print(ipca.values)
        self.assertTrue(np.allclose(ipca.values, [325.52805, 35.94132]))
        #print(ipca.vectors)
        vectors_R = [[-0.3792350, -0.4164001, -0.5165665, -0.5218317, -0.3790019],
                  [-0.4771219, -0.4290828, -0.1729495, 0.4036324, 0.6288179]]
        self.assertTrue(np.allclose(ipca.vectors, vectors_R))
    
    def test_learn_after_initialization(self):  # compare to incRpca from R (onlinePCA package)
        data = np.array([[1, 2, 2.5, 5, 5], 
                 [10, 10.5, 11, 8, 4], 
                 [3, 3.5, 7, 10, 9]])
        x = {f'feature_{i}': val for i, val in enumerate([2, 3, 3.5, 11, 5])}
        y = {f'feature_{i}': val for i, val in enumerate([4, 3.4, 9.5, 1, 1])}
        data_stream = [{f'feature_{i}': val for i, val in enumerate(dp)} for dp in data]
        ipca = IncrementalPCA(2,center=False, n0=3)
        for data_point in data_stream:  #learning data including n0
            ipca.learn_one(data_point)
        #learn x
        ipca.learn_one(x)
        self.assertTrue(np.allclose(ipca.values, [263.20439, 31.11496]))
        #print(ipca.vectors)
        vectors_R = [[-0.3380927, -0.3841677, -0.4758005, -0.5986203, -0.3916327],
                    [0.5038719, 0.4439442, 0.2912170, -0.4938852, -0.4693579]]
        self.assertTrue(np.allclose(ipca.vectors, vectors_R))

        #learn y
        ipca.learn_one(y)
        self.assertTrue(np.allclose(ipca.values, [215.25829, 31.20006]))
        #print(ipca.vectors)
        vectors_R = [[-0.3528824, -0.3885382, -0.5330556, -0.5541488, -0.3650793],
                    [-0.3990229, -0.3074019, -0.4358383, 0.5795662, 0.4695027]]
        self.assertTrue(np.allclose(ipca.vectors, vectors_R))

    def test_transform_one(self):
        q = 2  # subspace
        data = np.array([[1, 2, 2.5, 5, 5], 
                 [10, 10.5, 11, 8, 4], 
                 [3, 3.5, 7, 10, 9]])
        x = {f'feature_{i}': val for i, val in enumerate([2, 3, 3.5, 11, 5])}

        data_stream = [{f'feature_{i}': val for i, val in enumerate(dp)} for dp in data]
        ipca = IncrementalPCA(q, center=False, n0=3)
        for data_point in data_stream:  #learning data including n0
            ipca.learn_one(data_point)
        
        x_transformed = ipca.transform_one(x)
        print(x_transformed)
        self.assertEqual(q, len(x_transformed))


if __name__ == "__main__":
    unittest.main()
