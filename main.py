from multiprocessing import reduction
import h5py
import numpy as np
import pickle

pge_path = 'colon_nct_feature/pge_dim_reduced_feature.h5'
resnet50_path = 'colon_nct_feature/resnet50_dim_reduced_feature.h5'
inceptionv3_path = 'colon_nct_feature/inceptionv3_dim_reduced_feature.h5'
vgg16_path = 'colon_nct_feature/vgg16_dim_reduced_feature.h5'


# input_data = input('enter your choice of data values , pge_path , resnet50_path, inceptionv3_path,vgg16_path: ')

# reduction_method = input('method1(pca) or method2(umap): ') #type method1 or method2

# method1 = 'pca_feature'
# method2 = 'umap_feature'

class Data:
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c # number of elements from the feature 1- 5000
        
    def out(self):
        
        content = h5py.File(self.a,mode = 'r')
        feature = content[self.b][...]
        filename = np.squeeze(content['file_name'])
        labels = np.char.decode(np.array([x.split(b'/')[2] for x in filename]))
        import random
        random.seed(0)
        selected_index = random.sample(list(np.arange(len(feature))), self.c)
        test_data = feature[selected_index]
        test_label = labels[selected_index]
        return test_data,test_label,labels
   
d1 = Data(pge_path,'pca_feature',100)
[test_data,test_label,labels] = d1.out()