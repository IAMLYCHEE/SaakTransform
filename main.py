'''
@ author: Jiali Duan
@ greatly modified by li yicheng
@ function: Saak Transform
@ Date: 10/29/2017
@ To do: parallelization
'''

# load libs
import torch
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from data.datasets import MNIST
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
from sklearn import svm
# argument parsing
print (torch.__version__)
batch_size=1
test_batch_size=1
kwargs={}
train_loader=data_utils.DataLoader(MNIST(root='./data',train=True,process=False,transform=transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor(),
])),batch_size=batch_size,shuffle=True,**kwargs)


test_loader=data_utils.DataLoader(MNIST(root='./data',train=False,process=False,transform=transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor(),
])),batch_size=test_batch_size,shuffle=True,**kwargs)



# show sample
def show_sample(inv):
    inv_img=inv.data.numpy()[0][0]
    plt.imshow(inv_img)
    plt.gray()
    plt.savefig('./image/demo.png')
    plt.show()

'''
@ For demo use, only extracts the first 1000 samples
'''
def create_numpy_dataset():
    datasets = []
    datalabel = []
    for data in train_loader:
        data_numpy = data[0].numpy()
        data_numpy = np.squeeze(data_numpy)
        datasets.append(data_numpy)
        datalabel.append(data[1].numpy())

    datasets = np.array(datasets)
    datasets=np.expand_dims(datasets,axis=1)
    print( 'Numpy dataset shape is {}'.format(datasets.shape))
    return datasets[:1000],datalabel[:1000]

def create_numpy_testdataset():
    datasets = []
    datalabel = []
    for data in test_loader:
        data_numpy = data[0].numpy()
        data_numpy = np.squeeze(data_numpy)
        datasets.append(data_numpy)
        datalabel.append(data[1].numpy())

    datasets = np.array(datasets)
    datasets=np.expand_dims(datasets,axis=1)
    print( 'Numpy dataset shape is {}'.format(datasets.shape))
    return datasets[:1000],datalabel[:1000]

'''
@ data: flatten patch data: (14*14*60000,1,2,2)
@ return: augmented anchors
'''
def PCA_and_augment(data_in,number_important):
    # data reshape
    data=np.reshape(data_in,(data_in.shape[0],-1))
    print( 'PCA_and_augment: {}'.format(data.shape))
    # mean removal
    mean = np.mean(data, axis=0)
    datas_mean_remov = data - mean
    print( 'PCA_and_augment meanremove shape: {}'.format(datas_mean_remov.shape))

    # PCA, retain all components
    pca=PCA(n_components = number_important)
    pca.fit(datas_mean_remov)
    comps=pca.components_

    # augment, DC component doesn't
    comps_aug=[vec*(-1) for vec in comps]
    comps_complete=np.vstack((comps,comps_aug))
    shapeComps = comps.shape
    mean_kernel = np.ones(shapeComps[1])
    mean_kernel = mean_kernel / np.sqrt(shapeComps[1])
    comps_complete = np.vstack((comps_complete,mean_kernel))
    print( 'PCA_and_augment comps_complete shape: {}'.format(comps_complete.shape))
    return comps,comps_complete



'''
@ datasets: numpy data as input
@ depth: determine shape, initial: 0
'''

def fit_pca_shape(datasets,depth):
    factor=np.power(2,depth)
    length=32/factor
    print ('fit_pca_shape: length: {}'.format(length))
    idx1=range(0,int(length),2)
    idx2=[i+2 for i in idx1]
    print ('fit_pca_shape: idx1: {}'.format(idx1))
    data_lattice=[datasets[:,:,i:j,k:l] for ((i,j),(k,l)) in product(zip(idx1,idx2),zip(idx1,idx2))]
    data_lattice=np.array(data_lattice)
    print ('fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape))

    #shape reshape
    data=np.reshape(data_lattice,(data_lattice.shape[0]*data_lattice.shape[1],data_lattice.shape[2],2,2))
    print ('fit_pca_shape: reshape: {}'.format(data.shape))
    return data


'''
@ Prepare shape changes. 
@ return filters and datasets for convolution
@ aug_anchors: [7,4] -> [7,input_shape,2,2]
@ output_datas: [60000*num_patch*num_patch,channel,2,2]

'''
def ret_filt_patches(aug_anchors,input_channels):
    shape=int(aug_anchors.shape[1]/4)
    num=aug_anchors.shape[0]
    filt=np.reshape(aug_anchors,(num,shape,4))

    # reshape to kernels, (7,shape,2,2)
    filters=np.reshape(filt,(num,shape,2,2))

    # reshape datasets, (60000*shape*shape,shape,28,28)
    # datasets=np.expand_dims(dataset,axis=1)

    return filters



'''
@ input: numpy kernel and data
@ output: conv+relu result
'''
def conv_and_relu(filters,datasets,stride=2):
    # torch data change
    filters_t=torch.from_numpy(filters)
    datasets_t=torch.from_numpy(datasets)

    # Variables
    filt=Variable(filters_t).type(torch.FloatTensor)
    data=Variable(datasets_t).type(torch.FloatTensor)

    # Convolution
    output=F.conv2d(data,filt,stride=stride)

    # Relu
    relu_output=F.relu(output)

    return relu_output,filt



'''
@ One-stage Saak transform
@ input: datasets [60000,channel, size,size]
'''
def one_stage_saak_trans(datasets=None,depth=0,number_important=3):


    # load dataset, (60000,1,32,32)
    # input_channel: 1->7
    print ('one_stage_saak_trans: datasets.shape {}'.format(datasets.shape))
    input_channels=datasets.shape[1]

    # change data shape, (14*60000,4)
    data_flatten=fit_pca_shape(datasets,depth)

    # augmented components
    comps,comps_complete=PCA_and_augment(data_flatten,number_important)
    print ('one_stage_saak_trans: comps_complete: {}'.format(comps_complete.shape))
    # print('one_saak_trans, non-aug kernel size:{}'.format(comps.shape))

    # get filter and datas, (7,1,2,2) (60000,1,32,32)
    filters=ret_filt_patches(comps_complete,input_channels)
    print ('one_stage_saak_trans: filters: {}'.format(filters.shape))

    # output (60000,7,14,14)
    relu_output,filt=conv_and_relu(filters,datasets,stride=2)

    data=relu_output.data.numpy()
    print ('one_stage_saak_trans: output: {}'.format(data.shape))
    return data,filt,relu_output



'''
@ Multi-stage Saak transform
'''
def multi_stage_saak_trans():
    filters = []
    outputs = []
    data,datalabel=create_numpy_dataset()
    dataset=data
    num=0
    img_len=data.shape[-1]
    while(img_len>=2):
        num+=1
        img_len/=2

    stage_number = {0:3,1:4,2:7,3:6,4:8}
    for i in range(num):
        print ('{} stage of saak transform: '.format(i))
        data,filt,output=one_stage_saak_trans(data,depth=i,number_important=stage_number[i])
        filters.append(filt)
        outputs.append(output)
        print ('')


    return dataset,filters,outputs,datalabel


'''
@ Multi-stage Saak transform
'''
def multi_stage_saak_trans_test():
    filters = []
    outputs = []

    data,datalabel=create_numpy_testdataset()
    dataset=data
    num=0
    img_len=data.shape[-1]
    while(img_len>=2):
        num+=1
        img_len/=2


    for i in range(num):
        print ('{} stage of saak transform: '.format(i))
        data,filt,output=one_stage_saak_trans(data,depth=i)
        filters.append(filt)
        outputs.append(output)
        print ('shape of outputs:{}'.format(outputs.shape))


    return dataset,filters,outputs,datalabel

'''
@ Reconstruction from the second last stage
@ In fact, reconstruction can be done from any stage
'''
# def toy_recon(outputs,filters):
#     outputs=outputs[::-1][1:]
#     filters=filters[::-1][1:]
#     num=len(outputs)
#     data=outputs[0]
#     for i in range(num):
#         data = F.conv_transpose2d(data, filters[i], stride=2)
#
#     return data



if __name__=='__main__':

    infoSize = 1000
    dataset, filters, outputs, datalabel = multi_stage_saak_trans()
    # data = toy_recon(outputs, filters)
    print(dataset.shape)
    print(len(outputs))
    for i in range(len(outputs)):
        print(outputs[i].shape)

    feature_data = outputs[-1].data.numpy()
    feature_data = np.squeeze(feature_data)
    print("feature vector dimension", feature_data.shape)
    # take the top 75% information
    feature_data = feature_data[:, :infoSize]
    print("take only infoSize feature", feature_data.shape)
    featuredataSet = feature_data
    print("feature data set has shape",featuredataSet.shape)
    datalabel = np.asarray(datalabel)
    datalabel = np.squeeze(datalabel)
    print(datalabel.shape)
    datalabelSet = datalabel
    print('shape of label set',datalabelSet.shape)

    for i in range(59):
        dataset,filters,outputs,datalabel=multi_stage_saak_trans()
        # data=toy_recon(outputs,filters)
        print(dataset.shape)
        feature_data = outputs[-1].data.numpy()
        feature_data = np.squeeze(feature_data)
        print("feature vector dimension",feature_data.shape)
        feature_data = feature_data[:,:infoSize]
        print("take only infoSize feature",feature_data.shape)
        featuredataSet = np.concatenate((featuredataSet,feature_data),axis=0)
        print("feature data set has shape", featuredataSet.shape)
        datalabel = np.asarray(datalabel)
        datalabel = np.squeeze(datalabel)
        print(datalabel.shape)
        datalabelSet = np.concatenate((datalabelSet,datalabel),axis=0)
        print('shape of label set', datalabelSet.shape)


    pcaComNum = 128
# perform another round of PCA
    pca = PCA(n_components=pcaComNum)
    pca.fit(featuredataSet)
    featuredataSetPCA = np.dot(featuredataSet,pca.components_.transpose())


    # print(feature_data)
    # train the SVM
    print("SVM training start")
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(featuredataSetPCA, datalabelSet)
    print("svm training finished")

    print("Now perform saak tranform on the test data")
    datatestset,filters,outputs,datalabeltest=multi_stage_saak_trans_test()
    testFeatureData = outputs[-1].data.numpy()
    testFeatureData = np.squeeze(testFeatureData)
    testFeatureData = testFeatureData[:,:infoSize]
    datalabel = np.asarray(datalabel)
    testdatalabel = np.squeeze(datalabel)
    testFeatureDataSet = testFeatureData
    testdatalabelSet = testdatalabel
    for i in range(9):
        datatestset, filters, outputs, datalabeltest = multi_stage_saak_trans_test()
        testFeatureData = outputs[-1].data.numpy()
        testFeatureData = np.squeeze(testFeatureData)
        testFeatureData = testFeatureData[:, :infoSize]
        datalabel = np.asarray(datalabel)
        testdatalabel = np.squeeze(datalabel)
        testFeatureDataSet = np.concatenate((testFeatureDataSet,testFeatureData),axis=0)
        testdatalabelSet = np.concatenate((testdatalabelSet,testdatalabel),axis=0)
        print(testFeatureDataSet.shape)

    # perform another round of PCA for test data
    pca = PCA(n_components=pcaComNum)
    pca.fit(testFeatureDataSet)
    testFeatureDataSetPCA = np.dot(testFeatureDataSet, pca.components_.transpose())






    # start the test
    error = 0
    for i in range(10000):
        if clf.predict([testFeatureDataSetPCA[i]]) != testdatalabelSet[i]:
            error = error + 1

    print(error/10000)





