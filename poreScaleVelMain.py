'''
ML for predicting steady state flow fields either in vel form or fq form, in 2D and 3D
'''
from sys import stdout
import argparse
import numpy as np
import glob
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorlayer as tl
import datetime
from scipy import io
from CNNModels import *
from timeit import default_timer as timer
parser = argparse.ArgumentParser(description='PSMLS')
from skimage.measure import compare_psnr as psnr
from tifffile import imsave
import h5py

#please dont be lazy and place function defs within the body of a script like a degenerate
def summarise_model(layerVars):
    gParams=0
    for variable in layerVars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        print(variable.name+f' numParams: {variable_parameters}')
        print(shape)
        gParams += variable_parameters
    print(f'Network Parameters: {gParams}')
    return gParams
    
def load3DDatasetToRAM(path, trainIDs, valIDs):
    print('Loading Datasets into RAM for 3D training')
    
    trainA = np.sort(glob.glob(path+'train_inputs/*'))[0:np.max(trainIDs)]
    trainB = np.sort(glob.glob(path+'train_outputs/*'))[0:np.max(trainIDs)]
    testA = np.sort(glob.glob(path+'validation_inputs/*'))[0:np.max(valIDs)]
    testB = np.sort(glob.glob(path+'validation_outputs/*'))[0:np.max(valIDs)]
    imgA=[]#np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.input_nc), dtype='uint8')
    imgB=[]#np.zeros((len(batch_filesB), args.fine_size*4,  args.fine_size*4, args.output_nc), dtype='uint8')
    n=0
    for imgADir, imgBDir in zip(trainA, trainB):
        stdout.write(f'\rLoading image {imgBDir}')
        stdout.flush()
        img_A = np.load(imgADir).astype('float32')
        img_B = np.load(imgBDir).astype('float32')
        if len(img_A.shape)==3:
            img_A=np.expand_dims(img_A,3)
        else:
            img_A=np.transpose(img_A,(1,2,3,0))
        img_B=np.transpose(img_B,(1,2,3,0))

        imgA.append(img_A[:,:,:,0:3])
        imgB.append(img_B[:,:,:,0:3])
        n=n+1
    stdout.write("\n")
    
    imgAV=[]#np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.input_nc), dtype='uint8')
    imgBV=[]#np.zeros((len(batch_filesB), args.fine_size*4,  args.fine_size*4, args.output_nc), dtype='uint8')
    n=0
    for imgADir, imgBDir in zip(testA, testB):
        stdout.write(f'\rLoading image {imgBDir}')
        stdout.flush()
        img_A = np.load(imgADir).astype('float32')
        img_B = np.load(imgBDir).astype('float32')
        
        if len(img_A.shape)==3:
            img_A=np.expand_dims(img_A,3)
        else:
            img_A=np.transpose(img_A,(1,2,3,0))
        img_B=np.transpose(img_B,(1,2,3,0))
        
        imgAV.append(img_A[:,:,:,0:3])
        imgBV.append(img_B[:,:,:,0:3])
        n=n+1
    stdout.write("\n")

    imgA=np.vstack(np.expand_dims(imgA,0))
    imgB=np.vstack(np.expand_dims(imgB,0))
    imgAV=np.vstack(np.expand_dims(imgAV,0))
    imgBV=np.vstack(np.expand_dims(imgBV,0))
    return imgA, imgB, imgAV, imgBV

def loadDataset(iterNum, batch_size, trainImageIDs, img_width, img_height, path, subset, numOutputs, numInputs, inputType, outputType):

    numBatches=np.floor(np.max(trainImageIDs)/batch_size)
    index = np.mod(iterNum, numBatches)
    beg = index * batch_size
    end = (index + 1) * batch_size
    trainBatchIDs=trainImageIDs[int(beg):int(end)]
  
    output_batch = np.zeros((len(trainBatchIDs), img_width, img_height, numOutputs), dtype='float32')
    
    input_batch = np.zeros((len(trainBatchIDs), img_width, img_height, numInputs), dtype='float32')
    
    for i, id in enumerate(trainBatchIDs):
        hr_path = _hr_image_path(path, subset, 'outputs', id)
        lr_path = _lr_image_path(path, subset, 'inputs' ,id)
        inputs = np.load(lr_path)
        outputs = np.load(hr_path)
        

        if inputs.shape[0]<img_width:
             inputs = np.transpose(inputs,(1,2,0))
        if inputType == 'P':
            inputs=inputs[:,:,-1]
        
        if inputType=='bin':
            inputs=np.expand_dims(inputs,2)

        inputs=inputs[:,:,0:numInputs]
#        outputs=outputs[:,:,0:1]
        input_batch[i] = inputs#[0:128, 0:128]
        if outputs.shape[0]<img_width:
             outputs = np.transpose(outputs,(1,2,0))
        if outputType == 'P':
            outputs=outputs[:,:,-1]
        if numOutputs==1:
            outputs=np.expand_dims(outputs,2)
        #np.where(outputs==0,1.0,outputs)
        outputs=outputs[:,:,0:numOutputs]
        output_batch[i] = outputs#[0:128, 0:128]

    #input_batch=(input_batch-127.5)/127.5
    #output_batch=(output_batch-127.5)/127.5
    return input_batch, output_batch
    
def loadDatasetReg(iterNum, batch_size, trainImageIDs, img_width, img_height, path, subset, numOutputs, numInputs, inputType, outputVec):

    numBatches=np.floor(np.max(trainImageIDs)/batch_size)
    index = np.mod(iterNum, numBatches)
    beg = index * batch_size
    end = (index + 1) * batch_size
    trainBatchIDs=trainImageIDs[int(beg):int(end)]
  
    output_batch = np.zeros((len(trainBatchIDs)), dtype='float32')
    
    input_batch = np.zeros((len(trainBatchIDs), img_width, img_height, numInputs), dtype='float32')
    outputs=[]
    for i, id in enumerate(trainBatchIDs):
        lr_path = _lr_image_path(path, subset, 'inputs' ,id)
        inputs = np.load(lr_path)
        

        if inputs.shape[0]<img_width:
             inputs = np.transpose(inputs,(1,2,0))
        if inputType == 'P':
            inputs=inputs[:,:,-1]
        if inputType=='bin':
            inputs=np.expand_dims(inputs,2)

        inputs=inputs[:,:,0:numInputs]
#        outputs=outputs[:,:,0:1]
        input_batch[i] = inputs#[0:128, 0:128]
        output_batch[i] = outputVec[id-1]#[0:128, 0:128]
        #pdb.set_trace()
    #input_batch=(input_batch-127.5)/127.5
    #output_batch=(output_batch-127.5)/127.5
    return input_batch, output_batch
    
def loadDataset3D(iterNum, batch_size, trainImageIDs, img_width, img_height, img_depth, path, subset, numInputs, numOutputs):

    numBatches=np.floor(np.max(trainImageIDs)/batch_size)
    index = np.mod(iterNum, numBatches)
    beg = index * batch_size
    end = (index + 1) * batch_size
    trainBatchIDs=trainImageIDs[int(beg):int(end)]
  
    output_batch = np.zeros((len(trainBatchIDs), img_width, img_height, img_depth, numOutputs), dtype='float32')
    
    input_batch = np.zeros((len(trainBatchIDs), img_width, img_height, img_depth, numInputs), dtype='float32')
    
    for i, id in enumerate(trainBatchIDs):
        hr_path = _hr_image_path(path, subset, 'outputs', id)
        lr_path = _lr_image_path(path, subset, 'inputs' ,id)
        inputs = np.load(lr_path)
        outputs = np.load(hr_path)
        if numInputs == 1:
            inputs=np.expand_dims(inputs,3)
        if inputs.shape[0]<img_width:
             inputs = np.transpose(inputs,(1,2,3,0))
             

#        outputs=outputs[:,:,0:1]
        input_batch[i] = inputs#[0:128, 0:128]
        if outputs.shape[0]<img_width:
             outputs = np.transpose(outputs,(1,2,3,0))
        #np.where(outputs==0,1.0,outputs) # what the fuck is this?
        output_batch[i] = outputs#[0:128, 0:128]
        
    #input_batch=(input_batch-127.5)/127.5
    #output_batch=(output_batch-127.5)/127.5
    return input_batch, output_batch


def _hr_image_path(path, subset, IO, id):
    return os.path.join(path, f'{subset}_{IO}', f'{id:04}-vels.npy')

def _lr_image_path(path, subset, IO, id):
    return os.path.join(path, f'{subset}_{IO}', f'{id:04}-geom.npy')


def int_range(s):
    try:
        fr, to = s.split('-')
        return range(int(fr), int(to) + 1)
    except Exception:
        raise argparse.ArgumentTypeError(f'invalid integer range: {s}')
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
    
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v
def VoxelMomentum3D(labels):
    vxR = labels[:,:,:,:,1]-labels[:,:,:,:,2]+labels[:,:,:,:,7]-labels[:,:,:,:,8]+labels[:,:,:,:,9]-labels[:,:,:,:,10]+labels[:,:,:,:,11]-labels[:,:,:,:,12]+labels[:,:,:,:,13]-labels[:,:,:,:,14];

    vyR = labels[:,:,:,:,3]-labels[:,:,:,:,4]+labels[:,:,:,:,7]-labels[:,:,:,:,8]-labels[:,:,:,:,9]+labels[:,:,:,:,10]+labels[:,:,:,:,15]-labels[:,:,:,:,16]+labels[:,:,:,:,17]-labels[:,:,:,:,18];

    vzR = labels[:,:,:,:,5]-labels[:,:,:,:,6]+labels[:,:,:,:,11]-labels[:,:,:,:,12]-labels[:,:,:,:,13]+labels[:,:,:,:,14]+labels[:,:,:,:,15]-labels[:,:,:,:,16]-labels[:,:,:,:,17]+labels[:,:,:,:,18];
    
    vels = tf.concat([tf.expand_dims(vxR,4), tf.expand_dims(vyR,4), tf.expand_dims(vzR,4)],4)
    return vels
    
def VoxelMomentum2D(labels):
    vxR = labels[:,:,:,1]-labels[:,:,:,2]+labels[:,:,:,7]-labels[:,:,:,8]+labels[:,:,:,9]-labels[:,:,:,10]+labels[:,:,:,11]-labels[:,:,:,12]+labels[:,:,:,13]-labels[:,:,:,14];

    vyR = labels[:,:,:,3]-labels[:,:,:,4]+labels[:,:,:,7]-labels[:,:,:,8]-labels[:,:,:,9]+labels[:,:,:,10]+labels[:,:,:,15]-labels[:,:,:,16]+labels[:,:,:,17]-labels[:,:,:,18];

    vzR = labels[:,:,:,5]-labels[:,:,:,6]+labels[:,:,:,11]-labels[:,:,:,12]-labels[:,:,:,13]+labels[:,:,:,14]+labels[:,:,:,15]-labels[:,:,:,16]-labels[:,:,:,17]+labels[:,:,:,18];

    vels = tf.concat([tf.expand_dims(vxR,3), tf.expand_dims(vyR, 3), tf.expand_dims(vzR, 3)], 3)
    return vels
##
# Module Flags
##

parser.add_argument('--train', type=str2bool, default=False, help='Flag to activate for training')

parser.add_argument('--test', type=str2bool, default=False, help='Flag to activate for testing')

parser.add_argument('--gpuIDs', type=str, default='0', help='Flag to activate for CPU testing')

##
# Model Parameters
##

parser.add_argument('--nDims', type=str2int, default=2, help='Number dimensions')

parser.add_argument('--residual-blocks', default=1, help='Number of residual blocks')

parser.add_argument('--numFilters', default=64, help='Number of filters')

parser.add_argument('--baseKernelSize', default=3, help='base kernel size')

parser.add_argument('--gan',type=str2bool, default=False)

parser.add_argument('--advRatio', type=str2float, default=1e-3, help='Adversarial and Discriminatory Coefficient')

parser.add_argument('--gLoss', type=str, default='L1', help='L1, L2, LR, or perceptual loss for generator')

parser.add_argument('--alpha',type=str2float, default=0, help='alpha coefficient for mass flux loss for generator')

parser.add_argument('--beta',type=str2float, default=1, help='beta coefficient for density scaling in VelP training')

parser.add_argument('--gamma',type=str2float, default=0, help='gamma coefficient for density scaling in VelP training')

parser.add_argument('--delta',type=str2float, default=1, help='gamma coefficient for vel scaling in Vel training')

parser.add_argument('--batch-size',type=str2int, default=16)

parser.add_argument('--keep-prob',type=str2int, default=0.7)

parser.add_argument('--reluType',type=str, default='concat_relu')

parser.add_argument('--gatedResFlag',type=str2bool, default=True)

parser.add_argument('--width', type=str2int, default=256)
parser.add_argument('--height', type=str2int, default=256)
parser.add_argument('--depth', type=str2int, default=1)
parser.add_argument('--num-epochs',type=str2int, default=500)

parser.add_argument('--epoch-step',type=str2int, default=50)

parser.add_argument('--learnRate',type=str2float, default=1e-4) 

parser.add_argument('--outputType',type=str, default='vel', help='vel or fq') 
parser.add_argument('--inputType',type=str, default='bin', help='bin or velP') 
##
# Model IO
##

parser.add_argument('--dataset',type=str, default='./velMLdistDataset_BIN', help='path to dataset')

parser.add_argument('--trainIDs', type=int_range, default='1-8000', help='training image ids')

parser.add_argument('--valIDs', type=int_range, default='8001-8160', help='validation image ids')

parser.add_argument('--restore', default=None, help='Checkpoint path to restore training')

parser.add_argument('--restoreD', default=None, help='Checkpoint path to restore training')

parser.add_argument('--contEpoch', default=0, help='Restart epoch')

parser.add_argument('--valPlot', type=str2bool, default=True, help='Flag to activate for saving figures')

parser.add_argument('--testInputs', type=str, default='./test', help='where to read test inputs')

args = parser.parse_args()

trainFlag=args.train
testFlag=args.test
gpuIDs=args.gpuIDs
gLoss=args.gLoss
alpha=args.alpha
beta=args.beta
gamma=args.gamma
delta=args.delta
valPlotFlag=args.valPlot

#generator arguments
numFilters=int(args.numFilters)
residual_blocks=int(args.residual_blocks)
baseKernelSize=int(args.baseKernelSize)
keep_prob=args.keep_prob
reluType=args.reluType
gatedResFlag=args.gatedResFlag
outputType=args.outputType
inputType=args.inputType
nDims=args.nDims
#checkpointing
restore=args.restore
restoreD=args.restoreD
#input info
path=args.dataset
trainImageIDs=args.trainIDs
valImageIDs=args.valIDs
img_width=args.width
img_height=args.height
img_depth=args.depth
batch_size=args.batch_size

#training arguments
epochs=args.num_epochs
iterations_train=int(np.ceil(np.max(trainImageIDs)/ batch_size))
learnRate=args.learnRate
numValIterations=int(np.ceil((np.max(valImageIDs)-np.min(valImageIDs))/ batch_size))
# gan arguments
ganFlag=args.gan
advRatio=args.advRatio
'''
TRAINING AND TESTING
'''

# create the learning rate variable
with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(learnRate, trainable=False)
global_step = tf.Variable(0, trainable=False)
learning_rate = lr_v#tf.train.exponential_decay(lr_v, global_step=global_step, decay_rate=decay_rate, decay_steps=decay_steps) 

# create the network input and output variables
if outputType == 'vel':
    numOutputs=nDims
elif outputType == 'fq':
    numOutputs=19
elif outputType == 'velP':
    numOutputs=nDims+1
elif outputType == 'P':
    numOutputs=1
elif outputType == 'k':
    numOutputs=0

if inputType=='bin':
    numInputs=1
elif inputType == 'velP':
    numInputs=nDims+1
elif inputType == 'vel':
    numInputs=nDims
elif inputType == 'P':
    numOutputs=1
if nDims == 2:
    outputShape=[batch_size, img_width, img_height, numOutputs]
    inputShape=[batch_size, img_width, img_height, numInputs]
elif nDims == 3:
    outputShape=[batch_size, img_width, img_height, img_depth, numOutputs]
    inputShape=[batch_size, img_width, img_height, img_depth, numInputs]

if outputType == 'k':
    outputShape=[batch_size]
    outputVec=np.load('./regKTrains.npy')

realVelField = tf.placeholder('float32',outputShape, name='realVelocityFieldsFromCFD')
inputGeom = tf.placeholder('float32',inputShape, name='inputGeometriesToGenerator')

# create the network
predVelField = gatedResnetGenerator(inputGeom, nr_res_blocks=residual_blocks, keep_prob=keep_prob, nonlinearity_name=reluType, gated=gatedResFlag, filter_size = numFilters, kernel_size = baseKernelSize, nDims = nDims, outputType = outputType)

# define the mse loss and other comparative metrics
if gLoss=='L1':
    mse_loss = tf.reduce_sum(tf.abs(predVelField-realVelField), name='MSEGeneratorLoss')
elif gLoss=='L2':
    mse_loss = tf.reduce_sum(tf.square(predVelField-realVelField), name='MSEGeneratorLoss')
elif gLoss=='L0.5':
    mse_loss = tf.reduce_sum(tf.sqrt(tf.abs(predVelField-realVelField)), name='MSEGeneratorLoss')
elif gLoss=='L2Scaled': # only works for input type bin
    weights=inputGeom+1.0
    #weights=tf.compat.v1.div_no_nan(tf.ones_like(weights),weights)+1.0
    #weights=tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)
    if nDims ==2:
        tf.tile(weights,[1,1,1,numOutputs])
    elif nDims == 3:
        tf.tile(weights,[1,1,1,1,numOutputs])
    mse_loss = tf.reduce_sum(tf.square((predVelField-realVelField)*weights), name='MSEGeneratorLoss')
elif gLoss=='L1Scaled':
    weights=inputGeom
    #weights=tf.compat.v1.div_no_nan(tf.ones_like(weights),weights)+1.0
    #weights=tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)
    if nDims ==2:
        tf.tile(weights,[1,1,1,numOutputs])
    elif nDims == 3:
        tf.tile(weights,[1,1,1,1,numOutputs])
    mse_loss = tf.reduce_sum(tf.abs((predVelField-realVelField)*weights), name='MSEGeneratorLoss')
g_loss = mse_loss

#datasetDelta=tf.math.reduce_max(realVelField)-tf.math.reduce_min(realVelField)
#psnrX = tf.image.psnr(predVelField[:,:,:,0],realVelField[:,:,:,0], datasetDelta)
#psnrY = tf.image.psnr(predVelField[:,:,:,1],realVelField[:,:,:,1], datasetDelta)
if outputType == 'vel' or outputType == 'velP' or outputType == 'P' or  outputType == 'k' :
    realVel=realVelField
    predVel=predVelField
elif outputType == 'fq':
    if nDims == 2:
        realVel=VoxelMomentum2D(realVelField)
        predVel=VoxelMomentum2D(predVelField)
    elif nDims ==3:
        realVel=VoxelMomentum2D(realVelField)
        predVel=VoxelMomentum2D(predVelField)
    g_loss = g_loss+tf.reduce_sum(tf.square(predVel-realVel), name='fqMomentumMSE')
if alpha>0:
    if nDims == 2:
           
        flowRateXReal=(tf.reduce_sum(realVel[:,:,:,0], axis=1))
        flowRateXPred=(tf.reduce_sum(predVel[:,:,:,0], axis=1))
        flowRateYReal=(tf.reduce_sum(realVel[:,:,:,1], axis=2))
        flowRateYPred=(tf.reduce_sum(predVel[:,:,:,1], axis=2))
        flowRateXXReal=(tf.reduce_sum(realVel[:,:,:,0], axis=2))
        flowRateXXPred=(tf.reduce_sum(predVel[:,:,:,0], axis=2))
        flowRateYYReal=(tf.reduce_sum(realVel[:,:,:,1], axis=1))
        flowRateYYPred=(tf.reduce_sum(predVel[:,:,:,1], axis=1))

        conservationLoss= alpha*(tf.reduce_sum(tf.square(flowRateXReal-flowRateXPred))+tf.reduce_sum(tf.square(flowRateYReal-flowRateYPred))+tf.reduce_sum(tf.square(flowRateXXReal-flowRateXXPred))+tf.reduce_sum(tf.square(flowRateYYReal-flowRateYYPred)))
    elif nDims == 3:
           
        flowRateXReal=(tf.reduce_sum(realVel[:,:,:,:,0], axis=[2,3]))
        flowRateXPred=(tf.reduce_sum(predVel[:,:,:,:,0], axis=[2,3]))
        flowRateYReal=(tf.reduce_sum(realVel[:,:,:,:,1], axis=[1,3]))
        flowRateYPred=(tf.reduce_sum(predVel[:,:,:,:,1], axis=[1,3]))
        flowRateZReal=(tf.reduce_sum(realVel[:,:,:,:,2], axis=[1,2]))
        flowRateZPred=(tf.reduce_sum(predVel[:,:,:,:,2], axis=[1,2]))
        
        
        flowRateXXReal=(tf.reduce_sum(realVel[:,:,:,:,0], axis=[1,3]))
        flowRateXXPred=(tf.reduce_sum(predVel[:,:,:,:,0], axis=[1,3]))
        flowRateYYReal=(tf.reduce_sum(realVel[:,:,:,:,1], axis=[2,3]))
        flowRateYYPred=(tf.reduce_sum(predVel[:,:,:,:,1], axis=[2,3]))
        flowRateZZReal=(tf.reduce_sum(realVel[:,:,:,:,2], axis=[2,3]))
        flowRateZZPred=(tf.reduce_sum(predVel[:,:,:,:,2], axis=[2,3]))
        
                
        flowRateXXXReal=(tf.reduce_sum(realVel[:,:,:,:,0], axis=[1,2]))
        flowRateXXXPred=(tf.reduce_sum(predVel[:,:,:,:,0], axis=[1,2]))
        flowRateYYYReal=(tf.reduce_sum(realVel[:,:,:,:,1], axis=[1,2]))
        flowRateYYYPred=(tf.reduce_sum(predVel[:,:,:,:,1], axis=[1,2]))
        flowRateZZZReal=(tf.reduce_sum(realVel[:,:,:,:,2], axis=[1,3]))
        flowRateZZZPred=(tf.reduce_sum(predVel[:,:,:,:,2], axis=[1,3]))
        
        conservationLoss= alpha*(tf.reduce_sum(tf.square(flowRateXReal-flowRateXPred))+tf.reduce_sum(tf.square(flowRateYReal-flowRateYPred))+tf.reduce_sum(tf.square(flowRateZReal-flowRateZPred))+tf.reduce_sum(tf.square(flowRateXXReal-flowRateXXPred))+tf.reduce_sum(tf.square(flowRateYYReal-flowRateYYPred))+tf.reduce_sum(tf.square(flowRateZZReal-flowRateZZPred))+tf.reduce_sum(tf.square(flowRateXXXReal-flowRateXXXPred))+tf.reduce_sum(tf.square(flowRateYYYReal-flowRateYYYPred))+tf.reduce_sum(tf.square(flowRateZZZReal-flowRateZZZPred)))
else:
    conservationLoss= tf.convert_to_tensor(0.0, name = 'consLossDummy')

g_loss = g_loss + conservationLoss


if outputType == 'P':
    g_loss = g_loss + 1e-4*tf.reduce_sum(tf.image.total_variation(predVelField))
# add the perm error?

#g_loss=tf.abs(flowRateXReal-flowRateXPred)+tf.abs((flowRateYReal-flowRateYPred))
# compile the mse generator
g_vars = tf.global_variables() #tl.layers.get_variables_with_name('PSMLS_g', True, True)
#gParams=summarise_model(g_vars)
ganStr=''
if ganFlag and trainFlag:
    ganStr='-gan'
    disc_out_real, logits_real = discriminatorTF(input_disc=realVelField, kernel=3, filters=32, is_train=True, reuse=False, nDims=nDims)
    disc_out_fake, logits_fake = discriminatorTF(input_disc=predVelField, kernel=3, filters=32, is_train=True, reuse=True, nDims=nDims)

    d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))
    d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))
    
    d_loss = d_loss1 + d_loss2
    advLoss = advRatio*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)))#tf.reduce_mean(-tf.log(disc_out_fake+eps))
    g_loss = g_loss + advLoss
    d_vars = [var for var in tf.compat.v1.trainable_variables() if 'discriminator' in var.name]
    d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars, global_step=global_step)
    #dParams=summarise_model(d_vars)
else:
    advLoss = tf.convert_to_tensor(0.0, name = 'advLossDummy')
g_optim = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars, global_step=global_step)
# Resources
t_vars = tf.compat.v1.trainable_variables()
total_parameters = summarise_model(t_vars)
print(f'Total Network Parameters: {total_parameters}')
    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpuIDs
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)
#    tl.layers.initialize_global_variables(session)
session.run(tf.initialize_all_variables())
#    tf.global_variables_initializer()

#checkpointing
saver = tf.train.Saver(g_vars, max_to_keep = 10000)
if restore is not None:
    saver.restore(session, restore)
    
if ganFlag:
    #checkpointing
    saverD = tf.train.Saver(d_vars, max_to_keep = 10000)
    if restoreD is not None:
        saverD.restore(session, restoreD)
#    initEpoch=epochs
#else:
initEpoch = 0
    
#some housekeeping
name = path.split('_')[0]
name = name.split('/')[-1]
#training
if trainFlag:
    if nDims == 3:
        trainInputsDataset, trainOutputsDataset, valInputsDataset, valOutputsDataset = load3DDatasetToRAM(path, trainImageIDs, valImageIDs)
    # data saving
    rightNow=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folderName=f'{rightNow}-velCNN{ganStr}-{name}-{residual_blocks}-{baseKernelSize}-{numFilters}-{gLoss}-{alpha}-{beta}-{gamma}-{delta}'
    trainingDir="./outputs/"+folderName
    os.mkdir(trainingDir)
    
    trainOutputDir='./trainingOutputs/'+folderName
    os.mkdir(trainOutputDir)
    oldMeanValMSE=1e10
    oldTrainMSE=1e10
    lrFactor=1.0
    errd=0 
    errdR=0 
    errdF=0
    epochsMSE=np.zeros(epochs)
    for epochNum in range(initEpoch, epochs + initEpoch):
        start=timer()
        # arrays for storing iteration averaged metrics
        trainingMSE=np.zeros(iterations_train)
        
        #learnRate = learnRate/2
        learnRate = args.learnRate*(0.5**(epochNum/args.epoch_step))*lrFactor #if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
        session.run(tf.assign(learning_rate, learnRate))
            
        for iterNum in range(iterations_train):
            #load training batch
            if nDims == 2:
                if outputType == 'k':
                    inputsTrain, outputsTrain = loadDatasetReg(iterNum, batch_size, trainImageIDs, img_width, img_height, path, 'train', numOutputs, numInputs, inputType, outputVec)
                else:
                    inputsTrain, outputsTrain = loadDataset(iterNum, batch_size, trainImageIDs, img_width, img_height, path, 'train', numOutputs, numInputs, inputType, outputType)
            elif nDims == 3:
                numBatches=np.floor(np.max(trainImageIDs)/batch_size)
                index = np.mod(iterNum, numBatches)
                beg = index * batch_size
                end = (index + 1) * batch_size
                inputsTrain = trainInputsDataset[int(beg):int(end)]
                outputsTrain = trainOutputsDataset[int(beg):int(end)]
#            elif nDims == 3:
#                inputsTrain, outputsTrain = loadDataset3D(iterNum, batch_size, trainImageIDs, img_width, img_height, img_depth, path, 'train', numOutputs)
#            plt.figure(2)
#            plt.imshow(inputsTrain[2,:,:,0],origin='lower')
#            plt.quiver(outputsTrain[2,:,:,0], outputsTrain[2,:,:,1])
#            plt.show()
#            pdb.dsfgkjh
            # update G by requesting the optimiser
            if outputType == 'velP':
                if nDims==2:
                    outputsTrain[:,:,:,-1] = (outputsTrain[:,:,:,-1]+gamma)/beta
                elif nDims == 3:
                    outputsTrain[:,:,:,:,-1] = (outputsTrain[:,:,:,:,-1]+gamma)/beta
            elif outputType == 'vel':
                if nDims==2:
                    outputsTrain[:,:,:,0:2] = outputsTrain[:,:,:,0:2]*delta
                elif nDims == 3:
                    outputsTrain[:,:,:,:,0:3] = outputsTrain[:,:,:,:,0:3]*delta
            # train G
            errg, errMSE, errCons, errAdv, learn_Rate, _ = session.run([g_loss, mse_loss, conservationLoss, advLoss, learning_rate, g_optim], {inputGeom: inputsTrain, realVelField: outputsTrain})   
            
            if ganFlag:
                # train D
                errd, errdR, errdF, _ = session.run([d_loss, disc_out_real, disc_out_fake, d_optim], {inputGeom: inputsTrain, realVelField: outputsTrain})
            
            stdout.write("\rLR: {%.4e} Epoch [%4d/%4d] [%4d/%4d]: g_loss: %.4f (mse_loss: %.4f consLoss: %.4f advLoss: %.4f) d_loss: %.4f dR: %.4f dF: %.4f" % (learn_Rate, epochNum+1, epochs + initEpoch, iterNum+1, iterations_train, errg, errMSE, errCons, errAdv, errd, np.mean(errdR), np.mean(errdF)))
            stdout.flush()

            trainingMSE[iterNum]=errg
#            if errMSE>100*oldTrainMSE:
#                print('Training has become unstable, halving the learning rate')
#                lrFactor=lrFactor/2
#                learnRate = args.learnRate*(0.5**(epochNum/args.epoch_step))*lrFactor 
#                session.run(tf.assign(learning_rate, learnRate))
            oldTrainMSE=errMSE
        # end of epoch
        stdout.write("\n")
        print('Mean metrics: GLoss: %.4f' %(np.mean(trainingMSE)))
        
        epochsMSE[epochNum-initEpoch]=np.mean(trainingMSE)

        # run validation every XX epochs
        if np.mod(epochNum+1, 10)==0 or epochNum==0:
            os.mkdir(f'{trainOutputDir}/epoch-{epochNum+1}')
            valMSE=np.zeros(numValIterations)
            
            for n in range(numValIterations):

                numBatches=np.floor(np.max(valImageIDs)/batch_size)
                index = np.mod(n, numBatches)
                beg = index * batch_size
                end = (index + 1) * batch_size
                if nDims == 2:
                    if outputType == 'k':
                        inputsVal, outputsVal = loadDatasetReg(n, batch_size, valImageIDs, img_width, img_height, path, 'validation', numOutputs, numInputs, inputType, outputVec)
                    else: 
                        inputsVal, outputsVal = loadDataset(n, batch_size, valImageIDs, img_width, img_height, path, 'validation', numOutputs, numInputs, inputType, outputType)
                elif nDims == 3:
                    inputsVal = valInputsDataset[int(beg):int(end)]
                    outputsVal = valOutputsDataset[int(beg):int(end)]
                if outputType == 'velP':
                    if nDims==2:
                        outputsVal[:,:,:,-1] = (outputsVal[:,:,:,-1]+gamma)/beta
                    elif nDims == 3:
                        outputsVal[:,:,:,:,-1] = (outputsVal[:,:,:,:,-1]+gamma)/beta
                elif outputType == 'vel':
                    if nDims==2:
                        outputsVal[:,:,:,0:2] = outputsVal[:,:,:,0:2]*delta
                    elif nDims == 3:
                        outputsVal[:,:,:,:,0:3] = outputsVal[:,:,:,:,0:3]*delta
                predictsVal, errvg = session.run([predVelField, g_loss], {inputGeom: inputsVal, realVelField: outputsVal})  
                predictsVal=np.array(predictsVal)
                valMSE[n]=errvg
                if outputType == 'velP':
                    if nDims==2:
                        predictsVal[:,:,:,-1] = predictsVal[:,:,:,-1]*beta-gamma
                        outputsVal[:,:,:,-1] = outputsVal[:,:,:,-1]*beta-gamma
                    elif nDims == 3:
                        predictsVal[:,:,:,:,-1] = predictsVal[:,:,:,:,-1]*beta-gamma
                        outputsVal[:,:,:,:,-1] = outputsVal[:,:,:,:,-1]*beta-gamma
                elif outputType == 'vel':
                    if nDims==2:
                        predictsVal[:,:,:,0:2] = predictsVal[:,:,:,0:2]/delta
                        outputsVal[:,:,:,0:2] = outputsVal[:,:,:,0:2]/delta
                    elif nDims == 3:
                        predictsVal[:,:,:,:,0:3] = predictsVal[:,:,:,:,0:3]/delta
                        outputsVal[:,:,:,:,0:3] = outputsVal[:,:,:,:,0:3]/delta
                if valPlotFlag and np.mod(epochNum+1, 100)==0  or epochNum==0:

                    
                    for i in range(batch_size):
                        # TODO change this to save  files for reading in matlab
                        if outputType == 'vel':
                            if nDims ==2:
                                domainDictPred={'solid':inputsVal[i,:,:,0], 'velX': predictsVal[i,:,:,0], 'velY': predictsVal[i,:,:,1]}
                                domainDictReal={'solid':inputsVal[i,:,:,0], 'velX': outputsVal[i,:,:,0], 'velY': outputsVal[i,:,:,1]}
                            elif nDims ==3:
                                domainDictPred={'solid':inputsVal[i,:,:,:,0], 'velX': predictsVal[i,:,:,:,0], 'velY': predictsVal[i,:,:,:,1], 'velZ': predictsVal[i,:,:,:,2]}
                                domainDictReal={'solid':inputsVal[i,:,:,:,0], 'velX': outputsVal[i,:,:,:,0], 'velY': outputsVal[i,:,:,:,1], 'velZ': outputsVal[i,:,:,:,2]}
                        elif outputType == 'fq':
                            domainDictPred={'solid':inputsVal[i], 'fq': predictsVal[i]}
                            domainDictReal={'solid':inputsVal[i], 'fq': outputsVal[i]}
                        elif outputType == 'velP':
                            if nDims ==2:
                                domainDictPred={'solid':inputsVal[i,:,:,0], 'velX': predictsVal[i,:,:,0], 'velY': predictsVal[i,:,:,1], 'density': predictsVal[i,:,:,2]}
                                domainDictReal={'solid':inputsVal[i,:,:,0], 'velX': outputsVal[i,:,:,0], 'velY': outputsVal[i,:,:,1], 'density': outputsVal[i,:,:,2]}
                            elif nDims ==3:
                                domainDictPred={'solid':inputsVal[i,:,:,:,0], 'velX': predictsVal[i,:,:,:,0], 'velY': predictsVal[i,:,:,:,1], 'velZ': predictsVal[i,:,:,:,2], 'density': predictsVal[i,:,:,:,3]}
                                domainDictReal={'solid':inputsVal[i,:,:,:,0], 'velX': outputsVal[i,:,:,:,0], 'velY': outputsVal[i,:,:,:,1], 'velZ': outputsVal[i,:,:,:,2], 'density': outputsVal[i,:,:,:,3]}
                        elif outputType == 'P':
                            if nDims ==2:
                                domainDictPred={'solid':inputsVal[i,:,:,0], 'P': predictsVal[i,:,:,0]}
                                domainDictReal={'solid':inputsVal[i,:,:,0], 'P': outputsVal[i,:,:,0]}
                            elif nDims ==3:
                                domainDictPred={'solid':inputsVal[i,:,:,:,0], 'P': predictsVal[i,:,:,:,0]}
                                domainDictReal={'solid':inputsVal[i,:,:,:,0], 'P': outputsVal[i,:,:,:,0]}
                        elif outputType == 'k':
                            domainDictPred={'perm':inputsVal[i]}
                            domainDictReal={'perm':outputsVal[i]}
                        io.savemat(f'{trainOutputDir}/epoch-{epochNum+1}/{n+1:04}-{i}-pred.mat', domainDictPred, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')
                        io.savemat(f'{trainOutputDir}/epoch-{epochNum+1}/{n+1:04}-{i}-real.mat', domainDictReal, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')
                        
#                        plt.figure()
#                        plt.imshow(inputsVal[i,:,:,0],origin='lower')
#                        plt.quiver(predictsVal[i,:,:,0], predictsVal[i,:,:,1], color='g')
#                        plt.savefig(f'{trainOutputDir}/epoch-{epochNum+1}/{n+1:04}-{valBatchIDs[i]}-pred.png',dpi=600)

#                        plt.figure()
#                        plt.imshow(inputsVal[i,:,:,0],origin='lower')
#                        plt.quiver(outputsVal[i,:,:,0], outputsVal[i,:,:,1], color='g')
#                        plt.savefig(f'{trainOutputDir}/epoch-{epochNum+1}/{n+1:04}-{valBatchIDs[i]}-real.png',dpi=600)
                        
        #            plt.figure(3)
        #            plt.imshow(inputsTrain[1,:,:,0],origin='lower')
        #            plt.quiver(predictsTrain[1,:,:,0], predictsTrain[1,:,:,1])
        #            plt.show()
                    #imsave(f'{trainOutputDir}/epoch-{epochNum+1}/{n+1:04}.tif', np.array(predictsVal.astype('float32'), dtype='float32'))
                
                stdout.write("\rValidation: [%4d/%4d] MSE: %.4f" % (n+1, numValIterations, errvg))
                stdout.flush()
            stdout.write("\n")
            end=timer()

            meanValMSE=np.mean(valMSE)
            
            print('Mean Val Loss: %.4f Epoch Time: %.2f' %(meanValMSE, end-start))

            if meanValMSE<oldMeanValMSE:
                save_path = saver.save(session, f"{trainingDir}/epoch-{epochNum+1}-mse{meanValMSE}.ckpt")
                if ganFlag:
                    save_path = saverD.save(session, f"{trainingDir}/epoch-{epochNum+1}-mse{meanValMSE}-Disc.ckpt")
                oldMeanValMSE=meanValMSE
if testFlag:
    sample_files=glob.glob(args.testInputs+'/*.mat')
    outdir=args.testInputs+'/CNNOutputs-'+restore.split('/')[-1].split('.')[0]+'-'+name
    os.makedirs(outdir, exist_ok=True)
    for sampleFile in sample_files:
        fileName=sampleFile.split('/')[-1].split('.')[0] 
        print('Estimating Velocity Field for file: ' + sampleFile)
        arrays = {}
        f = h5py.File(sampleFile)
        for k, v in f.items():
            arrays[k] = np.array(v)
        img=arrays['temp'] 
        img = np.array(img, dtype='uint8')
        img=np.expand_dims(img,nDims)
        img=np.expand_dims(img,0)
        img=np.repeat(img, args.batch_size,0)
        pred = session.run([predVelField], {inputGeom: img}) 
        pred=np.array(pred)
        pred=np.squeeze(pred)
        if outputType == 'velP':
            if nDims==2:
                pred[:,:,nDims] = pred[:,:,nDims]*beta-gamma
            elif nDims == 3:
                pred[:,:,:,nDims] = pred[:,:,:,nDims]*beta-gamma
        elif outputType == 'vel':
            if nDims==2:
                pred[:,:,0:2] = pred[:,:,0:2]/delta
            elif nDims == 3:
                pred[:,:,:,0:3] = pred[:,:,:,0:3]/delta
        if nDims ==2:
            if outputType == 'vel':
                domainDictPred={'solid':img[0,:,:,0], 'velX': pred[:,:,0], 'velY': pred[:,:,1]}
            elif outputType == 'velP':
                domainDictPred={'solid':img[0,:,:,0], 'velX': pred[:,:,0], 'velY': pred[:,:,1], 'density': pred[:,:,2]}
            elif outputType == 'P':
                domainDictPred={'solid':img[0,:,:,0], 'density': pred[:,:]}
        elif nDims ==3:
            if outputType == 'vel':
                domainDictPred={'solid':img[0,:,:,:,0], 'velX': pred[:,:,:,0], 'velY': pred[:,:,:,1], 'velZ': pred[:,:,:,2]}
            elif outputType == 'velP':
                domainDictPred={'solid':img[0,:,:,:,0], 'velX': pred[:,:,:,0], 'velY': pred[:,:,:,1], 'velZ': pred[:,:,:,2], 'density': pred[:,:,:,3]}
            elif outputType == 'P':
                domainDictPred={'solid':img[0,:,:,:,0], 'density': pred[:,:,:]}
        io.savemat(f'{outdir}/{fileName}-pred.mat', domainDictPred, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')
#    predictsTrain = session.run([predVelField], {inputGeom: inputsTrain, realVelField: outputsTrain})  
#    predictsTrain=np.array(predictsTrain)
#    predictsTrain=np.squeeze(predictsTrain)
#    
#    temp=psnr(predictsTrain, outputsTrain, data_range = 0.1)
#    
#    plt.figure(2)
#    plt.imshow(inputsTrain[1,:,:,0],origin='lower')
#    plt.quiver(outputsTrain[1,:,:,0], outputsTrain[1,:,:,1])

#    
#    plt.figure(3)
#    plt.imshow(inputsTrain[1,:,:,0],origin='lower')
#    plt.quiver(predictsTrain[1,:,:,0], predictsTrain[1,:,:,1])
#    plt.show()
##        
#        
#        
#        
