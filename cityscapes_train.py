###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn 
# Copyright (c) 2018
###########################################################################
import os
import time
import torch
import timeit
import pickle
import random
import numpy as np
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from argparse import ArgumentParser
#user
from model import CGNet  # network
from utils.metric import get_iou
from utils.modeltools import netParams
from utils.loss import CrossEntropyLoss2d  # loss function
from utils.convert_state import convert_state_dict
from  dataset.cityscapes import CityscapesDataSet,CityscapesValDataSet, CityscapesTrainInform  # dataset

def val(args, val_loader, model, criterion):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
      criterion: loss function
    return: IoU class, and mean IoU
    """
    #evaluation mode
    model.eval()
    total_batches = len(val_loader)
   
    data_list=[]
    for i, (input, label, size, name) in enumerate(val_loader):
        start_time = time.time()
        input_var = Variable(input, volatile=True).cuda()
        output = model(input_var)
        time_taken = time.time() - start_time
        print("[%d/%d]  time: %.2f" % (i, total_batches, time_taken))
        output= output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype = np.uint8)
        output= output.transpose(1,2,0)
        output= np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append( [gt.flatten(), output.flatten()])

    meanIoU, per_class_iu= get_iou(data_list, args.classes)
    return meanIoU, per_class_iu

def adjust_learning_rate( args, cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    """
    poly learning stategyt
    lr = baselr*(1-iter/max_iter)^power
    """
    cur_iter = cur_epoch*perEpoch_iter + curEpoch_iter
    max_iter=max_epoch*perEpoch_iter
    lr = baselr*pow( (1 - 1.0*cur_iter/max_iter), 0.9)

    return lr


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """
    model.train()
    epoch_loss = []

    data_list=[]
    total_batches = len(train_loader)
    print("=====> the number of iterations per epoch: ", total_batches)
    for iteration, batch in enumerate( train_loader, 0 ):
        lr= adjust_learning_rate( args, cur_epoch = epoch, max_epoch = args.max_epochs, 
                                  curEpoch_iter = iteration, perEpoch_iter = total_batches, baselr = args.lr )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr;
        
        start_time = time.time()
        images, labels, _, _ = batch
        images = Variable( images ).cuda()
        labels = Variable( labels.long() ).cuda()
        output = model( images )
        loss = criterion(output, labels)
        optimizer.zero_grad()  #set the grad to zero
        loss.backward()
        optimizer.step()
        epoch_loss.append( loss.item() )
        time_taken = time.time() - start_time
        
        gt = np.asarray( labels.cpu().data[0].numpy(), dtype = np.uint8 )
        output = output.cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        output = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )

        data_list.append( [gt.flatten(), output.flatten()] )

        print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % ( epoch, args.max_epochs,
              iteration, total_batches, lr,loss.item(), time_taken ) )

    average_epoch_loss_train = sum( epoch_loss ) / len( epoch_loss )
    meanIoU, per_class_iu = get_iou( data_list, args.classes )

    return average_epoch_loss_train, per_class_iu, meanIoU, lr

def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> checking if inform_data_file exists")
    if not os.path.isfile(args.inform_data_file):
        print("%s is not found" %( args.inform_data_file ) )
        dataCollect = CityscapesTrainInform(args.data_dir, args.classes, train_set_file = args.dataset_list, 
                                            inform_data_file = args.inform_data_file) #collect mean std, weigth_class information
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(args.inform_data_file))
        datas = pickle.load( open( args.inform_data_file, "rb") )
    
    print(args)
    global network_type
     
    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    args.seed = random.randint(1, 10000)
    print("====> Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
    
    cudnn.enabled = True
    M = args.M
    N = args.N
    model = CGNet.Context_Guided_Network(classes= args.classes, M= M, N= N)
    network_type="CGNet"
    print("=====> current architeture:  CGNet")
    
    print("=====> computing network parameters")
    total_paramters = netParams(model)
    print("the number of parameters: " + str(total_paramters))
    
    print("data['classWeights']: ", datas['classWeights'])
    print('=====> Dataset statistics')
    print('mean and std: ', datas['mean'], datas['std'])
    
    # define optimization criteria
    weight = torch.from_numpy(datas['classWeights'])
    criteria = CrossEntropyLoss2d(weight)

    if args.cuda:
        criteria = criteria.cuda()
        if torch.cuda.device_count()>1:
            print("torch.cuda.device_count()=",torch.cuda.device_count())
            args.gpu_nums = torch.cuda.device_count()
            model = torch.nn.DataParallel(model).cuda()  #multi-card data parallel
        else:
            print("single GPU for training")
            model = model.cuda()  #1-card data parallel
    
    args.savedir = ( args.savedir + args.dataset + '/'+ network_type +"_M"+ str(M) + 'N' +str(N) + 'bs' 
                    + str(args.batch_size)+ 'gpu' + str(args.gpu_nums)+ "_"+str(args.train_type)+'/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    train_transform= transforms.Compose([
        transforms.ToTensor()])
    trainLoader = data.DataLoader( CityscapesDataSet( args.data_dir, args.train_data_list, crop_size = input_size, scale = args.random_scale, 
                                                      mirror = args.random_mirror, mean = datas['mean'] ),
                                   batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, 
                                   pin_memory = True, drop_last = True )
    valLoader = data.DataLoader( CityscapesValDataSet( args.data_dir, args.val_data_list,f_scale = 1,  mean = datas['mean']),
                                 batch_size = 1, shuffle = True, num_workers = args.num_workers, pin_memory = True, drop_last = True )

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            #model.load_state_dict(convert_state_dict(checkpoint['model']))
            print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))
    
    model.train()
    cudnn.benchmark= True
    
    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val)'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
 
    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        #training
        lossTr, per_class_iu_tr, mIOU_tr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        
        #validation
        if epoch % 50 ==0:
            mIOU_val, per_class_iu = val(args, valLoader, model, criteria)
            # record train information
            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_tr, mIOU_val, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("\nEpoch No.: %d\tTrain Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f\t lr= %.6f" % (epoch,
                   lossTr, mIOU_tr, mIOU_val, lr))
        else:
            # record train information
            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_tr, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("\nEpoch No.: %d\tTrain Loss = %.4f\t mIOU(tr) = %.4f\t lr= %.6f" % (epoch, lossTr, mIOU_tr, lr))
        
        #save the model
        model_file_name = args.savedir +'/model_' + str(epoch + 1) + '.pth'
        state = {"epoch": epoch+1, "model": model.state_dict()}
        if epoch > args.max_epochs - 10 :
            torch.save(state, model_file_name)
        elif not epoch % 20:
            torch.save(state, model_file_name)

    logger.close()

if __name__ == '__main__':
    start = timeit.default_timer()
    parser = ArgumentParser()
    parser.add_argument('--model', default = "CGNet", help = "model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default = "cityscapes", help = "dataset: cityscapes or camvid")
    parser.add_argument('--data_dir', default = "/home/wty/AllDataSet/Cityscapes", help ='data directory')
    parser.add_argument('--dataset_list', default = "cityscapes_trainval_list.txt",
                        help = "train and val data, for computing the ration of all kinds, mean and std")
    parser.add_argument('--train_data_list', default = "./dataset/list/Cityscapes/cityscapes_trainval_list.txt", help = "train set")
    parser.add_argument('--train_type', type = str, default = "ontrainval", 
                         help = "ontrain for training on train set, ontrainval for training on train+val set")
    parser.add_argument('--max_epochs', type = int, default = 350, help = "the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--val_data_list', default = "./dataset/list/Cityscapes/cityscapes_val_list.txt", help = "val set")
    parser.add_argument('--scaleIn', type = int, default = 1, help = "for input image, default is 1, keep fixed size")  
    parser.add_argument('--input_size', type = str, default = "680,680", help = "input size of model") 
    parser.add_argument('--random_mirror', type = bool, default = True, help = "input image random mirror") 
    parser.add_argument('--random_scale', type = bool, default = True, help = "input image resize 0.5 to 2") 
    parser.add_argument('--num_workers', type = int, default = 1, help = " the number of parallel threads") 
    parser.add_argument('--batch_size', type = int, default = 16, help = "the batch size is set to 16 for 2 GPUs")

    parser.add_argument('--lr', type = float, default = 1e-3, help = "initial learning rate")
    parser.add_argument('--savedir', default = "./checkpoint/", help = "directory to save the model snapshot")
    parser.add_argument('--resume', type = str, default = "./checkpoint/cityscapes/CGNet_M3N21bs16gpu2_ontrainval/model_1.pth", 
                         help = "use this file to load last checkpoint for continuing training")  
    parser.add_argument('--classes', type = int, default = 19, 
                         help = "the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--inform_data_file', default = "./dataset/wtfile/cityscapes_inform.pkl", 
                         help = "saving statistic information of the dataset (train+val set), classes weigtht, mean and std")
    parser.add_argument('--M', type = int, default = 3, help = "the number of blocks in stage 2")
    parser.add_argument('--N', type = int, default = 21, help = "the number of blocks in stage 3")
    parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    parser.add_argument('--cuda', type = bool, default = True, help = "running on CPU or GPU")
    parser.add_argument('--gpus', type = str, default = "0,1", help = "default GPU devices (0,1)")
    args = parser.parse_args()
    train_model(args)
    end = timeit.default_timer()
    print("training time:", 1.0*(end-start)/3600)

