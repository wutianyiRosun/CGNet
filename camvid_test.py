###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn 
# Copyright (c) 2018
###########################################################################
import os
import time
import torch
import pickle
import timeit
import random
import numpy as np
import torch.nn as nn
from torch.utils import data
from argparse import ArgumentParser
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
#user
from model import CGNet
from utils.metric import get_iou
from utils.modeltools import netParams
from utils.loss import CrossEntropyLoss2d
from utils.convert_state import convert_state_dict
from  dataset.camvid import CamVidDataSet,CamVidValDataSet, CamVidTrainInform

def test(args, test_loader, model, criterion):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
      criterion: loss function
    return: class IoU and mean IoU
    """
    #evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
   
    data_list=[]
    for i, (input, label, size, name) in enumerate(test_loader):
        input_var = Variable(input, volatile=True).cuda()
        output = model(input_var)
        output= output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype = np.uint8)
        output= output.transpose(1,2,0)
        output= np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append( [gt.flatten(), output.flatten()])
    meanIoU, per_class_iu= get_iou(data_list, args.classes)
    return meanIoU, per_class_iu

def test_model(args):
    """
    main function for testing 
    args:
       args: global arguments
    """
    print("=====> Check if the cached file exists ")
    if not os.path.isfile(args.inform_data_file):
        print("%s is not found" %(args.inform_data_file))
        dataCollect = CamVidTrainInform(args.data_dir, args.classes, train_set_file= args.dataset_list, 
                                        inform_data_file = args.inform_data_file) #collect mean std, weigth_class information
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        print("%s exists" %(args.inform_data_file))
        datas = pickle.load(open(args.inform_data_file, "rb"))
    
    print(args)
    global network_type
     
    if args.cuda:
        print("=====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
    cudnn.enabled = True

    M = args.M
    N = args.N
    model = CGNet.Context_Guided_Network(classes= args.classes, M= M, N= N)
    network_type="CGNet"
    print("=====> current architeture:  CGNet_M%sN%s"%(M, N))
    total_paramters = netParams(model)
    print("the number of parameters: " + str(total_paramters))
    print("data['classWeights']: ", datas['classWeights'])
    weight = torch.from_numpy(datas['classWeights'])
    print("=====> Dataset statistics")
    print("mean and std: ", datas['mean'], datas['std'])
    
    # define optimization criteria
    criteria = CrossEntropyLoss2d(weight, args.ignore_label)
    if args.cuda:
        model = model.cuda()
        criteria = criteria.cuda()
    
    #load test set
    train_transform= transforms.Compose([
        transforms.ToTensor()])
    testLoader = data.DataLoader(CamVidValDataSet(args.data_dir, args.test_data_list,f_scale=1,  mean= datas['mean']),
                                  batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=====> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #model.load_state_dict(convert_state_dict(checkpoint['model']))
            model.load_state_dict(checkpoint['model'])
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark= True

    print("=====> beginning test")
    print("length of test set:", len(testLoader))
    mIOU_val, per_class_iu = test(args, testLoader, model, criteria)
    print(mIOU_val)
    print(per_class_iu)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type = str, default = "CGNet", help = "model name: Context Guided Network")
    parser.add_argument('--dataset', type = str, default = "camvid", help = "camvid or cityscapes")
    parser.add_argument('--ignore_label', type = int, default = 11, help = "nClass")
    parser.add_argument('--data_dir', default = "/home/wty/AllDataSet/CamVid", help = "data directory")
    parser.add_argument('--test_data_list', default = "./dataset/list/CamVid/camvid_test_list.txt", help= "data directory")
    parser.add_argument('--scaleIn', type = int, default = 1, help = "for input image, default is 1, keep fixed size")  
    parser.add_argument('--num_workers', type = int, default = 1, help = "the number of parallel threads") 
    parser.add_argument('--batch_size', type = int, default = 1, help = "the batch size is set to 1 when testing")
    parser.add_argument('--resume', type = str, default = './checkpoint/camvid/CGNet_M3N21bs8gpu1_ontrainval/model_800.pth', 
                         help = "use this file to load last checkpoint for testing")
    parser.add_argument('--classes', type = int, default = 11, 
                         help = "the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--inform_data_file', default = "./dataset/wtfile/camvid_inform.pkl", 
                         help = "storing classes weights, mean and std")
    parser.add_argument('--M', type = int, default = 3,  help = "the number of block in stage 2")
    parser.add_argument('--N', type = int, default = 21, help = "the number of block in stage 3")
    parser.add_argument('--cuda', type = bool, default = True, help = "running on CPU or GPU")
    parser.add_argument("--gpus", type = str, default = "0",  help = "gpu ids (default: 0)")
    parser.add_argument("--gpu_nums",  type = int, default=1 , help="the number of gpu")
    
    args = parser.parse_args()
    test_model(args)


