###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn 
# Copyright (c) 2018
###########################################################################
import os
import time
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from argparse import ArgumentParser
#user
from model import CGNet  # network
from utils.metric import get_iou
from utils.loss import CrossEntropyLoss2d  # loss function
from utils.convert_state import convert_state_dict
from utils.colorize_mask import cityscapes_colorize_mask
from  dataset.cityscapes import CityscapesTestDataSet, CityscapesTrainInform  # dataset


def test(args, test_loader, model):
    """
    args:
      test_loader: loaded for test set
      model: model
      criterion: loss function
    return: IoU class, and mean IoU
    """
    #evaluation mode
    model.eval()
    total_batches = len(test_loader) 
    for i, (input, size, name) in enumerate(test_loader):
        input_var = Variable(input, volatile=True).cuda()
        output = model(input_var)
        # save seg image
        output= output.cpu().data[0].numpy()  # 1xCxHxW ---> CxHxW
        output= output.transpose(1,2,0) # CxHxW --> HxWxC
        output= np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_color = cityscapes_colorize_mask(output)
        output = Image.fromarray(output)
        #output.save( "%s/%s.png " % (args.save_seg_dir, name[0]) )
        output_color.save( "%s/%s_color.png" % (args.save_seg_dir, name[0]))


def test_func(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)
    global network_type

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")
    
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
    
    print('=====> checking if processed cached_data_file exists')
    if not os.path.isfile(args.inform_data_file):
        dataCollect = CityscapesTrainInform(args.data_dir, args.classes, train_set_file = args.dataset_list, 
                                            inform_data_file = args.inform_data_file) #collect mean std, weigth_class information
        data= dataCollect.collectDataAndSave()
        if data is  None:
            print("error while pickling data, please check")
            exit(-1)
    else:
        data = pickle.load(open(args.inform_data_file, "rb"))
    M = args.M
    N = args.N
    
    model = CGNet.Context_Guided_Network(classes= args.classes, M= M, N= N)
    network_type="CGNet"
    print("Arch:  CGNet")
    # define optimization criteria
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    if args.cuda:
        weight = weight.cuda()
    criteria = CrossEntropyLoss2d(weight) #weight

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        criteria = criteria.cuda()
        cudnn.benchmark = True

    print('Dataset statistics')
    print('mean and std: ', data['mean'], data['std'])
    print('classWeights: ', data['classWeights'])

    if args.save_seg_dir:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # validation set
    testLoader = torch.utils.data.DataLoader( CityscapesTestDataSet(args.data_dir, args.test_data_list, mean = data['mean']),
                                             batch_size = 1, shuffle = False, num_workers = args.num_workers, pin_memory = True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=====> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #model.load_state_dict(checkpoint['model'])
            model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))
    
    print("=====> beginning testing")
    print("test set length: ", len(testLoader))
    test(args, testLoader, model)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default = "CGNet", help = "model name: Context Guided Network (CGNet)")
    parser.add_argument('--data_dir', default = "/home/wty/AllDataSet/Cityscapes", help = "data directory")
    parser.add_argument('--dataset_list', default = "cityscapes_trainval_list.txt",
                        help = "train and val data, for computing the ratio of all classes, mean and std")
    parser.add_argument('--test_data_list', default = "./dataset/list/Cityscapes/cityscapes_test_list.txt", help = "test set")
    parser.add_argument('--scaleIn', type = int, default = 1, help = "rescale input image, default is 1, keep fixed size")  
    parser.add_argument('--num_workers', type = int, default= 1, help = "the number of parallel threads") 
    parser.add_argument('--batch_size', type = int, default = 1, help=" the batch_size is set to 1 when evaluating or testing") 
    parser.add_argument('--resume', type = str, default = "./checkpoint/cityscapes/CGNet_M3N21bs16gpu2_ontrain/model_1.pth", 
                        help = "use the file to load last checkpoint for evaluating or testing ")
    parser.add_argument('--classes', type = int, default = 19, 
                        help = "the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--inform_data_file', default = "./dataset/wtfile/cityscapes_inform.pkl", 
                        help = "storing the classes weights, mean and std")
    parser.add_argument('--cuda', default = True, help = "run on CPU or GPU")
    parser.add_argument('--M', type = int, default = 3, help = "the number of blocks in stage 2")
    parser.add_argument('--N', type = int, default = 21, help = "the number of blocks in stage 3")
    parser.add_argument('--save_seg_dir', type = str, default = "./result/cityscapes/test/", help = "saving path of prediction result")
    parser.add_argument("--gpus", default = "7", type = str, help = "gpu ids (default: 2)")

    test_func(parser.parse_args())

