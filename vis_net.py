from utils.summary import summary
from model import CGNet

model = CGNet.Context_Guided_Network(19, M=3, N=21)
model.cuda()
summary(model,(3,640, 640))
