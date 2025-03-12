import torch.multiprocessing as mp
import argparse
from Models import modelpool
from Preprocess import datapool
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
from ImageNet.train import main_worker
import torch.nn as nn
import os

def WFWC(test_dataloader, model, device, time_len, c, r):
    model.eval()
    test_acc = 0
    test_samples = 0
    time=0.
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):
            voltage_change=0.
            voltage_total=0.
            max_old=0.
            max_new=0.
            counter=0.
            flag=0.
            img = img.cuda(device)
            label = label.cuda(device)
            for t in range(time_len):
                voltage_change = model(img)
                voltage_total=voltage_total+voltage_change
                max_new=voltage_total.argmax(dim=1)
                if t==0 :
                    top10_relu=torch.where(torch.topk(voltage_total,10).values>0.,torch.topk(voltage_total,10).values,0.)
                    top10_relu_quadratic_sum=top10_relu.norm(2) ** 2
                    rate=(voltage_total.max() ** 2)/top10_relu_quadratic_sum
                    if rate>r:
                        result=max_new
                        time=time+1
                        flag=1.
                        break
                    else:
                        max_old=max_new
                        counter=counter+1
                else:
                    if (max_new==max_old) and (voltage_change.argmax(dim=1) == max_old):
                        counter=counter+1
                        if counter>=c :             
                            result=max_new
                            time=time+t+1
                            flag=1.
                            break
                    else:
                        counter=1.
                        max_old=max_new 
                  
            if flag==0.:
                result= voltage_total.argmax(dim=1)
                time=time+time_len                  
            test_acc += (result == label).float().sum().item()
            test_samples += label.numel()
            reset_net(model)
    test_acc /= test_samples
    print('test accuracy:',test_acc)
    print('total time steps for test samples:',time) 
    print('number of test samples:',test_samples)
    print('avarage time steps:',time/test_samples) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--t', default=16, type=int, help='T')
    parser.add_argument('--l', default=16, type=int, help='L')
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--c', default=16, type=int, help='c')
    parser.add_argument('--r', default=0.5, type=float, help='r')
    args = parser.parse_args()
    
    seed_all()

    # only ImageNet using multiprocessing,
    if args.gpus > 1:
        if args.data.lower() != 'imagenet':
            AssertionError('Only ImageNet using multiprocessing.')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    else:
        # preparing data
        train, test = datapool(args.data, 1)
        # preparing model
        model = modelpool(args.model, args.data)
        model = replace_maxpool2d_by_avgpool2d(model)
        model = replace_activation_by_floor(model, t=args.l)
        state_dict = torch.load(os.path.join('./model/' + args.id + '.pth'), map_location=torch.device(args.device))
        keys = list(state_dict.keys())
        for k in keys:
            if "relu.thresh" in k:
                state_dict[k[:-11]+'act.up'] = state_dict.pop(k)
            elif "thresh" in k:
                state_dict[k[:-6]+'up'] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model = replace_activation_by_neuron(model)
        model.to(args.device)
        acc = WFWC(test, model, args.device, args.t, args.c, args.r)

