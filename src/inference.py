import cv2
import os
import datetime
import numpy as np
from model import SalEMA
from args import get_inference_parser
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from data_loader import Poles, Equator


"""
Before inferring check:
pt_model,
dst
"""
"""
dataset_name = "Equator"
dataset_name = "Poles"
"""
CLIP_LENGTH = 10
EMA_LOC = 30 # 30 is the bottleneck

frame_size = (256, 256)
# Destination for predictions:
frames_path = "./frame"
gt_path = "./gt"

params = {'batch_size': 1,
          'num_workers': 4,
          'pin_memory': True}

def main(args):


    dst = os.path.join(args.dst, "{}_predictions".format(args.pt_model.replace(".pt", "")))
    print("Output directory {}".format(dst))

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1

    if args.dataset == "Equator" or args.dataset == "Poles" or args.dataset == "other" :
        print("Commencing inference for dataset {}".format(args.dataset))
        dataset = TEST(
            root_path = args.src,
            clip_length = CLIP_LENGTH,
            resolution = frame_size)
        video_name_list = dataset.video_names() #match an index to the sample video name
    else :
        print('dataset not defined')
        exit()


    print("Size of test set is {}".format(len(dataset)))

    loader = data.DataLoader(dataset, **params)

    # =================================================
    # ================= Load Model ====================

    # Using same kernel size as they do in the DHF1K paper
    # Amaia uses default hidden size 128
    # input size is 1 since we have grayscale images

    if "EMA" in args.pt_model:
        if "poles" in args.pt_model:
            model = SalEMA.Poles_EMA(alpha=args.alpha, ema_loc=EMA_LOC)
        elif "equator" in args.pt_model:
            model = SalEMA.Equator_EAM(alpha=args.alpha, ema_loc=EMA_LOC)
        
        load_model(args.pt_model, model)
        print("Pre-trained model {} loaded succesfully".format(args.pt_model))

        TEMPORAL = True
        print("Alpha tuned to {}".format(model.alpha))

    else:
        print("Your model was not recognized not (pole or equator), check the name of the model and try again.")
        exit()

    dtype = torch.FloatTensor
    if args.use_gpu:
        assert torch.cuda.is_available(), \
            "CUDA is not available in your machine"
        cudnn.benchmark = True 
        model = model.cuda()
        dtype = torch.cuda.FloatTensor


    # ================== Inference =====================

    if not os.path.exists(dst):
        os.mkdir(dst)
    else:
        print(" you are about to write on an existing folder {}. If this is not intentional cancel now.".format(dst))

    # switch to evaluate mode
    model.eval()

    for i, video in enumerate(loader):

        count = 0
        state = None # Initially no hidden state

        elif args.dataset == "Poles" or args.dataset == "Equator":

            video_dst = os.path.join(dst, video_name_list[i])
            # if "shooting" in video_dst:
            #     # CUDA error: out of memory is encountered whenever inference reaches that vid.
            #     continue
            print("Destination: {}".format(video_dst))
            if not os.path.exists(video_dst):
                os.mkdir(video_dst)

            for j, (clip, _) in enumerate(video):
                clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)

                for idx in range(clip.size()[0]):
                    # Compute output
                    if TEMPORAL:
                        state, saliency_map = model.forward(input_ = clip[idx], prev_state = state)
                    else:
                        saliency_map = model.forward(input_ = clip[idx])

                    
                    saliency_map = saliency_map.squeeze(0)
    
                    post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                    utils.save_image(post_process_saliency_map, os.path.join(video_dst, "{}.png".format(str(count).zfill(4))))
                    if count == 0:
                        print("The final destination is {}".format(os.path.join(video_dst)))
                    count+=1
                if TEMPORAL:
                    state = repackage_hidden(state)
            print("Video {} done".format(i+int(args.start)))

def load_model(pt_model, new_model):

    temp = torch.load(pt_model)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    new_model.load_state_dict(checkpoint, strict=True)

    return new_model

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

if __name__ == '__main__':
    parser = get_inference_parser()
    args = parser.parse_args()
    main(args)