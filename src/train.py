import cv2
import os
import datetime
import numpy as np
from model import SalEMA
from args import get_training_parser
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from data_loader import DHF1K_frames, Hollywood_frames



"""
Be sure to check the name of args.new_model before running
"""
frame_size = (256, 256) # original shape is (360, 640, 3)

decay_rate = 0.1
momentum = 0.9
weight_decay = 1e-4

plot_every = 1
clip_length = 10

LEARN_ALPHA_ONLY = False

params = {'batch_size': 1, # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}


def main(args, params = params):

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    if args.dataset == "Poles" :
        print("Commencing training on dataset {}".format(args.dataset))
        train_set = Poles(
            root_path = args.src,
            load_gt = True,
            number_of_videos = int(args.end),
            starting_video = int(args.start),
            clip_length = clip_length,
            resolution = frame_size,
            val_perc = args.val_perc,
            split = "train")
        print("Size of train set is {}".format(len(train_set)))
        train_loader = data.DataLoader(train_set, **params)

        if args.val_perc > 0:
            val_set = Poles(
                root_path = args.src,
                load_gt = True,
                number_of_videos = int(args.end),
                starting_video = int(args.start),
                clip_length = clip_length,
                resolution = frame_size,
                val_perc = args.val_perc,
                split = "validation")
            print("Size of validation set is {}".format(len(val_set)))
            val_loader = data.DataLoader(val_set, **params)

    if  args.dataset == "Equator":
        print("Commencing training on dataset {}".format(args.dataset))
        train_set = Equator(
            root_path = args.src,
            load_gt = True,
            number_of_videos = int(args.end),
            starting_video = int(args.start),
            clip_length = clip_length,
            resolution = frame_size,
            val_perc = args.val_perc,
            split = "train")
        print("Size of train set is {}".format(len(train_set)))
        train_loader = data.DataLoader(train_set, **params)

        if args.val_perc > 0:
            val_set = Equator(
                root_path = args.src,
                load_gt = True,
                number_of_videos = int(args.end),
                starting_video = int(args.start),
                clip_length = clip_length,
                resolution = frame_size,
                val_perc = args.val_perc,
                split = "validation")
            print("Size of validation set is {}".format(len(val_set)))
            val_loader = data.DataLoader(val_set, **params)


    else:
        print('Your model was not recognized. Check the name again.')
        exit()
    # =================================================
    # ================ Define Model ===================

    # The seed pertains to initializing the weights with a normal distribution
    # Using brute force for 100 seeds I found the number 65 to provide a good starting point (one that looks close to a saliency map predicted by the original SalGAN)
    temporal = True

    elif 'EMA' in args.new_model:

        if 'Poles' in args.new_model:
            model = SalEMA.Poles_EMA(alpha=None, ema_loc=args.ema_loc)
            print("Initialized {} with residual set to {} and dropout set to {}".format(args.new_model))
        elif 'Equator' in args.new_model:
            model = SalEMA.Equator_EMA(alpha=None, ema_loc=args.ema_loc)
            print("Initialized {} with residual set to {} and dropout set to {}".format(args.new_model)) 
    else:
        print("Your model was not recognized, check the name of the model and try again.")
        exit()
    criterion = nn.BCELoss()



    optimizer = torch.optim.Adam([
        {'params':model.salgan.parameters() , 'lr': args.lr, 'weight_decay':weight_decay},
        {'params':model.alpha, 'lr': 0.1}])
    if LEARN_ALPHA_ONLY:
        optimizer = torch.optim.Adam([{'params':[model.alpha]}], 0.1)

    
    # Load an entire pretrained model
    checkpoint = load_weights(model, args.pt_model)
    model.load_state_dict(checkpoint, strict=False)
    start_epoch = torch.load(args.pt_model, map_location='cpu')['epoch']
   

    print("Model loaded, commencing training from epoch {}".format(start_epoch))

    dtype = torch.FloatTensor
    if args.use_gpu == 'parallel' or args.use_gpu == 'gpu':
        assert torch.cuda.is_available(), \
            "CUDA is not available in your machine"

        if args.use_gpu == 'parallel':
            model = nn.DataParallel(model).cuda()
        elif args.use_gpu == 'gpu':
            model = model.cuda()
        dtype = torch.cuda.FloatTensor
        cudnn.benchmark = True 
        criterion = criterion.cuda()



    train_losses = []
    val_losses = []
    starting_time = datetime.datetime.now().replace(microsecond=0)
    print("Training started at : {}".format(starting_time))

    n_iter = 0


    for epoch in range(start_epoch, args.epochs+1):

        try:

            train_loss, n_iter, optimizer = train(train_loader, model, criterion, optimizer, epoch, n_iter, args.use_gpu, temporal, dtype)

            print("Epoch {}/{} done with train loss {}\n".format(epoch, args.epochs, train_loss))

            if args.val_perc > 0:
                print("Running validation..")
                val_loss = validate(val_loader, model, criterion, epoch, temporal, dtype)
                print("Validation loss: {}".format(val_loss))

            if epoch % plot_every == 0:
                train_losses.append(train_loss.cpu())
                if args.val_perc > 0:
                    val_losses.append(val_loss.cpu())

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.cpu().state_dict(),
                'optimizer' : optimizer.state_dict()
                }, args.new_model+".pt")

            if args.use_gpu == 'parallel':
                model = nn.DataParallel(model).cuda()
            elif args.use_gpu == 'gpu':
                model = model.cuda()
            else:
                pass


        except RuntimeError:
            print("A memory error was encountered. Further training aborted.")
            epoch = epoch - 1
            break

    print("Training of {} started at {} and finished at : {} \n Now saving..".format(args.new_model, starting_time, datetime.datetime.now().replace(microsecond=0)))

    # If I try saving in regular intervals I have to move the model to CPU and back to GPU.
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.cpu().state_dict(),
        'optimizer' : optimizer.state_dict()
        }, args.new_model+".pt")



    if args.val_perc > 0:
        to_plot = {
            'epoch_ticks': list(range(start_epoch, args.epochs+1, plot_every)),
            'train_losses': train_losses,
            'val_losses': val_losses
            }
        with open('to_plot.pkl', 'wb') as handle:
            pickle.dump(to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)


mean = lambda x : sum(x)/len(x)


def adjust_learning_rate(optimizer, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (decay_rate ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_weights(model, pt_model, device='cpu'):
    # Load stored model:
    temp = torch.load(pt_model, map_location=device)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    return checkpoint

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(train_loader, model, criterion, optimizer, epoch, n_iter, use_gpu, double, thaw, temporal, dtype):


    # Switch to train mode
    model.train()

    video_losses = []
    print("Now commencing epoch {}".format(epoch))
    for i, video in enumerate(train_loader):

        #print(type(video))
        accumulated_losses = []
        start = datetime.datetime.now().replace(microsecond=0)
        print("Number of clips for video {} : {}".format(i,len(video)))
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            n_iter+=j

            optimizer.zero_grad()

            clip = Variable(clip.type(dtype).transpose(0,1))
            gtruths = Variable(gtruths.type(dtype).transpose(0,1))

            if temporal :

                loss = 0
                for idx in range(clip.size()[0]):
                    #print(clip[idx].size())

                    # Compute output
                    state, saliency_map = model.forward(input_ = clip[idx], prev_state = state) # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones

                    saliency_map = saliency_map.squeeze(0) # Target is 3 dimensional (grayscale image)
                    if saliency_map.size() != gtruths[idx].size():
                        #print(saliency_map.size())
                        #print(gtruths[idx].size())
                        a, b, c, _ = saliency_map.size()
                        saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3) #because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2


                    # Apply sigmoid before visualization
                    # logits will be whatever you have to rescale this

                    # Compute loss
                    loss = loss + criterion(saliency_map, gtruths[idx])

                # Keep score
                accumulated_losses.append(loss.data)

                # Compute gradient
                loss.backward()


                # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
                nn.utils.clip_grad_norm_(model.parameters(), 10*clip.size()[0])

                # Update parameters
                optimizer.step()

                # Repackage to avoid backpropagating further through time
                state = repackage_hidden(state)




            # Visualize some of the data
            if i%100==0 and j == 5:



                post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                utils.save_image(post_process_saliency_map, "./log/smap{}_epoch{}.png".format(i, epoch))

                if epoch == 1:
                    print(saliency_map.max())
                    print(saliency_map.min())
                    print(gtruths[idx].max())
                    print(gtruths[idx].min())
                    print(post_process_saliency_map.max())
                    print(post_process_saliency_map.min())
                    utils.save_image(gtruths[idx], "./log/gt{}.png".format(i))


        end = datetime.datetime.now().replace(microsecond=0)
        print('Epoch: {}\tVideo: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, i, mean(accumulated_losses), end-start))
        video_losses.append(mean(accumulated_losses))

    return (mean(video_losses), n_iter, optimizer)


def validate(val_loader, model, criterion, epoch, temporal, dtype):

    # switch to evaluate mode
    model.eval()

    video_losses = []
    print("Now running validation..")
    for i, video in enumerate(val_loader):
        accumulated_losses = []
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)
            gtruths = Variable(gtruths.type(dtype).transpose(0,1), requires_grad=False)

            loss = 0
            for idx in range(clip.size()[0]):
                # Compute output
                if temporal:
                    state, saliency_map = model.forward(clip[idx], state)
                else:
                    saliency_map = model.forward(clip[idx])

                saliency_map = saliency_map.squeeze(0)

                if saliency_map.size() != gtruths[idx].size():
                    a, b, c, _ = saliency_map.size()
                    saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3) #because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2

                # Compute loss
                loss = loss + criterion(saliency_map, gtruths[idx])

            if temporal:
                state = repackage_hidden(state)

            # Keep score
            accumulated_losses.append(loss.data)

        video_losses.append(mean(accumulated_losses))

    return(mean(video_losses))

if __name__ == '__main__':
    parser = get_training_parser()
    args = parser.parse_args()
    main(args)



