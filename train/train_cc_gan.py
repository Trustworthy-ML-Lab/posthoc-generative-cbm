import os
import sys
sys.path.append('.')
from utils.utils import get_dataset, create_image_grid, get_concept_index
import argparse
import numpy as np
from pathlib import Path
import yaml
import torch
from ast import literal_eval
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import nn
from models import cbae_stygan2, clip_pseudolabeler
import torchvision.transforms as transforms
from utils import gan_loss
import itertools
import warnings
import pickle
from torch.utils.tensorboard import SummaryWriter

# warnings.filterwarnings("ignore", category=UserWarning)
import time


def eval_classifier(model, save_image_loc, num_samples=64, dataset='celebahq', device='cuda', set_of_classes=[]):
    batch_size = 16
    for eval_idx in range(num_samples // batch_size):
        with torch.no_grad():
            # Sample noise and labels as generator input
            z = torch.randn((batch_size, model.gen.z_dim), device=device)
            latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)

            # get the concepts
            concepts = model.cbae.enc(latent)

            # generate images with original latent
            gen_imgs_latent = model.gen.synthesis(latent, noise_mode='const')
            # to make it from -1 to 1 range to 0 to 1
            gen_imgs_latent = gen_imgs_latent.mul(0.5).add_(0.5)

        pseudo_label_list = []
        pseudo_probs_list = []
        for c in range(model.n_concepts):
            start, end = get_concept_index(model, c)
            c_predicted_concepts = concepts[:, start:end]
            c_predicted_concepts = c_predicted_concepts.softmax(dim=1)
            values, indices = torch.max(c_predicted_concepts, dim=1)
            pseudo_label_list.append(indices)
            pseudo_probs_list.append(values)

        create_image_grid(gen_imgs_latent, pseudo_label_list, pseudo_probs_list, save_image_loc+"_%d.png" % eval_idx, n_row=4, n_col=batch_size // 4, set_of_classes=set_of_classes, figsize=(20, 20), textwidth=30)

def get_pseudo_concept_loss(model, predicted_concepts, pseudolabel_concepts, pseudolabel_probs, device, pl_prob_thresh=0.1, dataset='color_mnist', ignore_index=250, use_pl_thresh=True):
    concept_loss = 0
    batch_size = predicted_concepts.shape[0]
    if dataset == 'celebahq' or dataset == 'cub' or dataset == 'cub64':
        ### classification with CUB is more difficult and probability values are more varied, so we don't use the threshold
        ### using the threshold leads to lot of nans because some batches don't have enough samples over the threshold
        if dataset == 'cub' or dataset == 'cub64':
            use_pl_thresh = False
        if use_pl_thresh:
            for cdx in range(len(pseudolabel_concepts)):
                ## for predictions with probability lower than threshold, we assign ignore_index and CE loss will not be computed for those predictions
                pseudolabel_concepts[cdx][pseudolabel_probs[cdx] < pl_prob_thresh] = ignore_index
        concepts = [curr_conc.long() for curr_conc in pseudolabel_concepts]
    else:
        raise NotImplementedError('only implemented for color_mnist as of now')

    loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    concept_loss_lst = []
    for c in range(model.n_concepts):
        start,end = get_concept_index(model,c)
        c_predicted_concepts=predicted_concepts[:,start:end]
        c_real_concepts = concepts[c]
        c_concept_loss = loss_ce(c_predicted_concepts, c_real_concepts)
        concept_loss+=c_concept_loss
        concept_loss_lst.append(c_concept_loss)
    return concept_loss, concept_loss_lst



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset",default="celebahq",help="benchmark dataset")
    parser.add_argument("-e", "--expt-name", default="cc_stygan2", help="name for saving images and checkpoint")
    parser.add_argument("-t", "--tensorboard-name", default="clipzs", help="suffix for tensorboard experiment name")
    parser.add_argument("-p", "--pseudo-label", type=str, default='clipzs', help='choice of pseudo-label source: clip zero shot or supervised')
    parser.add_argument("--load-pretrained", action='store_true', default=False, help='whether to load pretrained CC checkpoint from models/checkpoints/.')
    parser.add_argument("--pretrained-load-name", type=str, default='', help='filename to load from models/checkpoints/')
    args = parser.parse_args()
    args.config_file = f"./config/{args.expt_name}/"+args.dataset+".yaml"

    writer = SummaryWriter(f'results/{args.dataset}_{args.expt_name}_{args.tensorboard_name}')

    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")

    use_cuda =  config["train_config"]["use_cuda"] and  torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device= torch.device("cpu")

    ignore_index = 250

    if config["train_config"]["save_model"]:
        save_model_name =  f"{config['dataset']['name']}_{args.expt_name}_{args.tensorboard_name}"

    if config["evaluation"]["save_images"] or config["evaluation"]["save_concept_image"]:
        os.makedirs("generation_checkpoints/", exist_ok=True)
        os.makedirs("images/", exist_ok=True)
        os.makedirs(f"images/{args.expt_name}_{args.tensorboard_name}/", exist_ok=True)
        os.makedirs(f"images/{args.expt_name}_{args.tensorboard_name}/"+config["dataset"]["name"]+"/", exist_ok=True)
    if config["evaluation"]["save_images"]:
        os.makedirs(f"images/{args.expt_name}_{args.tensorboard_name}/"+config["dataset"]["name"]+"/random/", exist_ok=True)
        save_image_loc = f"images/{args.expt_name}_{args.tensorboard_name}/"+config["dataset"]["name"]+"/random/"

    model_type = config["model"]["type"]
    dataset = config["dataset"]["name"]

    if (torch.cuda.is_available() and config["train_config"]["use_cuda"]):
        use_cuda=True
        device = torch.device("cuda")
    else:
        use_cuda=False
        device = torch.device("cpu")

    # Pretrained weights are already loaded for stylegan2 through config file
    model = cbae_stygan2.CC_StyGAN2(config)
    model.to(device)

    # if CC has to be finetuned
    if args.load_pretrained:
        print(f'loading pretrained CC checkpoint from models/checkpoints/{args.pretrained_load_name}')
        model.cbae.load_state_dict(torch.load(f'models/checkpoints/{args.pretrained_load_name}'))

    if args.dataset == 'celebahq':
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Eyebrows', 'Arched Eyebrows']
        ]
        clsf_model_type = 'rn18'
    elif args.dataset == 'cub' or args.dataset == 'cub64':
        set_of_classes = [
            ['Large size', 'Small size 5 to 9 inches'],
            ['NOT perching like shape', 'Perching like shape'],
            ['NOT solid breast pattern', 'Solid breast pattern'],
            ['NOT black bill color', 'Black bill color'],
            ['Bill length longer than head', 'Bill length shorter than head'],
            ['NOT black wing color', 'Black wing color'],
            ['NOT solid belly pattern', 'Solid belly pattern'],
            ['NOT All purpose bill shape', 'All purpose bill shape'],
            ['NOT black upperparts color', 'Black upperparts color'],
            ['NOT white underparts color', 'White underparts color'],
        ]
        # for supervised pseudo-label model if used
        clsf_model_type = 'rn50'

    if args.pseudo_label == 'clipzs':
        print('using CLIP zero-shot for pseudo-labels')
        clip_zs = clip_pseudolabeler.CLIP_PseudoLabeler(set_of_classes, device)
    elif args.pseudo_label == 'supervised':
        print('using supervised model for pseudo-labels')
        clip_zs = clip_pseudolabeler.Sup_PseudoLabeler(set_of_classes, device, dataset=args.dataset, model_type=clsf_model_type)
    elif args.pseudo_label == 'tipzs':
        print('using Training-free CLIP Adapter (TIP) for pseudo-labels')
        clip_zs = clip_pseudolabeler.TIPAda_PseudoLabeler(set_of_classes, device, dataset=args.dataset)

    # freezing base generator parameters
    for param in model.gen.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(model.cbae.parameters(), lr=config["train_config"]["conc_lr"], betas=literal_eval(config["train_config"]["betas"]))

    pl_prob_thresh = config["train_config"]["pl_prob_thresh"]
    print(f'using probability threshold {pl_prob_thresh} in pseudolabel CE loss')

    steps_per_epoch = config["train_config"]["steps_per_epoch"]
    batch_size = config["dataset"]["batch_size"]
    for epoch in range(config["train_config"]["epochs"]):
        model.train()
        start = time.time()

        for i in range(steps_per_epoch):

            # -----------------
            #  Train CC
            # -----------------
            opt.zero_grad()

            ### 1. using latent noises
            # Sample noise and labels as generator input
            # latent = model.sample_latent(batch_size)
            # sampled_latent_copy = latent.clone()
            z = torch.randn((batch_size, model.gen.z_dim), device=device)
            latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)
            sampled_latent_copy = latent.clone()

            # get the concepts
            concepts = model.cbae.enc(latent)

            # generate images with original latent
            gen_imgs_latent = model.gen.synthesis(latent, noise_mode='const')
            # to make it from -1 to 1 range to 0 to 1
            gen_imgs_latent = gen_imgs_latent.mul(0.5).add_(0.5)

            with torch.no_grad():
                ## get probabilities and predicted labels from pseudolabeler
                pseudo_prob, pseudo_labels = clip_zs.get_pseudo_labels(gen_imgs_latent, return_prob=True)
                pseudo_prob = [pm.detach() for pm in pseudo_prob]
                pseudo_labels = [pl.detach() for pl in pseudo_labels]

            # concept alignment loss
            loss, _ = get_pseudo_concept_loss(model, concepts, pseudo_labels, pseudo_prob, pl_prob_thresh=pl_prob_thresh, device=device, dataset=args.dataset)

            loss.backward()
            opt.step()

            # --------------
            # Log Progress
            # --------------
            batches_done = epoch * steps_per_epoch + i
            if batches_done % config["train_config"]["log_interval"] == 0:
                print(
                    "Model %s Dataset %s [Epoch %d/%d] [Batch %d/%d] [total loss: %.4f]"
                    % (model_type,dataset,epoch, config["train_config"]["epochs"], i, steps_per_epoch, loss.item())
                    )
                if config["train_config"]["plot_loss"]:
                    writer.add_scalar('loss/concept_loss', loss.item(), global_step=batches_done)

        model.eval()

        ## saving CC predictions overlayed on generated images for sanity check
        eval_classifier(model, save_image_loc+"%d" % epoch, device=device, dataset=args.dataset, set_of_classes=set_of_classes)
        
        ## saving images for sanity check
        if config["evaluation"]["save_images"]:
            save_image(gen_imgs_latent.data, save_image_loc+"%d_latent.png" % epoch, nrow=8, normalize=True)

        if config["train_config"]["save_model"]:
            torch.save(model.cbae.state_dict(), "models/checkpoints/"+save_model_name+"_cc.pt")

        end = time.time()
        print("epoch time", end - start)
        print()


if __name__ == '__main__':
    main()