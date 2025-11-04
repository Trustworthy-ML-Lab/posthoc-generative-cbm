import os
import sys
sys.path.append('.')
import argparse
import yaml
import torch
from torchvision.utils import save_image
from models import cbae_stygan2
from tqdm import tqdm


def get_concept_index(model, c):
    if c==0:
        start=0
    else:
        start=sum(model.concept_bins[:c])
    end= sum(model.concept_bins[:c+1])

    return start,end


def main():
    # We only specify the yaml file from argparse and handle rest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", default='cbae_stygan2', help='name of model file to use')
    parser.add_argument("-d", "--dataset",default="celebahq",help="benchmark dataset")
    parser.add_argument("-e", "--expt-name", default="cbae_stygan2", help="name used earlier for saving images and checkpoint")
    parser.add_argument("-g", "--gan-type", default='stylegan2', choices=['stylegan2', 'pgan', 'dcgan'], help='which base generative model')
    parser.add_argument("-t", "--tensorboard-name", default='clipzs_cbae', help='name used earlier for training')
    args = parser.parse_args()
    args.config_file = f"./config/{args.expt_name}/"+args.dataset+".yaml"

    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")
    

    use_cuda = config["train_config"]["use_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config["evaluation"]["save_images"] or config["evaluation"]["save_concept_image"]:
        # os.makedirs("generation_checkpoints/", exist_ok=True)
        os.makedirs("results/fid_eval_images/", exist_ok=True)
        os.makedirs(f"results/fid_eval_images/{args.expt_name}_{args.tensorboard_name}/", exist_ok=True)
        os.makedirs(f"results/fid_eval_images/{args.expt_name}_{args.tensorboard_name}/"+config["dataset"]["name"]+"/", exist_ok=True)
        save_image_loc = f"results/fid_eval_images/{args.expt_name}_{args.tensorboard_name}/{config['dataset']['name']}/"

    os.makedirs(f'{save_image_loc}clean', exist_ok=True)
    os.makedirs(f'{save_image_loc}interv', exist_ok=True)
    os.makedirs(f'{save_image_loc}recon', exist_ok=True)

    if args.dataset == 'celeba64':
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Hair', 'Wavy Hair']
        ]
    elif args.dataset == 'celebahq' or args.dataset == 'celebahq40':
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
    elif args.dataset == 'cub':
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

    if 'cc' in args.expt_name:
        control_type = 'cc'
        if args.gan_type == 'stylegan2':
            model = cbae_stygan2.CC_StyGAN2(config)
    elif 'cbae' in args.expt_name:
        control_type = 'cbae'
        if args.gan_type == 'stylegan2':
            model = cbae_stygan2.cbAE_StyGAN2(config)

    cbae_ckpt_path = f'models/checkpoints/{args.dataset}_{args.expt_name}_{args.tensorboard_name}_{control_type}.pt'

    model.cbae.load_state_dict(torch.load(cbae_ckpt_path, map_location='cpu'))
    model.to(device)
    model.eval()

    batch_size = 32
    print(f'saving images to {save_image_loc}')
    ## 10k images (313 * 32 = ~10k)
    for idx in tqdm(range(313)):
        # Sample noise and labels as generator input
        z = torch.randn((batch_size, model.gen.z_dim), device=device)
        latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)

        concepts = model.cbae.enc(latent)

        new_concepts = concepts.clone()

        ## using a random concept to get intervened images for FID computation
        # intervene on one randomly selected concept with randomly selected value
        concept_change = torch.randint(low=0, high=len(set_of_classes), size=(1,)).item()
        concept_value = torch.randint(low=0, high=len(set_of_classes[concept_change]), size=(1,)).item()
        start, end = get_concept_index(model, concept_change)
        c_concepts = concepts[:, start:end]
        _, num_c = c_concepts.shape

        # swapping the max value to the concept we need
        new_c_concepts = c_concepts.clone()
        old_vals = new_c_concepts[:, concept_value].clone()
        max_val, max_ind = torch.max(new_c_concepts, dim=1)
        new_c_concepts[:, concept_value] = max_val
        for swap_idx, (curr_ind, curr_old_val) in enumerate(zip(max_ind, old_vals)):
            new_c_concepts[swap_idx, curr_ind] = curr_old_val

        # print(new_concepts[:, start:end].shape, new_c_concepts.shape)
        new_concepts[:, start:end] = new_c_concepts

        new_latent = model.cbae.dec(new_concepts)
        recon_latent = model.cbae.dec(concepts)

        # Generate a batch of images
        # using original latent
        gen_imgs_latent_clean = model.gen.synthesis(latent, noise_mode='const')
        # to make it from -1 to 1 range to 0 to 1
        gen_imgs_latent_clean = gen_imgs_latent_clean.mul(0.5).add_(0.5)

        # using intervened latent
        gen_imgs_latent = model.gen.synthesis(new_latent, noise_mode='const')
        # to make it from -1 to 1 range to 0 to 1
        gen_imgs_latent = gen_imgs_latent.mul(0.5).add_(0.5)

        # using recon latent
        gen_imgs_latent_orig = model.gen.synthesis(recon_latent, noise_mode='const')
        # to make it from -1 to 1 range to 0 to 1
        gen_imgs_latent_orig = gen_imgs_latent_orig.mul(0.5).add_(0.5)

        ## saving individual images (original images in clean/, CB-AE reconstructed images in recon/ and CB-AE intervened images in interv/)
        for batch_idx in range(batch_size):
            img_num = (batch_size * idx) + batch_idx
            save_image(gen_imgs_latent_clean[batch_idx].unsqueeze(0).data, save_image_loc+f"clean/{img_num:05}.png", normalize=True)
            save_image(gen_imgs_latent[batch_idx].unsqueeze(0).data, save_image_loc+f"interv/{img_num:05}.png", normalize=True)
            save_image(gen_imgs_latent_orig[batch_idx].unsqueeze(0).data, save_image_loc+f"recon/{img_num:05}.png", normalize=True)

if __name__ == '__main__':
    main()