import os
import sys
sys.path.append('.')
from utils.utils import create_image_grid, get_concept_index
import argparse
import numpy as np
from pathlib import Path
import yaml
import torch
from ast import literal_eval
from torchvision.utils import save_image
from torch import nn
from models import cbae_unet2d, clip_pseudolabeler
import torchvision.transforms as transforms
import warnings
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import CelebAHQ_imgonly, CelebAHQ_dataset_multiconc, CUB_dataset_multiconc
from accelerate import Accelerator
from diffusers import DDPMScheduler
from PIL import Image
import time

warnings.filterwarnings("ignore", category=UserWarning)


class GeneratedImageDataset(torch.utils.data.Dataset):
    """Loads pre-generated images from a flat directory of PNGs."""
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted(Path(img_dir).glob('*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label; pseudo-labeler provides concept supervision


def test_acc_concepts(model, noise_scheduler, dataloader, device, max_t=100):
    no_samples = 0
    correct = [0 for _ in range(model.n_concepts)]

    for i, (real_imgs, concepts) in enumerate(dataloader):
        with torch.no_grad():
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.shape[0]
            no_samples += batch_size

            rand_t = torch.randint(1, max_t, (1,)).item()
            timesteps = torch.ones((batch_size,), device=device).long() * rand_t

            z = torch.randn(real_imgs.shape).to(device)
            noisy_images = noise_scheduler.add_noise(real_imgs, z, timesteps)
            latent, _, _ = model.gen.forward_part1(noisy_images, timesteps)
            logits = model.cbae.enc(latent)

            for c in range(model.n_concepts):
                concepts[c] = concepts[c].to(device)
                if model.concept_type[c] == "cat":
                    cat_onehot = torch.zeros(batch_size, model.concept_bins[c], dtype=torch.float, device=device)
                    cat_onehot.scatter_(1, concepts[c].long().unsqueeze(-1), 1)
                elif model.concept_type[c] == "bin":
                    cat_onehot = torch.zeros(batch_size, model.concept_bins[c], dtype=torch.float, device=device)
                    cat_onehot[:, 0] = 1 - concepts[c]
                    cat_onehot[:, 1] = concepts[c]
                c_real_concepts = torch.argmax(cat_onehot, dim=1)

                start, end = get_concept_index(model, c)
                c_pred_concepts = logits[:, start:end]
                correct[c] += torch.sum(c_real_concepts == torch.argmax(c_pred_concepts, dim=1)).item()

    for c in range(model.n_concepts):
        correct[c] = correct[c] / no_samples
        print("Concept %s [Acc %.4f]" % (model.concept_name[c], correct[c]), end=" ")
    print()
    return correct


def eval_classifier(testloader, model, save_image_loc, num_samples=64, dataset='celebahq', device='cuda', noise_scheduler=None, max_t=100, set_of_classes=[]):
    batch_size = testloader.batch_size
    for eval_idx, clean_images in enumerate(testloader):
        if eval_idx > (num_samples // batch_size):
            break
        with torch.no_grad():
            clean_images = clean_images.to(device)
            z = torch.randn(clean_images.shape).to(device)
            rand_t = torch.randint(1, max_t, (1,)).item()
            timesteps = torch.ones((batch_size,), device=device).long() * rand_t

            noisy_images = noise_scheduler.add_noise(clean_images, z, timesteps)
            latent, t_emb, unet_residual = model.gen.forward_part1(noisy_images, timesteps)
            concepts = model.cbae.enc(latent)

            noise_pred_latent = model.gen.forward_part2(latent, emb=t_emb, down_block_res_samples=unet_residual, return_dict=False)
            gen_imgs_latent = noise_scheduler.step(noise_pred_latent, rand_t, noisy_images).prev_sample
            gen_imgs_latent = gen_imgs_latent.mul(0.5).add_(0.5)

        pseudo_label_list = []
        pseudo_probs_list = []
        for c in range(model.n_concepts):
            start, end = get_concept_index(model, c)
            c_predicted_concepts = concepts[:, start:end].softmax(dim=1)
            values, indices = torch.max(c_predicted_concepts, dim=1)
            pseudo_label_list.append(indices)
            pseudo_probs_list.append(values)

        create_image_grid(gen_imgs_latent, pseudo_label_list, pseudo_probs_list, save_image_loc + "_%d.png" % eval_idx, n_row=4, n_col=batch_size // 4, set_of_classes=set_of_classes, figsize=(20, 20), textwidth=30)


def get_pseudo_concept_loss(model, predicted_concepts, pseudolabel_concepts, pseudolabel_probs, device, pl_prob_thresh=0.1, dataset='celebahq', ignore_index=250, use_pl_thresh=True):
    concept_loss = 0
    if dataset in ('celebahq', 'celeba64', 'cub', 'cub64'):
        if use_pl_thresh:
            for cdx in range(len(pseudolabel_concepts)):
                pseudolabel_concepts[cdx][pseudolabel_probs[cdx] < pl_prob_thresh] = ignore_index
        concepts = [curr_conc.long() for curr_conc in pseudolabel_concepts]
    else:
        raise NotImplementedError(f'not implemented for {dataset}')

    loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    concept_loss_lst = []
    for c in range(model.n_concepts):
        start, end = get_concept_index(model, c)
        c_predicted_concepts = predicted_concepts[:, start:end]
        c_real_concepts = concepts[c]
        c_concept_loss = loss_ce(c_predicted_concepts, c_real_concepts)
        concept_loss += c_concept_loss
        concept_loss_lst.append(c_concept_loss)
    return concept_loss, concept_loss_lst


def get_concept_loss(model, predicted_concepts, concepts, isList=False):
    concept_loss = 0
    loss_ce = nn.CrossEntropyLoss()
    concept_loss_lst = []
    for c in range(model.n_concepts):
        start, end = get_concept_index(model, c)
        c_predicted_concepts = predicted_concepts[:, start:end]
        c_real_concepts = concepts[c] if isList else concepts[:, start:end]
        c_concept_loss = loss_ce(c_predicted_concepts, c_real_concepts)
        concept_loss += c_concept_loss
        concept_loss_lst.append(c_concept_loss)
    return concept_loss, concept_loss_lst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", default="celebahq", help="benchmark dataset")
    parser.add_argument("-e", "--expt-name", default="cbae_ddpm", help="name for saving images and checkpoint")
    parser.add_argument("-t", "--tensorboard-name", default="clipzs", help="suffix for tensorboard experiment name")
    parser.add_argument("-p", "--pseudo-label", type=str, default='clipzs', help='pseudo-label source: clipzs, supervised, tipzs, or real')
    parser.add_argument("--base-root", default='', type=str, help='root directory containing datasets/')
    parser.add_argument("--max-t", type=int, default=400, help="max noise timestep for training (1=nearly clean, 1000=full noise)")
    parser.add_argument("--use-real-images", action='store_true', default=False, help='use real dataset images for training instead of pre-generated images')
    parser.add_argument("--generated-img-dir", type=str, default='datasets/generated/ddpm-celebahq-256', help='directory of pre-generated training images (used with --use-generated)')
    args = parser.parse_args()
    args.config_file = f"./config/{args.expt_name}/" + args.dataset + ".yaml"

    if not args.use_real_images and args.pseudo_label == 'real':
        raise ValueError('--pseudo-label real requires ground-truth concept labels, which generated images do not have.')

    writer = SummaryWriter(f'results/{args.dataset}_{args.expt_name}_{args.tensorboard_name}')

    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")

    use_cuda = config["train_config"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ignore_index = 250

    if config["train_config"]["save_model"]:
        save_model_name = f"{config['dataset']['name']}_{args.expt_name}_{args.tensorboard_name}"

    if config["evaluation"]["save_images"] or config["evaluation"]["save_concept_image"]:
        os.makedirs("generation_checkpoints/", exist_ok=True)
        os.makedirs("images/", exist_ok=True)
        os.makedirs(f"images/{args.expt_name}_{args.tensorboard_name}/", exist_ok=True)
        os.makedirs(f"images/{args.expt_name}_{args.tensorboard_name}/" + config["dataset"]["name"] + "/", exist_ok=True)
    if config["evaluation"]["save_images"]:
        os.makedirs(f"images/{args.expt_name}_{args.tensorboard_name}/" + config["dataset"]["name"] + "/random/", exist_ok=True)
        save_image_loc = f"images/{args.expt_name}_{args.tensorboard_name}/" + config["dataset"]["name"] + "/random/"
    if config["evaluation"]["save_results"]:
        os.makedirs("results/", exist_ok=True)

    os.makedirs("models/checkpoints", exist_ok=True)

    model_type = config["model"]["type"]
    dataset = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]

    model = cbae_unet2d.cbAE_DDPM(config)
    model.to(device)

    # --- Dataset / concept class setup ---
    if args.dataset == 'celebahq':
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Eyebrows', 'Arched Eyebrows'],
        ]
        dset_class_names = ['Attractive', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Smiling', 'High_Cheekbones', 'Heavy_Makeup', 'Male', 'Arched_Eyebrows']
        clsf_model_type = 'rn18'
    elif args.dataset == 'celeba64':
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Hair', 'Wavy Hair'],
        ]
        dset_class_names = ['Attractive', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Smiling', 'High_Cheekbones', 'Heavy_Makeup', 'Male', 'Wavy_Hair']
        clsf_model_type = 'rn18'
    elif args.dataset in ('cub', 'cub64'):
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
        dset_class_names = [219, 236, 55, 290, 152, 21, 245, 7, 36, 52]
        clsf_model_type = 'rn50'

    # --- Pseudo-labeler setup ---
    if args.pseudo_label == 'clipzs':
        print('using CLIP zero-shot for pseudo-labels')
        clip_zs = clip_pseudolabeler.CLIP_PseudoLabeler(set_of_classes, device)
    elif args.pseudo_label in ('supervised', 'real'):
        if args.pseudo_label == 'real':
            print('WARNING: using real labels for training, use only if intended, e.g. for sensitivity analysis')
        else:
            print('using supervised model for pseudo-labels')
        clip_zs = clip_pseudolabeler.Sup_PseudoLabeler(set_of_classes, device, dataset=args.dataset, model_type=clsf_model_type)
    elif args.pseudo_label == 'tipzs':
        print('using Training-free CLIP Adapter (TIP) for pseudo-labels')
        clip_zs = clip_pseudolabeler.TIPAda_PseudoLabeler(set_of_classes, device)

    # --- Data loading ---
    if args.dataset in ('celebahq', 'celeba64', 'cub', 'cub64'):
        base_root = args.base_root
        if args.dataset == 'celebahq':
            img_root = os.path.join(base_root, 'datasets/CelebAMask-HQ/CelebA-HQ-img')
            train_file = os.path.join(base_root, 'datasets/CelebAMask-HQ/train.txt')
            test_file = os.path.join(base_root, 'datasets/CelebAMask-HQ/test.txt')
            img_size = 256
        elif args.dataset == 'celeba64':
            img_root = os.path.join(base_root, 'datasets/img_align_celeba')
            train_file = os.path.join(base_root, 'datasets/celeba64_train_annotations.txt')
            test_file = os.path.join(base_root, 'datasets/celeba64_val_annotations.txt')
            img_size = 64
        elif args.dataset in ('cub', 'cub64'):
            img_root = os.path.join(base_root, 'datasets/CUB_200_2011/images')
            image_path = os.path.join(base_root, 'datasets/CUB_200_2011/images.txt')
            anno_file = os.path.join(base_root, 'datasets/CUB_200_2011/attributes/image_attribute_labels.txt')
            split_file = os.path.join(base_root, 'datasets/CUB_200_2011/train_test_split.txt')
            img_size = 64 if args.dataset == 'cub64' else 256

        transforms_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        transforms_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Test set always uses the real labeled dataset for evaluation
        if args.dataset in ('celeba64', 'celebahq'):
            testset = CelebAHQ_imgonly(img_root, test_file, transform=transforms_test)
            testset_with_labels = CelebAHQ_dataset_multiconc(img_root, test_file, set_of_classes=dset_class_names, transform=transforms_test)
        elif args.dataset in ('cub', 'cub64'):
            testset = CUB_dataset_multiconc(img_root, image_path, anno_file, split_file, set_of_classes=dset_class_names, transform=transforms_test, label_format='none', split='test', tipzs=False)
            testset_with_labels = CUB_dataset_multiconc(img_root, image_path, anno_file, split_file, set_of_classes=dset_class_names, transform=transforms_test, label_format='list', split='test', tipzs=False)

        # Training set: pre-generated images (default) or real dataset
        if not args.use_real_images:
            print(f'using pre-generated images from {args.generated_img_dir} for training')
            trainset = GeneratedImageDataset(args.generated_img_dir, transform=transforms_train)
        else:
            if args.dataset in ('celeba64', 'celebahq'):
                trainset = CelebAHQ_dataset_multiconc(img_root, train_file, set_of_classes=dset_class_names, transform=transforms_train)
            elif args.dataset in ('cub', 'cub64'):
                trainset = CUB_dataset_multiconc(img_root, image_path, anno_file, split_file, set_of_classes=dset_class_names, transform=transforms_train, label_format='list', split='train', tipzs=False)

        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
        test_dataloader_wlabels = torch.utils.data.DataLoader(testset_with_labels, batch_size=16, shuffle=False, num_workers=4, drop_last=True)

        for param in model.gen.parameters():
            param.requires_grad = False

    # --- Optimizers ---
    # opt: reconstruction + concept alignment losses; opt_interv: intervention losses
    opt = torch.optim.Adam(model.cbae.parameters(), lr=config["train_config"]["recon_lr"], betas=literal_eval(config["train_config"]["betas"]))
    opt_interv = torch.optim.Adam(model.cbae.parameters(), lr=config["train_config"]["conc_lr"], betas=literal_eval(config["train_config"]["betas"]))

    reconstr_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    accelerator = Accelerator(mixed_precision='fp16')
    model, opt, opt_interv, train_dataloader, test_dataloader_wlabels = accelerator.prepare(model, opt, opt_interv, train_dataloader, test_dataloader_wlabels)

    pl_prob_thresh = config["train_config"]["pl_prob_thresh"]
    print(f'using probability threshold {pl_prob_thresh} in pseudolabel CE loss')

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # --- Training loop ---
    for epoch in range(config["train_config"]["epochs"]):
        model.train()
        start = time.time()

        for i, (clean_images, concept_labels) in enumerate(train_dataloader):
            clean_images = clean_images.to(accelerator.device)
            if args.pseudo_label != 'real':
                concept_labels = None

            z = torch.randn(clean_images.shape).to(clean_images.device)
            rand_t = torch.randint(1, args.max_t, (1,)).item()
            timesteps = torch.ones((batch_size,), device=device).long() * rand_t

            noisy_images = noise_scheduler.add_noise(clean_images, z, timesteps)
            latent, t_emb, unet_residual = model.gen.forward_part1(noisy_images, timesteps)
            sampled_latent_copy = latent.clone()

            concepts = model.cbae.enc(latent)
            recon_latent = model.cbae.dec(concepts)

            noise_pred_latent = model.gen.forward_part2(latent, emb=t_emb, down_block_res_samples=unet_residual, return_dict=False)
            noise_pred_recon_latent = model.gen.forward_part2(recon_latent, emb=t_emb, down_block_res_samples=unet_residual, return_dict=False)
            gen_imgs_latent = noise_scheduler.step(noise_pred_latent, rand_t, noisy_images).prev_sample.mul(0.5).add_(0.5)
            gen_imgs_recon_latent = noise_scheduler.step(noise_pred_recon_latent, rand_t, noisy_images).prev_sample.mul(0.5).add_(0.5)

            if args.pseudo_label != 'real':
                with torch.no_grad():
                    clean_images_viz = clean_images.mul(0.5).add_(0.5)
                    # using clean images directly for pseudo-labels
                    pseudo_prob, pseudo_labels = clip_zs.get_pseudo_labels(clean_images_viz, return_prob=True)
                    pseudo_prob = [pm.detach() for pm in pseudo_prob]
                    pseudo_labels = [pl.detach() for pl in pseudo_labels]
                concept_loss, concept_loss_list = get_pseudo_concept_loss(model, concepts, pseudo_labels, pseudo_prob, pl_prob_thresh=pl_prob_thresh, device=device, dataset=args.dataset)
            else:
                with torch.no_grad():
                    clean_images_viz = clean_images.mul(0.5).add_(0.5)
                pseudo_labels = concept_labels
                concept_loss, concept_loss_list = get_pseudo_concept_loss(model, concepts, pseudo_labels, None, pl_prob_thresh=0.0, device=device, dataset=args.dataset, use_pl_thresh=False)

            recon_loss = reconstr_loss(recon_latent, latent)
            img_recon_loss = reconstr_loss(gen_imgs_latent, clean_images_viz) + reconstr_loss(gen_imgs_recon_latent, clean_images_viz)
            loss = recon_loss + img_recon_loss + concept_loss

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()

            # --- Intervention loss ---
            rand_concept = torch.randint(low=0, high=len(set_of_classes), size=(1,)).item()
            concept_value = torch.randint(low=0, high=len(set_of_classes[rand_concept]), size=(1,)).item()

            latent = sampled_latent_copy.clone()

            with torch.no_grad():
                concepts = model.cbae.enc(latent)
                start_idx, end_idx = get_concept_index(model, rand_concept)
                intervened_concepts = concepts.clone()
                curr_c_concepts = intervened_concepts[:, start_idx:end_idx]

                # swap: move max value to the target concept slot
                old_vals = curr_c_concepts[:, concept_value].clone()
                max_val, max_ind = torch.max(curr_c_concepts, dim=1)
                curr_c_concepts[:, concept_value] = max_val
                for swap_idx, (curr_ind, curr_old_val) in enumerate(zip(max_ind, old_vals)):
                    curr_c_concepts[swap_idx, curr_ind] = curr_old_val

                intervened_concepts[:, start_idx:end_idx] = curr_c_concepts
                intervened_concepts = intervened_concepts.detach()

                intervened_pseudo_label = [temp_pl.clone() for temp_pl in pseudo_labels]
                intervened_pseudo_label[rand_concept] = (torch.ones((batch_size,), device=device) * concept_value).long()
                intervened_pseudo_label = [temp_pl.detach() for temp_pl in intervened_pseudo_label]

            intervened_latent = model.cbae.dec(intervened_concepts)
            intervened_noise_pred = model.gen.forward_part2(intervened_latent, emb=t_emb, down_block_res_samples=unet_residual, return_dict=False)
            intervened_gen_imgs = noise_scheduler.step(intervened_noise_pred, rand_t, noisy_images).prev_sample.mul(0.5).add_(0.5)

            recon_intervened_concepts = model.cbae.enc(intervened_latent)

            pred_logits = clip_zs.get_soft_pseudo_labels(intervened_gen_imgs)
            intervened_pseudo_label_loss = sum(ce_loss(curr_logits, actual_pl) for curr_logits, actual_pl in zip(pred_logits, intervened_pseudo_label))

            intervened_concept_loss, _ = get_pseudo_concept_loss(model, recon_intervened_concepts, intervened_pseudo_label, None, use_pl_thresh=False, device=device, dataset=args.dataset)
            total_intervened_loss = intervened_concept_loss + intervened_pseudo_label_loss

            accelerator.backward(total_intervened_loss)
            opt_interv.step()
            opt_interv.zero_grad()

            # --- Logging ---
            batches_done = epoch * len(train_dataloader) + i
            if batches_done % config["train_config"]["log_interval"] == 0:
                print(
                    "Model %s Dataset %s [Epoch %d/%d] [Batch %d/%d] [total loss: %.4f] [conc: %.4f] [lat rec: %.4f] [img rec: %.4f] [interv loss: %.4f]"
                    % (model_type, dataset, epoch, config["train_config"]["epochs"], i, len(train_dataloader), loss.item(), concept_loss.item(), recon_loss.item(), img_recon_loss.item(), total_intervened_loss.item())
                )
                if config["train_config"]["plot_loss"]:
                    writer.add_scalar('loss/concept_loss', concept_loss.item(), global_step=batches_done)
                    writer.add_scalar('loss/recon_loss', recon_loss.item(), global_step=batches_done)
                    writer.add_scalar('loss/img_recon_loss', img_recon_loss.item(), global_step=batches_done)
                    writer.add_scalar('loss/interv_loss', total_intervened_loss.item(), global_step=batches_done)

        model.eval()

        eval_classifier(test_dataloader, model, save_image_loc + "%d" % epoch, device=device, dataset=args.dataset, noise_scheduler=noise_scheduler, max_t=args.max_t, set_of_classes=set_of_classes)
        test_acc = test_acc_concepts(model, noise_scheduler, test_dataloader_wlabels, device, max_t=args.max_t)
        print(f'Average Accuracy: {np.mean(test_acc):.4f}')
        if config["train_config"]["plot_loss"]:
            writer.add_scalar('acc/conc_test_acc', np.mean(test_acc), global_step=epoch * len(train_dataloader))

        if config["evaluation"]["save_images"]:
            noisy_images_viz = noisy_images.mul(0.5).add_(0.5)
            save_image(clean_images_viz.data, save_image_loc + "%d_real.png" % epoch, nrow=8, normalize=True)
            save_image(noisy_images_viz.data, save_image_loc + "%d_noisy.png" % epoch, nrow=8, normalize=True)
            save_image(gen_imgs_latent.data, save_image_loc + "%d_latent.png" % epoch, nrow=8, normalize=True)
            save_image(gen_imgs_recon_latent.data, save_image_loc + "%d_recon_latent.png" % epoch, nrow=8, normalize=True)
            save_image(intervened_gen_imgs.data, save_image_loc + "%d_interv.png" % epoch, nrow=8, normalize=True)

        if config["train_config"]["save_model"]:
            torch.save(model.cbae.state_dict(), "models/checkpoints/" + save_model_name + "_cbae.pt")

        end = time.time()
        print("epoch time", end - start)
        print()


if __name__ == '__main__':
    main()
