from cleanfid import fid
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder1",default="/path/to/CelebAMask-HQ/CelebA-HQ-img/",help="path to real dataset")
parser.add_argument("--folder2",default="results/fid_eval_images/cbae_stygan2/celebahq/",help="path to synthetic dataset")
args = parser.parse_args()

fdir1 = args.folder1
fdir2 = args.folder2 + 'clean/'

custom_name = "celebahq_"
if 'img_align_celeba' in args.folder1:
    custom_name = "celeba64_"

if not fid.test_stats_exists(custom_name, mode="clean"):
    fid.make_custom_stats(custom_name, fdir1, mode="clean")

score = fid.compute_fid(fdir2, dataset_name=custom_name, mode="clean", dataset_split="custom")
print(f'Clean StyleGAN2 FID: {score}')

fdir2 = args.folder2 + 'recon/'
score = fid.compute_fid(fdir2, dataset_name=custom_name, mode="clean", dataset_split="custom")
print(f'CB-AE reconstructed StyleGAN2 FID: {score}')

fdir2 = args.folder2 + 'interv/'
score = fid.compute_fid(fdir2, dataset_name=custom_name, mode="clean", dataset_split="custom")
print(f'CB-AE intervened StyleGAN2 FID: {score}')

# fdir2 = args.folder2 + 'optint/'
# score = fid.compute_fid(fdir2, dataset_name=custom_name, mode="clean", dataset_split="custom")
# print(f'CB-AE opt-int StyleGAN2 FID: {score}')