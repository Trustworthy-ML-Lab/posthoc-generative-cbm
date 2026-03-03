

### training CB-AE for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
# python3 -u train/train_cbae_gan.py -e cbae_stygan2_thr90 -p supervised -t sup_pl_cls8

### training CC for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
# python3 -u train/train_cc_gan.py -e cc_stygan2_thr90 -p supervised -t sup_pl_cls8

### training CB-AE for CelebA-HQ-pretrained DDPM with supervised models as pseudolabeler
python3 -u train/train_cbae_ddpm.py -e cbae_ddpm -p supervised -t sup_pl_cls8 --base-root PATH_TO_YOUR_DATASET
