import os
import argparse
from distutils.util import strtobool
from numpy.distutils.fcompiler import str2bool

def get_parser():
    parser = argparse.ArgumentParser()

    ## path 
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default='configs/swinv2.yaml')
    parser.add_argument("--data", type=str, help='path to data directory', default='../example/real_person')
    parser.add_argument("--G_ckpt", type=str, help='path to generator model', default='../pretrained/ffhqrebalanced512-128.pkl')
    parser.add_argument("--E_ckpt", type=str, help='path to GOAE encoder checkpoint', default='../pretrained/encoder_FFHQ.pt')
    parser.add_argument("--AFA_ckpt", type=str, help='path to AFA model checkpoint', default='../pretrained/afa_FFHQ.pt')
    parser.add_argument("--R_ckpt", type=str, help='path to Reflectance model checkpoint', default=None)
    parser.add_argument("--outdir", type=str, help='path to output directory', default='../output/')
    parser.add_argument("--cuda", type=str, help="specify used cuda idx ", default='0')

    ## model
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--start_from_latent_avg", type=bool, default=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    ## other
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--w_frames', type=int, default=240)
    # parser.add_argument("--multi_view", action="store_true", default=False)
    # parser.add_argument("--video", action="store_true", default=False)
    # parser.add_argument("--shape", action="store_true", default=False)
    # parser.add_argument("--edit", action="store_true", default=False)
    # parser.add_argument("--relight", action="store_true", default=False)
    parser.add_argument("--fix_density", action="store_true", default=False)
    parser.add_argument("--render_cam", action="store_true", default=False)
    parser.add_argument('--num_emaps', type=int, default=1)

    ## edit 
    parser.add_argument("--edit_attr", type=str, help="editing attribute direction", default="glass")
    parser.add_argument("--alpha", type=float, help="editing alpha", default=1.0)
    parser.add_argument("--emap_mode", type=str, default='emaps_ds')

    ## Evaluation
    parser.add_argument('--input_cam', type=str, default='Cam07')
    parser.add_argument('--scan_name', type=str, default='ID00307')
    # parser.add_argument("--save_gt", action="store_true", default=False)
    parser.add_argument("--save_gt", type=strtobool, default=False)
    parser.add_argument("--emap_path", type=str, help='path to data env map directory')

    ## render type
    parser.add_argument("--relight", type=strtobool, default=False)
    parser.add_argument("--multi_view", type=strtobool, default=False)
    parser.add_argument("--video", type=str2bool, default=False)
    parser.add_argument("--shape", type=str2bool, default=False)
    parser.add_argument("--edit", type=str2bool, default=False)


    return parser