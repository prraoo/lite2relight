import os
import datetime
import dnnlib
import numpy as np
import torch
from configs.infer_config import get_parser
from configs.swin_config import get_config
from inversion.networks import Net
from tqdm import tqdm
from camera_utils import LookAtPoseSampler
import glob
import cv2
from natsort import natsorted


def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
    
    if yaw is None:
        yaw = np.random.uniform(-yaw_range, yaw_range)
    if pitch is None:
            pitch = np.random.uniform(-pitch_range, pitch_range)

    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
    return c


def build_dataloader(data_path, batch=1, pin_memory=True, prefetch_factor=2):
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_face.CameraLabeledDataset', path=data_path, 
                    use_labels=True, max_size=None, xflip=False,resolution=256, use_512 = True)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, pin_memory=pin_memory, prefetch_factor=prefetch_factor, num_workers=4)
    return dataloader, dataset


@torch.no_grad()
def infer_main(opts, device, now):

    ## camera parameters
    cam_pivot = torch.tensor([0, 0, 0.2], device=device)
    intrinsics = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1], device=device)
    face_pool = torch.nn.AdaptiveAvgPool2d((512, 512))

    ## build model
    swin_config = get_config(opts)
    net = Net(device, opts, swin_config)
    net.eval()


    ## build data
    dataloader, dataset = build_dataloader(data_path=opts.data, batch=opts.batch)
    render_cams = []

    ## main loop
    for data in tqdm(dataloader, disable=True):
        real_img, real_label, real_img_512, img_name = data
        real_label = real_label.to(device)
        render_cams.append((img_name, real_label))

        real_img = real_img.to(device).to(torch.float32) / 127.5 - 1.
        real_img_512 = real_img_512.to(device).to(torch.float32) / 127.5 - 1.
        ## Inversion
        rec_img_dict, rec_img_dict_w = net(real_img, real_label, real_img_512)

        # Reuse for reconstruction
        rec_ws = rec_img_dict['rec_ws']

        # Reuse for fixed density relighting
        triplane_x_rec = rec_img_dict['triplane_x']
        feature_map_adain_rec = rec_img_dict['feature_map_adain']
        rec_triplane_samples = rec_img_dict['triplane_samples']

        base_dir = os.path.join(opts.outdir)
        cam_name, sub_name, input_emap_name = (img_name[0].split(".")[0]).split('_')
        save_dir = os.path.join(base_dir, 'invert')
        os.makedirs(save_dir, exist_ok=True)
        if opts.relight:
            ## Get Environment Maps
            if opts.emap_mode == 'emaps_ds':
                emap_path = 'sample/envmaps/'
                emap_config = natsorted(glob.glob1(emap_path, '*.png'), reverse=True)
                emap_list = emap_config
            else:
                raise NotImplementedError

            for ii, emap_name in enumerate(emap_list):
                if opts.emap_mode == 'emaps_ds':
                    from_emap = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)[:, :, ::-1]
                    from_emap = np.resize(from_emap, (10, 20, 3))
                    from_emap = (from_emap).transpose(2, 0, 1)  # L2R convention
                    from_emap = torch.tensor(from_emap).to(device) / 127.5 - 1

                edit_relit = net.reflectance_network(rec_ws, from_emap[None, ...])
                relit_ws = rec_ws + edit_relit
                if opts.fix_density:
                    img_edit_dict, img_edit_dict_w = net.forward_relight_fix_density(rec_ws, relit_ws, real_img_512, real_label,
                                                                     triplane_x_rec, feature_map_adain_rec, rec_triplane_samples,
                                                                                     opts.fix_density)
                else:
                    img_edit_dict, img_edit_dict_w = net.forward_relight(rec_ws, relit_ws, real_img_512, real_label,
                                                                         triplane_x_rec, feature_map_adain_rec)

                mix_triplane_relit = img_edit_dict["mix_triplane"]
                edit_img = face_pool(img_edit_dict["image"])

                # Save Results
                relit_save_dir = os.path.join(base_dir, 'relit')
                os.makedirs(relit_save_dir, exist_ok=True)

                # Covert to Numpy
                save_edit_img = (edit_img.data.cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, ::-1]*0.5+0.5)*255
                save_real_img_512 = (real_img_512.data.cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, ::-1]*0.5+0.5)*255

                # Paste environment map in the lower right corner
                scale = 8
                emap_png = cv2.imread(os.path.join(emap_path, emap_name))
                inset_emap = cv2.resize(emap_png, (20*scale, 10*scale), interpolation=cv2.INTER_AREA)

                # Determine the position to place the 20x40 image in the lower right corner
                x_offset = 512 - 20*scale  # Calculate the x-offset for the lower right corner
                y_offset = 512 - 10*scale  # Calculate the y-offset for the lower right corner

                # Copy the 20x40 image into the lower right corner of the 512x512 image
                save_edit_img[y_offset:y_offset + 10*scale, x_offset:x_offset + 20*scale] = inset_emap
                cv2.imwrite(os.path.join(relit_save_dir, f'{cam_name}_{sub_name}_{emap_name[:-4]}.png'), cv2.hconcat([save_edit_img]))

                inset_emap = torch.tensor(cv2.cvtColor(inset_emap,cv2.COLOR_BGR2RGB))
                inset_emap = inset_emap.permute(2, 0, 1)
                inset_emap = inset_emap/127.5 - 1

                if opts.multi_view:
                    mv_relit_save_dir = os.path.join(base_dir, 'relit_multi_view')
                    os.makedirs(mv_relit_save_dir, exist_ok=True)

                    imgs_multi_view = []
                    n_views = 5
                    coef = [-2, -1, 0, 1, 2]
                    coef_pitch = [0, 0, 0, 0, 0]
                    yaw_list = []
                    pitch_list = []
                    
                    for cam_idx in range(len(coef)):
                        yaw_list.append(coef[cam_idx] * np.pi * 25 / 360)
                        pitch_list.append(coef_pitch[cam_idx] * np.pi * 20 / 360)
                    
                    for cam_idx in range(n_views):
                        c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw_list[cam_idx], pitch=pitch_list[cam_idx])
                        
                        img_dict_novel_view = net.decoder.synthesis(ws=relit_ws, c=c, triplane=mix_triplane_relit, forward_triplane=True, noise_mode='const')
                        img_novel_view = face_pool(img_dict_novel_view["image"])
                        imgs_multi_view.append(img_novel_view)

                        img_novel_view = img_novel_view.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
                        img_novel_view = ((img_novel_view * 0.5 + 0.5) * 255).astype(np.uint8)[:, :, ::-1]
                        cv2.imwrite(os.path.join(mv_relit_save_dir, f'{cam_name}_{sub_name}_{emap_name[:-4]}_mv_{cam_idx:02d}.png'), img_novel_view)

                if opts.edit:
                    if "edit_d" not in locals():
                        ## EG3D W space attribution entanglement when applying InterFaceGAN
                        if opts.edit_attr != "glass":
                            edit_d = np.load(os.path.join("./sample/ws_edit", opts.edit_attr+".npy"))
                            edit_d_glass = np.load(os.path.join("./sample/ws_edit", "glass.npy"))
                            edit_d = opts.alpha*edit_d - 0.8*edit_d_glass
                        else:
                            edit_d = np.load(os.path.join("./sample/ws_edit", opts.edit_attr+".npy"))
                            edit_d = opts.alpha*edit_d

                        edit_d = torch.tensor(edit_d).to(device)

                    edit_ws = relit_ws + edit_d
                    img_edit_dict, img_edit_dict_w = net.edit(relit_ws, edit_ws, real_img_512, real_label)

                    mix_triplane_edit = img_edit_dict["mix_triplane"]
                    save_edit_relit_img = face_pool(img_edit_dict["image"])

                    # Save Results
                    edit_relit_save_dir = os.path.join(base_dir, 'edit_relit')
                    os.makedirs(edit_relit_save_dir, exist_ok=True)

                    save_edit_relit_img = (save_edit_relit_img.data.cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, ::-1]*0.5+0.5)*255
                    cv2.imwrite(os.path.join(edit_relit_save_dir, f'{cam_name}_{sub_name}_{emap_name[:-4]}.png'), save_edit_relit_img)

                    if opts.multi_view:
                        mv_relit_save_dir = os.path.join(base_dir, 'edit_relit_multi_view')
                        os.makedirs(mv_relit_save_dir, exist_ok=True)

                        imgs_multi_view = []
                        for j in range(4):
                            yaw =  coef[j] * np.pi*25/360
                            pitch = coef_pitch[j] * np.pi*25/360
                            c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)

                            img_dict_novel_view = net.decoder.synthesis(ws=edit_ws, c=c, triplane=mix_triplane_edit, forward_triplane=True, noise_mode='const')
                            img_novel_view = face_pool(img_dict_novel_view["image"])
                            img_novel_view[:, :, y_offset:y_offset + 10 * scale, x_offset:x_offset + 20 * scale] = inset_emap
                            imgs_multi_view.append(img_novel_view)
                            img_novel_view = img_novel_view.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
                            img_novel_view = ((img_novel_view * 0.5 + 0.5) * 255).astype(np.uint8)[:, :, ::-1]
                            cv2.imwrite(os.path.join(mv_relit_save_dir, f'{cam_name}_{sub_name}_{emap_name[:-4]}_mv_{cam_idx:02d}.png'), img_novel_view)




if __name__=="__main__":
    parser = get_parser()
    opts = parser.parse_args()

    # random.seed(opts.seed)
    # np.random.seed(opts.seed)
    print("="*50, "Using CUDA: " + opts.cuda, "="*50)
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.cuda
    device = torch.device('cuda:' + opts.cuda)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    ckpt_dir = os.path.dirname(opts.R_ckpt)
    ckpts = natsorted(glob.glob1(ckpt_dir, '*.pkl'))
    opts.R_ckpt = os.path.join(ckpt_dir, ckpts[-1])
    print(f'Loading {ckpts[-1]}')

    infer_main(opts, device, now)