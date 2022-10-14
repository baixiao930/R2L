import os, sys, copy, math, random, json, time

import imageio
from tqdm import tqdm, trange
import numpy as np
import lpips as lpips_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from model.nerf_raybased import MYraw2outputs

from utils.ssim_torch import ssim as ssim_
from utils.run_nerf_raybased_helpers import mse2psnr, img2mse, to8b
from utils.run_nerf_raybased_helpers import sample_pdf, ndc_rays, get_rays, get_embedder, get_rays_np
from utils.flip_loss import FLIP

def render_path(args,
                point_sampler, 
                positional_embedder, 
                render_poses,
                hwf,
                chunk,
                no_ndc,
                render_kwargs,
                gt_imgs=None,
                savedir=None,
                render_factor=0,
                gauss=False):

    ssim = lambda img, ref: ssim_(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))
    lpips = lpips_.LPIPS(net=args.lpips_net).cuda()
    flip = FLIP()

    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = int(H / render_factor)
        W = int(W / render_factor)
        focal = focal / render_factor

    render_kwargs['network_fn'].eval()
    rgbs, disps, errors, ssims, psnrs = [], [], [], [], []

    # for testing DONERF data
    if args.given_render_path_rays:
        loaded = torch.load(args.given_render_path_rays)
        all_rays_o = loaded['all_rays_o'].to(device)  # [N, H*W, 3]
        all_rays_d = loaded['all_rays_d'].to(device)  # [N, H*W, 3]
        if not no_ndc:
            all_rays_o, all_rays_d = ndc_rays(H, W, focal, 1., all_rays_o, all_rays_d)
        if 'gt_imgs' in loaded:
            gt_imgs = loaded['gt_imgs'].to(device)  # [N, H, W, 3]
        print(f'Use given render_path rays: "{args.given_render_path_rays}"')

        model = render_kwargs['network_fn']
        for i in range(len(all_rays_o)):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                pts = point_sampler.sample_train(
                    all_rays_o[i], all_rays_d[i],
                    perturb=0)  # [H*W, n_sample*3]
                model_input = positional_embedder(pts)
                torch.cuda.synchronize()
                t_input = time.time()
                if args.learn_depth:
                    if not gauss:
                        rgbd = model(model_input)
                        rgb = rgbd[:, :3]
                    else:
                        raw = model(model_input)
                        t_vals = torch.linspace(0., 1., steps=64).cuda()
                        z_vals = args.near * (1.-t_vals) + args.far * (t_vals)
                        rgb = MYraw2outputs(1000000, raw, z_vals, args.near, args.far, all_rays_d[i], N_gauss=args.N_gauss)
                else:
                    if not gauss:
                        rgb = model(model_input)
                    else:
                        raw = model(model_input)
                        t_vals = torch.linspace(0., 1., steps=64).cuda()
                        z_vals = args.near * (1.-t_vals) + args.far * (t_vals)
                        rgb = MYraw2outputs(1000000, raw, z_vals, args.near, args.far, all_rays_d[i], N_gauss=args.N_gauss)
                torch.cuda.synchronize()
                t_forward = time.time()
                print(
                    f'[#{i}] frame, prepare input (embedding): {t_input - t0:.4f}s'
                )
                print(
                    f'[#{i}] frame, model forward: {t_forward - t_input:.4f}s')

                # reshape to image
                if args.dataset_type == 'llff':
                    H_, W_ = H, W  # non-square images
                elif args.dataset_type == 'blender':
                    H_ = W_ = int(math.sqrt(rgb.numel() / 3))
                rgb = rgb.view(H_, W_, 3)
                disp = rgb  # placeholder, to maintain compability

                rgbs.append(rgb)
                disps.append(disp)

                # @mst: various metrics
                if gt_imgs is not None:
                    errors += [(rgb - gt_imgs[i][:H_, :W_, :]).abs()]
                    psnrs += [mse2psnr(img2mse(rgb, gt_imgs[i, :H_, :W_]))]
                    ssims += [ssim(rgb, gt_imgs[i, :H_, :W_])]

                if savedir is not None:
                    filename = os.path.join(savedir, '{:03d}.png'.format(i))
                    imageio.imwrite(filename, to8b(rgbs[-1]))
                    imageio.imwrite(filename.replace('.png', '_gt.png'),
                                    to8b(gt_imgs[i]))  # save gt images
                    if len(errors):
                        imageio.imwrite(filename.replace('.png', '_error.png'),
                                        to8b(errors[-1]))

                torch.cuda.synchronize()
                print(
                    f'[#{i}] frame, rendering done, time for this frame: {time.time()-t0:.4f}s'
                )
                print('')
    else:
        for i, c2w in enumerate(render_poses):
            torch.cuda.synchronize()
            t0 = time.time()
            print(f'[#{i}] frame, rendering begins')
            if args.model_name in ['nerf']:
                rgb, disp, acc, _ = render(H,
                                           W,
                                           focal,
                                           chunk=chunk,
                                           c2w=c2w[:3, :4],
                                           **render_kwargs)
                H_, W_ = H, W

            else:  # For R2L model
                model = render_kwargs['network_fn']
                perturb = render_kwargs['perturb']

                # Network forward
                with torch.no_grad():
                    if args.given_render_path_rays:  # To test DONERF data using our model
                        pts = point_sampler.sample_train(
                            all_rays_o[i], all_rays_d[i],
                            perturb=0)  # [H*W, n_sample*3]
                    else:
                        if args.plucker: # False
                            pts = point_sampler.sample_test_plucker(
                                c2w[:3, :4])
                        else:
                            pts, rays_d = point_sampler.sample_test(
                                c2w[:3, :4], args.no_ndc)  # [H*W, n_sample*3]
                    model_input = positional_embedder(pts)
                    torch.cuda.synchronize()
                    t_input = time.time()
                    if args.learn_depth:
                        if not gauss:
                            rgbd = model(model_input)
                            rgb = rgbd[:, :3]
                        else:
                            raw = model(model_input)
                            t_vals = torch.linspace(0., 1., steps=64).cuda()
                            z_vals = args.near * (1.-t_vals) + args.far * (t_vals)
                            rgb = MYraw2outputs(1000000, raw, z_vals, args.near, args.far, rays_d, N_gauss=args.N_gauss)
                    else:
                        if not gauss:
                            rgb = model(model_input)
                        else:
                            raw = model(model_input)
                            t_vals = torch.linspace(0., 1., steps=64).cuda()
                            z_vals = args.near * (1.-t_vals) + args.far * (t_vals)
                            rgb = MYraw2outputs(1000000, raw, z_vals, args.near, args.far, rays_d, N_gauss=args.N_gauss)
                    torch.cuda.synchronize()
                    t_forward = time.time()
                    print(
                        f'[#{i}] frame, prepare input (embedding): {t_input - t0:.4f}s'
                    )
                    print(
                        f'[#{i}] frame, model forward: {t_forward - t_input:.4f}s'
                    )

                # Reshape to image
                if args.dataset_type == 'llff':
                    H_, W_ = H, W  # non-square images
                elif args.dataset_type == 'blender':
                    H_ = W_ = int(math.sqrt(rgb.numel() / 3))
                rgb = rgb.view(H_, W_, 3)
                disp = rgb  # Placeholder, to maintain compability

            rgbs.append(rgb)
            disps.append(disp)

            # @mst: various metrics
            if gt_imgs is not None:
                errors += [(rgb - gt_imgs[i][:H_, :W_, :]).abs()]
                psnrs += [mse2psnr(img2mse(rgb, gt_imgs[i, :H_, :W_]))]
                ssims += [ssim(rgb, gt_imgs[i, :H_, :W_])]

            if savedir is not None:
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, to8b(rgbs[-1]))
                imageio.imwrite(filename.replace('.png', '_gt.png'),
                                to8b(gt_imgs[i]))  # save gt images
                if len(errors):
                    imageio.imwrite(filename.replace('.png', '_error.png'),
                                    to8b(errors[-1]))

            torch.cuda.synchronize()
            print(
                f'[#{i}] frame, rendering done, time for this frame: {time.time()-t0:.4f}s'
            )
            print('')

    rgbs = torch.stack(rgbs, dim=0)
    disps = torch.stack(disps, dim=0)

    # https://github.com/richzhang/PerceptualSimilarity
    # LPIPS demands input shape [N, 3, H, W] and in range [-1, 1]
    misc = {}
    if gt_imgs is not None:
        rec = rgbs.permute(0, 3, 1, 2)  # [N, 3, H, W]
        ref = gt_imgs.permute(0, 3, 1, 2)  # [N, 3, H, W]
        rescale = lambda x, ymin, ymax: (ymax - ymin) / (x.max() - x.min()) * (
            x - x.min()) + ymin
        rec, ref = rescale(rec, -1, 1), rescale(ref, -1, 1)
        lpipses = []
        mini_batch_size = 8
        for i in np.arange(0, len(gt_imgs), mini_batch_size):
            end = min(i + mini_batch_size, len(gt_imgs))
            lpipses += [lpips(rec[i:end], ref[i:end])]
        lpipses = torch.cat(lpipses, dim=0)

        # -- get FLIP loss
        # flip standard values
        monitor_distance = 0.7
        monitor_width = 0.7
        monitor_resolution_x = 3840
        pixels_per_degree = monitor_distance * (monitor_resolution_x /
                                                monitor_width) * (np.pi / 180)
        flips = flip.compute_flip(rec, ref,
                                  pixels_per_degree)  # shape [N, 1, H, W]
        # --

        errors = torch.stack(errors, dim=0)
        psnrs = torch.stack(psnrs, dim=0)
        ssims = torch.stack(ssims, dim=0)
        test_loss = img2mse(rgbs,
                            gt_imgs[:, :H_, :W_])  # @mst-TODO: remove H_, W_

        misc['test_loss'] = test_loss
        misc['test_psnr'] = mse2psnr(test_loss)
        misc['test_psnr_v2'] = psnrs.mean()
        misc['test_ssim'] = ssims.mean()
        misc['test_lpips'] = lpipses.mean()
        misc['test_flip'] = flips.mean()
        misc['errors'] = errors

    render_kwargs['network_fn'].train()
    torch.cuda.empty_cache()
    return rgbs, disps, misc