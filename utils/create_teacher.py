import torch
import numpy as np
from easydict import EasyDict as edict
from smilelogging import Logger

from utils.run_nerf_raybased_helpers import get_embedder, sample_pdf, load_weights_v2
from model.nerf_raybased import NeRF, raw2outputs, PositionalEmbedder, Teacher_raw2outputs
from utils.run_nerf_raybased_helpers import parse_expid_iter, to_tensor, to_array, mse2psnr, to8b, img2mse
from smilelogging.utils import Timer, LossLine, get_n_params_, get_n_flops_, AverageMeter, ProgressMeter

eval_args = edict(d = {
    'multires': 10, 
    'i_embed': 0, 
    'use_viewdirs': True, 
    'multires_views': 4,
    'dataset_type': 'llff',
    'no_ndc': False, 
    'netdepth': 8,
    'netdepth_fine': 8,
    'netwidth': 256,
    'netwidth_fine': 256,
    'pretrained_ckpt': 'Experiments/NeRF__blender_fern_SERVER-20221012-102142/weights/ckpt_best.tar',
    'perturb': 1,
    'perturb_test': 0,
    'N_importance': 64,
    'N_samples': 64,
    'white_bkgd': False, 
    'raw_noise_std': 1e0,
    'lindisp': False
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([
            fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)
        ], 0)

    return ret

def run_network(inputs,
                viewdirs,
                fn,
                embed_fn,
                embeddirs_fn,
                netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(
        inputs, [-1, inputs.shape[-1]]
    )  # @mst: shape: torch.Size([65536, 3]), 65536=1024*64 (n_rays * n_sample_per_ray)
    embedded = embed_fn(inputs_flat)  # shape: [n_rays*n_sample_per_ray, 63]

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat,
                            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_teacher(args, near, far):
    """Instantiate NeRF's MLP model.
    """
    # set up model
    model_fine = network_query_fn = None
    global embed_fn
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views,
                                                    args.i_embed)

    # @mst: use external positional embedding for our raybased nerf
    positional_embedder = PositionalEmbedder(L=args.multires)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth,
                    W=args.netwidth,
                    input_ch=input_ch,
                    output_ch=output_ch,
                    skips=skips,
                    input_ch_views=input_ch_views,
                    use_viewdirs=args.use_viewdirs).to(device)

    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine,
                        W=args.netwidth_fine,
                        input_ch=input_ch,
                        output_ch=output_ch,
                        skips=skips,
                        input_ch_views=input_ch_views,
                        use_viewdirs=args.use_viewdirs).to(device)

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn)

    # load state_dict
    ckpt = torch.load(args.pretrained_ckpt)
    load_weights_v2(model, ckpt, 'network_fn_state_dict')
    load_weights_v2(model_fine, ckpt, 'network_fine_state_dict')
    print(f'Load pretrained ckpt successfully: "{args.pretrained_ckpt}".')

    # set up training args
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # set up testing args
    render_kwargs_test = {
        k: render_kwargs_train[k]
        for k in render_kwargs_train
    }
    render_kwargs_test['perturb'] = args.perturb_test
    render_kwargs_test['raw_noise_std'] = 0.

    # get FLOPs and params
    n_params = get_n_params_(model)
    dummy_input = torch.randn(1, input_ch + input_ch_views).to(device)
    n_flops = get_n_flops_(model, input=dummy_input, count_adds=False) * (
        args.N_samples + args.N_samples + args.N_importance)

    print(
        f'Model complexity per pixel: FLOPs {n_flops/1e6:.10f}M, Params {n_params/1e6:.10f}M'
    )
    return render_kwargs_train, render_kwargs_test


def predict_teacher(teacher_args, args, z_vals, rays_o, rays_d, viewdirs, near, far, pytest=True):
    network_fn = teacher_args['network_fn']
    network_fine = teacher_args['network_fine']
    network_query_fn = teacher_args['network_query_fn']
    
    # pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    network_fn.eval()
    network_fine.eval()
    # raw = network_query_fn(pts, viewdirs, network_fine)
    #########################################################################
    t_vals2 = torch.linspace(0., 1., steps=64).to(device)
    z_vals2 = near * (1. - t_vals2) + far * (t_vals2)
    z_vals2 = z_vals2.expand([rays_o.shape[0], 64])

    # @mst: perturbation of depth z, with each depth value at the middle point
    if True:
        # get intervals between samples
        mids = .5 * (z_vals2[..., 1:] + z_vals2[..., :-1])
        upper = torch.cat([mids, z_vals2[..., -1:]], -1)
        lower = torch.cat([z_vals2[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals2.shape).to(device)  # uniform dist [0, 1)

        # Pytest, overwrite u with numpy's fixed random numbers
        if True:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals2.shape))
            t_rand = to_tensor(t_rand)

        z_vals2 = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals2[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals2, rays_d, args.raw_noise_std, args.white_bkgd, pytest=pytest)


    if args.N_importance > 0:

        z_vals_mid = .5 * (z_vals2[..., 1:] + z_vals2[..., :-1])
        z_samples = sample_pdf(z_vals_mid.cpu(),
                               weights[..., 1:-1].cpu(),
                               args.N_importance,
                               det=(args.perturb == 0.),
                               pytest=pytest)
        z_samples = z_samples.detach().to(device)

        z_vals2, _ = torch.sort(
            torch.cat([z_vals2, z_samples], -1), -1)  
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals2[..., :, None]

        run_fn = network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals2, rays_d, args.raw_noise_std, args.white_bkgd, pytest=pytest)

    return rgb_map, disp_map, acc_map, depth_map
    #########################################################################

    # rgb_map, depth_map, alpha_raw, rgb_raw = Teacher_raw2outputs(
    #     raw,
    #     z_vals,
    #     rays_d,
    #     args.raw_noise_std,
    #     args.white_bkgd,
    #     pytest=pytest,
    #     verbose=False)

    # ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'alpha_raw': alpha_raw, 'rgb_raw': rgb_raw}

    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #         print(f"! [Numerical Error] {k} contains nan or inf.")

    # return rgb_map, depth_map, alpha_raw, rgb_raw
    