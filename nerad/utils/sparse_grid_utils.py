import torch
from nerad.utils.io_utils import save_dict, load_pickle
from nerad.model.sampler import ShapeSampler
import mitsuba as mi
from pathlib import Path
import os

def octetize(a):
    l = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    octets = torch.tensor(l).to(a.device)
    return a[:,None,:] + octets


def getAxesInds(pTens, dimSize):
        a = torch.floor(pTens*dimSize)
        a[a>=dimSize] = dimSize-1
        a[a<0] = 0

        diffs = (pTens*dimSize - a)
        return a, diffs


def create_sparse_grid(scene, gridSize, shape_sampler):

    indice_prev, indice, itr, consecutive_iters_no_increase = 0, 0 ,0, 0

    print(str(gridSize) + ": buidling the sparse grid hash-table ...")
    all_points = torch.tensor([[-1000, -1000, -1000]]).int().cuda()
    while(True):
        si, _ = shape_sampler.sample_input(scene=scene, n=1000000, seed=itr)
        assert isinstance(si, mi.SurfaceInteraction3f)
        #pTens = si.p.torch()
        bsdf = si.bsdf()
        bsdf_sample, _ = bsdf.sample(mi.BSDFContext(), si,
                                               shape_sampler.sampler.next_1d(),
                                               shape_sampler.sampler.next_2d(),
                                               si.is_valid())
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        si_bsdf = scene.ray_intersect(ray,
                                 ray_flags=mi.RayFlags.All,
                                 coherent=True)




        pTens = (si.p - scene.bbox().min) / (scene.bbox().max - scene.bbox().min)
        pTens_bsdf = (si_bsdf.p - scene.bbox().min) / (scene.bbox().max - scene.bbox().min)
        pTens = torch.cat([pTens.torch(), pTens_bsdf.torch()], dim = 0)

        assert pTens.max()<=1.01 and pTens.min()>=-0.01

        axInds,_ = getAxesInds(pTens, gridSize)
        octet_axInds = octetize(axInds)
        axInds = torch.floor(octet_axInds.view(-1,3)).int()

        all_points = torch.cat([all_points,axInds], dim = 0).unique(dim = 0)

        indice = all_points.shape[0]

        if(indice == indice_prev):
            consecutive_iters_no_increase+=1
        else:
            consecutive_iters_no_increase = 0
        if(consecutive_iters_no_increase == 100 or indice== (gridSize+1)*(gridSize+1)*(gridSize+1)+1):
            break
        if(itr % 50 ==0):
            print("itr " + str(itr) + ": voxels found: "+ str(indice))
        itr +=1
        indice_prev = indice


    print(str(gridSize) + ": Sparse grid found after "+ str(itr) + " iterations. Voxels containing surface: " + str(indice))
    return all_points


def create_occupancy_cache(scene, scene_path, max_grid_size):
    sampler = ShapeSampler(scene, no_specular_samples=False)
    grid_folder = Path(scene_path).parent / 'sparse_grids'

    gridSize = 2
    big_dict = {}
    while gridSize <= max_grid_size:
        grid_path = grid_folder / f'grid_{gridSize}.pkl'
        if grid_path.exists():
            sparse_grid = load_pickle(grid_path)
        else:
            sparse_grid = create_sparse_grid(scene, gridSize, sampler)
            os.makedirs(grid_path.parent, exist_ok=True)
            save_dict(sparse_grid, grid_path)

        big_dict[gridSize] = sparse_grid
        gridSize*=2
    return big_dict
