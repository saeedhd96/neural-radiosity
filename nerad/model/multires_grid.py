import torch
from nerad.utils.sparse_grid_utils import getAxesInds,octetize
from nerad.model.embedding import Embedding



class Grid(Embedding):
    def __init__(self,resolution):
        super(Grid, self).__init__()
        self.resolution = resolution
        self.feature_vec = None

    def get_features_at_index(self, index):
        raise NotImplementedError('This is an abstract class! Call one of the children: DenseGrid, SparseGrid')


    def get_octet_features(self,pTens):
        axInds,diffs = getAxesInds(pTens, self.resolution)
        octet_axInds =  octetize(axInds)
        ind = octet_axInds.long()
        fv = self.get_features_at_index(ind)
        return fv, diffs


    def interpolate(self,fv,diffs):
        # Trilinear interpolation
        c000 = fv[:,0,:]
        c100 = fv[:,4,:]
        c001 = fv[:,1,:]
        c101 = fv[:,5,:]
        c010 = fv[:,2,:]
        c110 = fv[:,6,:]
        c011 = fv[:,3,:]
        c111 = fv[:,7,:]
        xd = (diffs[:,0].view(-1,1))
        yd = (diffs[:,1].view(-1,1))
        zd = (diffs[:,2].view(-1,1))
        c00 = c000*(1-xd) + xd*c100
        c01 = c001*(1-xd) + xd*c101
        c10 = c010*(1-xd) + xd*c110
        c11 = c011*(1-xd) + xd*c111
        c0  = c00*(1-yd) +  yd* c10
        c1  = c01*(1-yd) +  yd* c11
        c   = c0*(1-zd)  +  c1* zd
        return c

    def forward(self,x):
        octet_feature, distances = self.get_octet_features(x)
        return self.interpolate(octet_feature, distances)


class DenseGrid(Grid):
    '''
    Original implementation of 'dense grid' mentioned in the paper.
    This was before InstantNGP and the hash grids, but we still provide the implementation for completeness.
    '''
    def __init__(self, resolution, feature_len):
        super(DenseGrid, self).__init__(resolution)
        self.feature_vec = torch.nn.Parameter(torch.zeros((resolution+1),(resolution+1),(resolution+1),feature_len).normal_(0,0.01).cuda())

    def get_features_at_index(self, index):
        return self.feature_vec[index[:,:,0],index[:,:,1],index[:,:,2]]

    def str(self):
        return f'DenseGrid(res={self.resolution}, params={self.feature_vec.shape}'

class SparseGrid(Grid):
    '''
    Original implementation of 'sparse grid' mentioned in the paper.
    This was before InstantNGP and the hash grids, but we still provide the implementation for completeness.
    '''
    def __init__(self,resolution, feature_len, occupancy_cache):
        super(SparseGrid, self).__init__(resolution)
        self.feature_vec = torch.nn.Parameter(torch.zeros(len(occupancy_cache),feature_len).normal_(0,0.01).cuda())
        self.sorted_occupancy_hash, ind = torch.sort(self.hash_function(occupancy_cache))
        self.sorted_sparse = occupancy_cache[ind]

    def get_features_at_index(self, index):
        query_hash = self.hash_function(index)
        query_inds_in_sorted = torch.bucketize(query_hash, self.sorted_occupancy_hash)
        invalid = query_inds_in_sorted>=len(self.feature_vec)
        query_inds_in_sorted[invalid] = 0
        fv = self.feature_vec[query_inds_in_sorted]
        fv[invalid] = 0
        return fv


    def hash_function(self, sparse_grid):
        s = self.resolution + 1
        return (torch.tensor([[s*s, s, 1]]).to(sparse_grid.device)* sparse_grid).sum(dim = -1)

    def __str__(self):
        return f'SparseGrid(res={self.resolution}, params={self.feature_vec.shape}'


class MutliResGrid(Embedding):
    '''
    Original implementation of 'multi resolution grids' mentioned in the paper.
    This was before InstantNGP and the hash grids, but we still provide the implementation for completeness.
    '''
    def __init__(self, otype: str, base_resolution, step_resolution, resolution, feature_len, occupancy_cache=None):
        super(MutliResGrid, self).__init__()
        self.embedding_type  = otype
        self.base_resolution = base_resolution
        self.step_resolution = step_resolution
        self.resolution = resolution
        self.feature_len = feature_len
        self.grids = None
        self.n_output_dims = feature_len
        self.occupancy_cache = occupancy_cache

        self.init_()

    def init_(self):
        is_dense = True
        if 'dense' in self.embedding_type.lower():
            is_dense = True
        elif 'sparse' in self.embedding_type.lower():
            is_dense = False
            _class = SparseGrid
            if self.occupancy_cache is None:
                raise Exception('A pre-computed occupancy map should be passed \
                                 for sparse grid! Set "occupancy" field in embedding!')
        else:
            raise Exception('Grid type not in the available options!')

        current_res = self.base_resolution
        grids = torch.nn.ModuleDict([])
        while current_res<=self.resolution:
            if is_dense:
                kwargs = {}
                _class = DenseGrid
            else:
                occ = self.occupancy_cache.get(current_res)
                if occ is None:
                    raise Exception(f'The occupancy cache does not have resolution({self.resolution})!')
                if occ.max()>self.resolution:
                    raise Exception(f'The passed occupancy cache has entries{self.occupancy_cache.max()} \
                                    above the resolution({self.resolution}) being used!')
                kwargs = {'occupancy_cache': occ}
                _class = SparseGrid

            grids[str(current_res)] = _class(current_res,self.feature_len, **kwargs)
            current_res*=self.step_resolution
        self.grids = grids

    def forward(self,x):
        result = 0
        for _,g in self.grids.items():
            result += g.forward(x)
        return result
