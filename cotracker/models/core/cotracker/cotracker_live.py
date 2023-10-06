import torch

from cotracker.models.core.cotracker.cotracker import CoTracker
from cotracker.models.core.model_utils import meshgrid2d, bilinear_sample2d, smart_cat

class CoTrackerLive(CoTracker):
    def __init__(
        self,
        queries,
        device,
        S=8,
        stride=8,
        add_space_attn=True,
        num_heads=8,
        hidden_size=384,
        space_depth=12,
        time_depth=12,
    ):
        super(CoTrackerLive, self).__init__(S, 
                                            stride=stride, 
                                            add_space_attn=add_space_attn, 
                                            num_heads=num_heads, 
                                            hidden_size=hidden_size,
                                            space_depth=space_depth,
                                            time_depth=time_depth)
        
        B, self.N, __ = queries.shape
        assert B == 1
        
        self.device = device
        self.fmaps_ = None
        self.coords = None
        self.vis = None
        self.feat_init = None
        self.queries = queries

        # INIT for the first sequence
        # We want to sort points by the first frame they are visible to add them to the tensor of tracked points consequtively
        self.first_positive_inds = queries[:, :, 0].long()

        __, self.sort_inds = torch.sort(self.first_positive_inds[0], dim=0, descending=False)
        self.inv_sort_inds = torch.argsort(self.sort_inds, dim=0)
        self.first_positive_sorted_inds = self.first_positive_inds[0][self.sort_inds]

        assert torch.allclose(
            self.first_positive_inds[0], self.first_positive_inds[0][self.sort_inds][self.inv_sort_inds]
        )

        coords_init = queries[:, :, 1:].reshape(B, 1, self.N, 2).repeat(
            1, self.S, 1, 1
        ) / float(self.stride)

        # these are logits, so we initialize visibility with something that would give a value close to 1 after softmax
        vis_init = torch.ones((B, self.S, self.N, 1), device=device).float() * 10

        self.coords_init_ = coords_init[:, :, self.sort_inds].clone()
        self.vis_init_ = vis_init[:, :, self.sort_inds].clone()

        self.ind = 0
        self.prev_wind_idx = 0

    def forward(self, rgbs_seq, iters=4, feat_init=None, is_train=False):
        B, T, C, H, W = rgbs_seq.shape
        
        assert B == 1
        assert T == self.S

        rgbs_seq = 2 * (rgbs_seq / 255.0) - 1.0

        # account for remaining frames that do not fill up a whole new window
        S = S_local = rgbs_seq.shape[1]
        if S < self.S:
            rgbs_seq = torch.cat(
                [rgbs_seq, rgbs_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                dim=1,
            )
            S = rgbs_seq.shape[1]
        rgbs_ = rgbs_seq.reshape(B * S, C, H, W)

        if self.fmaps_ is None:
            self.fmaps_ = self.fnet(rgbs_)
        else:
            self.fmaps_ = torch.cat(
                [self.fmaps_[self.S // 2 :], self.fnet(rgbs_[self.S // 2 :])], dim=0
            )
        fmaps = self.fmaps_.reshape(
            B, S, self.latent_dim, H // self.stride, W // self.stride
        )

        curr_wind_points = torch.nonzero(self.first_positive_sorted_inds < self.ind + self.S)
        if curr_wind_points.shape[0] == 0:
            self.ind = self.ind + self.S // 2
            raise RuntimeError("No tracked points")
        wind_idx = curr_wind_points[-1] + 1

        if wind_idx - self.prev_wind_idx > 0:
            fmaps_sample = fmaps[
                :, self.first_positive_sorted_inds[self.prev_wind_idx:wind_idx] - self.ind
            ]

            feat_init_ = bilinear_sample2d(
                fmaps_sample,
                self.coords_init_[:, 0, self.prev_wind_idx:wind_idx, 0],
                self.coords_init_[:, 0, self.prev_wind_idx:wind_idx, 1],
            ).permute(0, 2, 1)

            feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
            self.feat_init = smart_cat(self.feat_init, feat_init_, dim=2)

        if self.prev_wind_idx > 0:
            new_coords = self.coords[-1][:, self.S // 2 :] / float(self.stride)

            self.coords_init_[:, : self.S // 2, :self.prev_wind_idx] = new_coords
            self.coords_init_[:, self.S // 2 :, :self.prev_wind_idx] = new_coords[
                :, -1
            ].repeat(1, self.S // 2, 1, 1)

            new_vis = self.vis[:, self.S // 2 :].unsqueeze(-1)
            self.vis_init_[:, : self.S // 2, :self.prev_wind_idx] = new_vis
            self.vis_init_[:, self.S // 2 :, :self.prev_wind_idx] = new_vis[:, -1].repeat(
                1, self.S // 2, 1, 1
            )

        track_mask_ = torch.arange(self.S, device=self.device) >= 0
        track_mask_ = track_mask_.repeat(self.queries.shape[1]).reshape(1, -1, self.queries.shape[1], 1)

        self.coords, self.vis, __ = self.forward_iteration(
            fmaps=fmaps,
            coords_init=self.coords_init_[:, :, :wind_idx],
            feat_init=self.feat_init[:, :, :wind_idx],
            vis_init=self.vis_init_[:, :, :wind_idx],
            track_mask=track_mask_,
            iters=iters,
        )
        
        traj_e = self.coords[-1][:, :S_local].clone()
        vis_e = torch.sigmoid(self.vis[:, :S_local]).clone()

        self.ind = self.ind + self.S // 2

        self.prev_wind_idx = wind_idx

        return traj_e, self.feat_init, vis_e