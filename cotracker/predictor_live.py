# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from tqdm import tqdm
from cotracker.models.core.cotracker.cotracker import get_points_on_a_grid
from cotracker.models.core.cotracker.cotracker_live import CoTrackerLive
from cotracker.models.core.model_utils import smart_cat

def _build_cotracker(
    queries,
    device,
    stride,
    sequence_len,
    checkpoint=None,
):
    cotracker = CoTrackerLive(
        queries=queries,
        device=device,
        stride=stride,
        S=sequence_len,
        add_space_attn=True,
        space_depth=6,
        time_depth=6,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict)
    return cotracker


class CoTrackerPredictor(torch.nn.Module):
    def __init__(
        self,
        device,
        checkpoint="cotracker/checkpoints/cotracker_stride_4_wind_8.pth",
        queries=None,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0
    ):
        super().__init__()
        self.interp_shape = (384, 512)

        self.support_grid_size = 6
        self.add_support_grid = add_support_grid

        self.checkpoint = checkpoint

        self.queries = self._refine_queries(device, queries, segm_mask, grid_size, grid_query_frame)

        self.model = _build_cotracker(self.queries, device, 4, 8, self.checkpoint)
        # self.model.to(device)
        self.model.eval()

    def _refine_queries(self, device, queries, segm_mask, grid_size, grid_query_frame):
        if queries is not None:
            queries = queries.clone()
            B, N, D = queries.shape
            assert D == 3
            queries[:, :, 1] *= self.interp_shape[1] / W
            queries[:, :, 2] *= self.interp_shape[0] / H
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=device)
            if segm_mask is not None:
                segm_mask = F.interpolate(
                    segm_mask, tuple(self.interp_shape), mode="nearest"
                )
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            )

        if self.add_support_grid:
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape, device=device)
            grid_pts = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
            )
            queries = torch.cat([queries, grid_pts], dim=1)
        
        return queries


    @torch.no_grad()
    def forward(
        self,
        video,  # (1, T, 3, H, W)
    ):
        tracks, visibilities = self._compute_sparse_tracks(video)

        return tracks, visibilities

    def _compute_sparse_tracks(
        self,
        video,  # (1, T, 3, H, W)
    ):
        B, T, C, H, W = video.shape
        assert B == 1

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear")
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        tracks, __, visibilities = self.model(rgbs_seq=video, iters=6)

        if self.add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size ** 2]
            visibilities = visibilities[:, :, : -self.support_grid_size ** 2]
        thr = 0.9
        visibilities = visibilities > thr

        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])
        return tracks, visibilities