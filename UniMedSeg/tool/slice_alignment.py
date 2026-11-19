"""Stage-one slice matching by foreground overlap and relative position."""

import random

import torch


def aligned_slices(img_3d_group, lab_3d_group, axis, num_samples):
    G = img_3d_group.shape[0]
    selected_indices_matrix = torch.zeros(
        (G, num_samples), dtype=torch.long, device=lab_3d_group.device
    )
    local_target_dim = axis + 1
    dims_to_sum = [0, 1, 2, 3]
    dims_to_sum.remove(local_target_dim)
    group_metadata = []
    for g in range(G):
        lab = lab_3d_group[g]
        slice_areas = lab.sum(dim=tuple(dims_to_sum)).float()
        if slice_areas.max() > 0:
            threshold = 0.1 * slice_areas.max()
            valid_mask = slice_areas > max(threshold, 1.0)
            valid_indices = torch.nonzero(valid_mask).view(-1)
        else:
            valid_indices = torch.tensor([], device=lab_3d_group.device)
        if valid_indices.numel() > 0:
            valid_weights = slice_areas[valid_indices]
            center_idx = (valid_indices.float() * valid_weights).sum() / (
                valid_weights.sum() + 1e-06
            )
            center_idx = center_idx.item()
            length = valid_indices[-1].item() - valid_indices[0].item() + 1.0
            group_metadata.append(
                {
                    "valid": True,
                    "valid_indices": valid_indices,
                    "weights": valid_weights,
                    "lab_ref": lab,
                    "center": center_idx,
                    "length": length,
                    "max_dim": lab.shape[local_target_dim],
                }
            )
        else:
            group_metadata.append(
                {"valid": False, "max_dim": lab.shape[local_target_dim]}
            )
    for n in range(num_samples):
        valid_g_indices = [
            i for (i, meta) in enumerate(group_metadata) if meta["valid"]
        ]
        if not valid_g_indices:
            for g in range(G):
                selected_indices_matrix[g, n] = torch.randint(
                    0, group_metadata[g]["max_dim"], (1,)
                ).item()
            continue
        leader_g = random.choice(valid_g_indices)
        leader_meta = group_metadata[leader_g]
        sample_idx_local = torch.multinomial(
            leader_meta["weights"], 1, replacement=True
        ).item()
        leader_abs_idx = leader_meta["valid_indices"][sample_idx_local].item()
        selected_indices_matrix[leader_g, n] = leader_abs_idx
        leader_lab_slice = torch.index_select(
            leader_meta["lab_ref"],
            local_target_dim,
            torch.tensor([leader_abs_idx], device=lab_3d_group.device),
        ).squeeze(local_target_dim)
        leader_bin = (leader_lab_slice > 0.5).float()
        leader_sum = leader_bin.sum()
        for g in range(G):
            if g == leader_g:
                continue
            follower_meta = group_metadata[g]
            if follower_meta["valid"]:
                all_valid_indices = follower_meta["valid_indices"]
                if all_valid_indices.numel() > 128:
                    subset_indices = all_valid_indices[
                        torch.randperm(all_valid_indices.numel())[:128]
                    ]
                else:
                    subset_indices = all_valid_indices
                candidate_labels = torch.index_select(
                    follower_meta["lab_ref"], local_target_dim, subset_indices
                )
                if local_target_dim == 1:
                    candidate_labels = candidate_labels.permute(1, 0, 2, 3).squeeze(1)
                elif local_target_dim == 2:
                    candidate_labels = candidate_labels.permute(2, 0, 1, 3).squeeze(1)
                elif local_target_dim == 3:
                    candidate_labels = candidate_labels.permute(3, 0, 1, 2).squeeze(1)
                candidate_bin = (candidate_labels > 0.5).float()
                intersection = (candidate_bin * leader_bin).sum(dim=(1, 2))
                candidate_sum = candidate_bin.sum(dim=(1, 2))
                dice_scores = 2.0 * intersection / (leader_sum + candidate_sum + 1e-06)
                (best_val, best_local_idx) = torch.max(dice_scores, dim=0)
                if best_val > 0.05:
                    best_abs_idx = subset_indices[best_local_idx].item()
                else:
                    norm_offset = (leader_abs_idx - leader_meta["center"]) / (
                        leader_meta["length"] * 0.5 + 1e-06
                    )
                    target_float = follower_meta["center"] + norm_offset * (
                        follower_meta["length"] * 0.5
                    )
                    target_idx = int(round(target_float))
                    dists = torch.abs(follower_meta["valid_indices"] - target_idx)
                    nearest_valid_pos = torch.argmin(dists).item()
                    best_abs_idx = follower_meta["valid_indices"][
                        nearest_valid_pos
                    ].item()
                selected_indices_matrix[g, n] = best_abs_idx
            else:
                selected_indices_matrix[g, n] = torch.randint(
                    0, follower_meta["max_dim"], (1,)
                ).item()

    def gather(volumes):
        return torch.stack(
            [
                volume.index_select(local_target_dim, indices).movedim(
                    local_target_dim, 0
                )
                for (volume, indices) in zip(volumes, selected_indices_matrix)
            ]
        )

    return (gather(img_3d_group), gather(lab_3d_group))


def prepare_aligned_batch(imgs_3d, labs_3d, flipped_status_group, ns_list):
    target_in_list = []
    target_out_list = []
    context_in_list = []
    context_label_list = []
    context_flip_list = []
    for axis_i, num_s in enumerate(ns_list):
        if num_s <= 0:
            continue
        (imgs_2d_grp, labs_2d_grp) = aligned_slices(imgs_3d, labs_3d, axis_i, num_s)
        num_rolls = imgs_2d_grp.shape[0]
        K = imgs_2d_grp.shape[1]
        curr_flipped = flipped_status_group.clone()
        for r in range(num_rolls):
            imgs_2d_grp = torch.roll(imgs_2d_grp, shifts=1, dims=0)
            labs_2d_grp = torch.roll(labs_2d_grp, shifts=1, dims=0)
            curr_flipped = torch.roll(curr_flipped, shifts=1, dims=0)
            t_in = imgs_2d_grp[0]
            t_out = labs_2d_grp[0]
            c_in = imgs_2d_grp[1:].permute(1, 0, 2, 3, 4)
            c_lbl = labs_2d_grp[1:].permute(1, 0, 2, 3, 4)
            c_flip = curr_flipped[1:].unsqueeze(0).repeat(K, 1)
            target_in_list.append(t_in)
            target_out_list.append(t_out)
            context_in_list.append(c_in)
            context_label_list.append(c_lbl)
            context_flip_list.append(c_flip)
    if len(target_in_list) == 0:
        return None
    return (
        torch.cat(target_in_list, dim=0),
        torch.cat(target_out_list, dim=0),
        torch.cat(context_in_list, dim=0),
        torch.cat(context_label_list, dim=0),
        torch.cat(context_flip_list, dim=0),
    )
