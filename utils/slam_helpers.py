import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2rendervar(params, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def project_points(points_3d, intrinsics):
    """
    Function to project 3D points to image plane.
    params:
    points_3d: [num_gaussians, 3]
    intrinsics: [3, 3]
    out: [num_gaussians, 2]
    """
    points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_2d = points_2d / points_2d[:, 2:]
    points_2d = points_2d[:, :2]
    return points_2d

def params2silhouette(params):
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2silhouette(params, transformed_pts):
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette


def params2depthplussilhouette(params, w2c):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2depthplussilhouette(params, w2c, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transform_to_frame(params, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose
    
    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    # Get Frame Camera Pose
    if camera_grad:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params['means3D']
    else:
        pts = params['means3D'].detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts

def procrustes(S1, S2,weights=None):

    if len(S1.shape)==4:
        out = procrustes(S1.flatten(0,1),S2.flatten(0,1),weights.flatten(0,1) if weights is not None else None)
        return out[0],out[1].unflatten(0,S1.shape[:2])
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (BxNx3) closest to a set of 3D points, S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale. / mod : assuming scale is 1
    i.e. solves the orthogonal Procrutes problem.
    '''
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        if weights is not None:
            weights=weights.permute(0,2,1)
        transposed = True

        if weights is None:
            weights = torch.ones_like(S1[:,:1])

        # 1. Remove mean.
        weights_norm = weights/(weights.sum(-1,keepdim=True)+1e-6)
        mu1 = (S1*weights_norm).sum(2,keepdim=True)
        mu2 = (S2*weights_norm).sum(2,keepdim=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        diags = torch.stack([torch.diag(w.squeeze(0)) for w in weights]) # does batched version exist?

        # 3. The outer product of X1 and X2.
        K = (X1@diags).bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 6. Recover translation.
        t = mu2 - ((R.bmm(mu1)))

        # 7. Error:
        S1_hat = R.bmm(S1) + t

        # Combine recovered transformation as single matrix
        R_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        R_[:,:3,:3]=R
        T_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        T_[:,:3,-1]=t.squeeze(-1)
        S_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        transf = T_@S_@R_

        return (S1_hat-S2).square().mean(),transf