import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange


def load_observations(points2d_path, device):
    """
    Load points2d.npz.

    Returns:
        obs_xy: (V, N, 2)
        vis:    (V, N)
        keys:   list of view names
    """
    data = np.load(points2d_path)
    keys = sorted(data.files)

    views = []
    for k in keys:
        views.append(data[k].astype(np.float32))

    arr = np.stack(views, axis=0)  # (V, N, 3)

    obs_xy = torch.from_numpy(arr[..., :2]).to(device)
    vis = torch.from_numpy(arr[..., 2]).to(device)

    return obs_xy, vis, keys


def euler_xyz_to_matrix(euler):
    """
    Convert Euler angles to rotation matrices.

    Args:
        euler: (V, 3), each row is [rx, ry, rz]

    Returns:
        R: (V, 3, 3)

    This implementation uses:
        R = Rx @ Ry @ Rz

    For bundle adjustment, the exact Euler convention is acceptable
    as long as it is used consistently during optimization.
    """
    rx = euler[:, 0]
    ry = euler[:, 1]
    rz = euler[:, 2]

    cx = torch.cos(rx)
    sx = torch.sin(rx)

    cy = torch.cos(ry)
    sy = torch.sin(ry)

    cz = torch.cos(rz)
    sz = torch.sin(rz)

    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)

    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cx, -sx], dim=-1),
        torch.stack([zeros, sx, cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sy, zeros, cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cz, -sz, zeros], dim=-1),
        torch.stack([sz, cz, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-2)

    R = Rx @ Ry @ Rz
    return R


def initialize_points_from_observations(obs_xy, vis, f0, distance, cx, cy):
    """
    Initialize 3D points from average 2D observations.

    When R ~= I and T ~= [0, 0, -d]:

        u = -f * X / Zc + cx
        v =  f * Y / Zc + cy

    Since Zc ~= -d:

        u - cx ~= f * X / d
        v - cy ~= -f * Y / d

    Therefore:

        X ~= (u - cx) * d / f
        Y ~= -(v - cy) * d / f

    Z is initialized as small random noise.
    """
    with torch.no_grad():
        valid_count = vis.sum(dim=0).clamp_min(1.0)  # (N,)
        mean_xy = (obs_xy * vis[..., None]).sum(dim=0) / valid_count[:, None]

        x = (mean_xy[:, 0] - cx) * distance / f0
        y = -(mean_xy[:, 1] - cy) * distance / f0
        z = 0.05 * torch.randn_like(x)

        points = torch.stack([x, y, z], dim=-1)

    return points


def project_points(points3d, euler, trans, log_f, image_size):
    """
    Project 3D points into all selected camera views.

    Args:
        points3d: (P, 3)
        euler:    (B, 3)
        trans:    (B, 3)
        log_f:    scalar
        image_size: int

    Returns:
        pred_xy: (B, P, 2)
        z:       (B, P)
    """
    cx = image_size / 2.0
    cy = image_size / 2.0

    f = torch.exp(log_f)

    R = euler_xyz_to_matrix(euler)  # (B, 3, 3)

    # Pc[b, p, :] = R[b] @ points3d[p] + trans[b]
    Pc = torch.einsum("bij,pj->bpi", R, points3d) + trans[:, None, :]

    Xc = Pc[..., 0]
    Yc = Pc[..., 1]
    Zc = Pc[..., 2]

    # In this assignment, valid points should have Zc < 0.
    # Avoid division by zero or positive depth explosion.
    Z_safe = torch.where(Zc < -1e-4, Zc, torch.full_like(Zc, -1e-4))

    u = -f * Xc / Z_safe + cx
    v = f * Yc / Z_safe + cy

    pred_xy = torch.stack([u, v], dim=-1)

    return pred_xy, Zc


def evaluate_rmse(points3d, euler, trans, log_f, obs_xy, vis, image_size, max_points=5000):
    """
    Evaluate pixel RMSE on a subset of points.
    """
    with torch.no_grad():
        V, N, _ = obs_xy.shape
        device = obs_xy.device

        if N > max_points:
            point_ids = torch.randperm(N, device=device)[:max_points]
        else:
            point_ids = torch.arange(N, device=device)

        pred_xy, _ = project_points(
            points3d[point_ids], euler, trans, log_f, image_size)

        obs = obs_xy[:, point_ids]
        mask = vis[:, point_ids] > 0.5

        sq_err = ((pred_xy - obs) ** 2).sum(dim=-1)
        rmse = torch.sqrt(sq_err[mask].mean())

    return rmse.item()


def save_colored_obj(path, points3d, colors):
    """
    Save colored point cloud as OBJ.

    OBJ vertex format:
        v x y z r g b
    """
    points = points3d.detach().cpu().numpy()

    colors = colors.astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0

    with open(path, "w") as f:
        for p, c in zip(points, colors):
            f.write(
                f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
            )


def plot_curve(xs, ys, path, ylabel):
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs_task1")

    parser.add_argument("--image_size", type=int, default=1024)

    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--batch_views", type=int, default=10)
    parser.add_argument("--batch_points", type=int, default=4096)

    parser.add_argument("--init_f", type=float, default=900.0)
    parser.add_argument("--init_distance", type=float, default=2.5)

    # If views are ordered from left to right, yaw initialization can help.
    # Try 0 first. If convergence is poor, try 40 or 70.
    parser.add_argument("--yaw_init_deg", type=float, default=0.0)

    parser.add_argument("--lr_points", type=float, default=1e-2)
    parser.add_argument("--lr_camera", type=float, default=1e-3)
    parser.add_argument("--lr_focal", type=float, default=1e-3)

    parser.add_argument("--depth_weight", type=float, default=0.01)
    parser.add_argument("--pose_reg_weight", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=10.0)

    parser.add_argument("--eval_every", type=int, default=100)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)

    points2d_path = os.path.join(args.data_dir, "points2d.npz")
    colors_path = os.path.join(args.data_dir, "points3d_colors.npy")

    print("Loading observations...")
    obs_xy, vis, keys = load_observations(points2d_path, device)

    V, N, _ = obs_xy.shape
    print(f"Num views: {V}")
    print(f"Num points: {N}")
    print(f"Observation tensor: {obs_xy.shape}")
    print(f"Visible observations: {(vis > 0.5).sum().item()}")

    cx = args.image_size / 2.0
    cy = args.image_size / 2.0

    print("Initializing 3D points...")
    init_points = initialize_points_from_observations(
        obs_xy=obs_xy,
        vis=vis,
        f0=args.init_f,
        distance=args.init_distance,
        cx=cx,
        cy=cy,
    )

    points3d = torch.nn.Parameter(init_points)

    print("Initializing cameras...")
    init_euler = torch.zeros(V, 3, device=device)

    if args.yaw_init_deg != 0.0:
        yaw = torch.linspace(
            -args.yaw_init_deg,
            args.yaw_init_deg,
            V,
            device=device,
        ) * math.pi / 180.0

        # rotation around Y axis
        init_euler[:, 1] = yaw

    euler = torch.nn.Parameter(init_euler)

    init_trans = torch.zeros(V, 3, device=device)
    init_trans[:, 2] = -args.init_distance
    trans = torch.nn.Parameter(init_trans)

    log_f = torch.nn.Parameter(
        torch.log(torch.tensor(args.init_f, device=device)))

    optimizer = torch.optim.Adam([
        {"params": [points3d], "lr": args.lr_points},
        {"params": [euler, trans], "lr": args.lr_camera},
        {"params": [log_f], "lr": args.lr_focal},
    ])

    train_loss_history = []
    rmse_history = []
    rmse_iters = []

    print("Start optimization on CPU...")

    for it in trange(args.iters):
        optimizer.zero_grad()

        if args.batch_views < V:
            view_ids = torch.randperm(V, device=device)[:args.batch_views]
        else:
            view_ids = torch.arange(V, device=device)

        if args.batch_points < N:
            point_ids = torch.randperm(N, device=device)[:args.batch_points]
        else:
            point_ids = torch.arange(N, device=device)

        batch_points = points3d[point_ids]
        batch_euler = euler[view_ids]
        batch_trans = trans[view_ids]

        pred_xy, Zc = project_points(
            points3d=batch_points,
            euler=batch_euler,
            trans=batch_trans,
            log_f=log_f,
            image_size=args.image_size,
        )

        obs = obs_xy[view_ids][:, point_ids]
        mask = vis[view_ids][:, point_ids] > 0.5

        # Normalize pixel residuals by image size for more stable optimization.
        residual = (pred_xy - obs) / args.image_size
        sq_err = (residual ** 2).sum(dim=-1)

        reproj_loss = sq_err[mask].mean()

        # Penalize points that go behind invalid side.
        # In this assignment, valid camera-space depth should satisfy Zc < 0.
        depth_penalty = torch.relu(Zc + 0.05).pow(2).mean()

        # Small pose regularization to avoid unnecessary drift.
        pose_reg = euler.pow(2).mean() + trans[:, :2].pow(2).mean()

        loss = (
            reproj_loss
            + args.depth_weight * depth_penalty
            + args.pose_reg_weight * pose_reg
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [points3d, euler, trans, log_f],
            max_norm=args.grad_clip,
        )

        optimizer.step()

        train_loss_history.append(loss.item())

        if (it % args.eval_every == 0) or (it == args.iters - 1):
            rmse = evaluate_rmse(
                points3d=points3d,
                euler=euler,
                trans=trans,
                log_f=log_f,
                obs_xy=obs_xy,
                vis=vis,
                image_size=args.image_size,
                max_points=5000,
            )

            rmse_history.append(rmse)
            rmse_iters.append(it)

            f_value = torch.exp(log_f).item()

            print(
                f"\nIter {it:05d} | "
                f"loss={loss.item():.6f} | "
                f"rmse={rmse:.3f} px | "
                f"f={f_value:.2f}"
            )

    print("Saving results...")

    result_path = os.path.join(args.out_dir, "ba_result.pt")
    torch.save({
        "points3d": points3d.detach().cpu(),
        "euler": euler.detach().cpu(),
        "trans": trans.detach().cpu(),
        "f": torch.exp(log_f).detach().cpu(),
        "view_keys": keys,
        "args": vars(args),
    }, result_path)

    plot_curve(
        xs=list(range(len(train_loss_history))),
        ys=train_loss_history,
        path=os.path.join(args.out_dir, "training_loss.png"),
        ylabel="Training Loss",
    )

    plot_curve(
        xs=rmse_iters,
        ys=rmse_history,
        path=os.path.join(args.out_dir, "rmse_curve.png"),
        ylabel="Pixel RMSE",
    )

    colors = np.load(colors_path)
    obj_path = os.path.join(args.out_dir, "reconstruction.obj")
    save_colored_obj(obj_path, points3d, colors)

    print("Done.")
    print(f"Saved: {result_path}")
    print(f"Saved: {obj_path}")
    print(f"Saved: {os.path.join(args.out_dir, 'training_loss.png')}")
    print(f"Saved: {os.path.join(args.out_dir, 'rmse_curve.png')}")


if __name__ == "__main__":
    main()
