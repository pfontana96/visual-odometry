from pathlib import Path
import json
import logging
from time import time
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import yaml

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from tqdm import tqdm

from pyvo.core.cpu_dense_visual_odometry import CPUDenseVisualOdometry
from pyvo.utils import quaternion_to_rotmat, rotmat_to_quaternion, inverse


STDOUT_FORMAT = "%(asctime)s %(name)-4s %(levelname)-4s %(funcName)-8s %(message)s"


_SUPPORTED_BENCHMARKS = ["tum-fr1"]


logger = logging.getLogger(__name__)


def set_root_logger(verbose: bool = False):
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    logger.addHandler(handler)
    formatter = logging.Formatter(STDOUT_FORMAT)
    handler.setFormatter(formatter)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("benchmark", type=str, choices=_SUPPORTED_BENCHMARKS, help="Benchmark to run")
    parser.add_argument("-d", "--data-dir", type=str, help="Path to data", default=None)
    parser.add_argument("-v", "--visualize", action="store_true", help="Plot 3d trajectory")
    parser.add_argument(
        "-s", "--size", type=int, default=None,
        help="Number of data samples to use (first 'size' samples are selected)"
    )
    parser.add_argument("-c", "--config-file", type=str, required=True, help="Dense visual odometry method's configuration YAML")
    parser.add_argument("-i", "--intrinsics-file", type=str, required=True, help="Camera intrinsics configuration YAML")

    args = parser.parse_args()

    set_root_logger()

    gt_transforms, rgb_images, depth_images, additional_info = load_benchmark(
        type=args.benchmark, data_dir=args.data_dir, size=args.size
    )

    cam_intrinsics, depth_scale, height, width = load_camera(args.intrinsics_file)

    dvo = CPUDenseVisualOdometry.load_from_yaml(args.config_file)
    dvo.update_camera_info(cam_intrinsics, height, width, depth_scale)

    additional_info.update({"visualize": args.visualize})

    return dvo, gt_transforms, rgb_images, depth_images, additional_info


def load_camera(filepath: str):
    camera_intrinsics_file = Path(filepath).resolve()
    if not camera_intrinsics_file.exists():
        raise FileNotFoundError("Could not find camera intrinsics at '{}'".format(
            str(camera_intrinsics_file)
        ))

    with camera_intrinsics_file.open("r") as fp:
            data = yaml.load(fp, yaml.Loader)

    try:
        camera_intrinsics = np.array(data["intrinsics"], dtype=np.float32)
        depth_scale = data["depth_scale"]
        height = data["height"]
        width = data["width"]

    except KeyError as e:
        raise ValueError("Invalid intrinsics camera file, does not contain '{}' entry")

    return camera_intrinsics, depth_scale, height, width


def load_benchmark(type: str, data_dir: str = None, size: int = None, pyr_down: bool = False):
    if type == "tum-fr1":
        if data_dir is None:
            raise ValueError("When running 'tum-fr1' path to data (-d) should be specified")

        data_dir = Path(data_dir).resolve()

        if not data_dir.is_dir():
            raise FileNotFoundError("Could not find data dir at '{}'".format(str(data_dir)))

        # Try to get camera intrinsics on dataset directory, if not default to the test one:
        camera_intrinsics_file = data_dir / "camera_intrinsics.yaml"
        if not camera_intrinsics_file.exists():
            logger.warning("No camera intrinsics file for dataset at '{}', defaulting to test one..".format(str(data_dir)))
            camera_intrinsics_file = Path(__file__).resolve().parent.parent / "tests/test_data/camera_intrinsics.yaml"

        gt_transforms, rgb_images, depth_images, additional_info = _load_tum_benchmark(
            data_path=data_dir, size=size
        )

    return gt_transforms, rgb_images, depth_images, additional_info


def _load_tum_benchmark(data_path: Path, size: int = None):
    """Loads TUM RGB-D benchmarks. See https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats

    Parameters
    ----------
    data_path : Path
        Path to dir where benchmark is downloaded.
    size: Amount of images to load

    Returns
    -------
    List[np.ndarray] :
        List of ground truth camera poses.
    List[np.ndarray] :
        List of RGB images.
    List[np.ndarray] :
        List of depth images.
    dict :
        Dictionary containing info about where the loaded images are stored on the filesystem.
    """

    filenames = ["rgb.txt", "depth.txt", "groundtruth.txt"]
    filedata = {}

    # Load data from txt files
    for filename in tqdm(filenames, ascii=True, desc="Reading txt files"):
        filepath = data_path / filename

        if not filepath.exists():
            raise FileNotFoundError("Expected TUM RGB-D dataset to contain a file named '{}' at '{}'".format(
                filename, str(data_path)
            ))

        with filepath.open("r") as fp:
            content = fp.readlines()

        timestamps = []
        data = []
        for line in content:
            # Avoid comments
            if line.startswith('#'):
                continue

            line_stripped = line.rstrip("\r\n")
            line_stripped = line_stripped.split(" ")
            timestamps.append(float(line_stripped[0]))

            # If groundtruth then save SE(3) pose
            if filename == "groundtruth.txt":
                pose = np.eye(4, dtype=np.float32)
                pose[:3,:3] = quaternion_to_rotmat(np.array(line_stripped[4:], dtype=np.float32))
                pose[:3, 3] = np.asarray(line_stripped[1:4], dtype=np.float32)
                data.append(pose)

            else:
                image_path = data_path / line_stripped[1]
                if not image_path.exists():
                    raise FileNotFoundError("Could not find {} image at '{}'".format(filepath.stem, str(image_path)))
                data.append(str(image_path))

        filedata[filepath.stem] = {"timestamp": np.array(timestamps), "data": np.array(data)}

    # Find timestamps correspondances between depth timestamps w.r.t. color timestamps
    logger.info("Finding closest timestamps..")
    distances = np.abs(filedata["rgb"]["timestamp"].reshape(-1, 1) - filedata["depth"]["timestamp"])
    potential_closest = distances.argmin(axis=1)

    ids = np.arange(distances.shape[0])

    # Drop too distant timestamps from rgb and closest depth (0.02 second threshold)
    mask = (distances[ids, potential_closest] < 0.02)
    ids = ids[mask]
    closest_found = potential_closest[mask]

    rgb_timestamps = filedata["rgb"]["timestamp"][ids]
    depth_timestamps = filedata["depth"]["timestamp"][closest_found]

    rgb_images_paths = filedata["rgb"]["data"][ids]
    depth_images_paths = filedata["depth"]["data"][closest_found]
    
    # Avoid duplicates
    # closest_found, ids = np.unique(potential_closest, return_index=True)

    # Find closest groundtruth (ugly average between the timestamps of rgb and dept)
    frames_timestamps = (rgb_timestamps + depth_timestamps) / 2
    closests_groundtruth = np.argmin(
        cdist(frames_timestamps.reshape(-1, 1), filedata["groundtruth"]["timestamp"].reshape(-1, 1)), axis=1
    )
    logger.info("DONE")

    gt_transforms = list(filedata["groundtruth"]["data"][closests_groundtruth])
    gt_timestamps = list(filedata["groundtruth"]["timestamp"][closests_groundtruth])

    if (size is not None) and (size < len(rgb_images_paths)):
        gt_transforms = gt_transforms[:size]
        rgb_images_paths = rgb_images_paths[:size]
        depth_images_paths = depth_images_paths[:size]
        frames_timestamps = frames_timestamps[:size]
        gt_timestamps = gt_timestamps[:size]

    rgb_images = []
    depth_images = []
    for rgb_path, depth_path in tqdm(zip(rgb_images_paths, depth_images_paths), ascii=True, desc="Loading images"):
        rgb_image = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        rgb_images.append(rgb_image)
        depth_images.append(depth_image)

    # Additional data needed for report
    additional_info = {
        "type": "TUM",
        "rgb": rgb_images_paths.tolist(), "depth": depth_images_paths.tolist(),
        "timestamps": gt_timestamps
    }

    return gt_transforms, rgb_images, depth_images, additional_info


def visualize_trajectory(estimated_poses: np.ndarray, ground_truth_poses: np.ndarray):

    # Load Np array with positions (Nx3)
    poses = np.empty((len(estimated_poses), 3), dtype=np.float32)
    gt_poses = np.empty((len(estimated_poses), 3), dtype=np.float32)
    for i, (pose, gt_pose) in enumerate(zip(estimated_poses, ground_truth_poses)):
        poses[i, :] = np.array(pose[:3], dtype=np.float32)
        gt_poses[i, :] = np.array(gt_pose[:3], dtype=np.float32)

    ax = plt.subplot(projection="3d")
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2])
    ax.plot(gt_poses[:, 0], gt_poses[:, 1], gt_poses[:, 2], color="red")

    plt.show()

def main():

    dvo, gt_transforms, rgb_images, depth_images, additional_info = parse_arguments()

    visualize = additional_info.pop("visualize")

    accumulated_transform = gt_transforms[0]
    gt_transforms_tum_fmt = []
    transforms = []
    poses = []
    errors = []
    for i, (rgb_image, depth_image, gt_transform) in enumerate(zip(rgb_images, depth_images, gt_transforms)):

        s = time()
        transform = dvo.step(rgb_image, depth_image, np.eye(4, dtype=np.float32))
        e = time()

        accumulated_transform = accumulated_transform @ inverse(transform)

        # logger.info("Pose: {} Gt:{}".format(accumulated_transform[:3, 3].tolist(), gt_transform))

        # Error is only the euclidean distance (not taking rotation into account)
        if gt_transform is not None:
            t_error = float(np.linalg.norm(accumulated_transform[:3, 3] - gt_transform[:3, 3]))
            logger.info("[Frame {} ({:.3f} s)] | Translational error: {:.4f} m ".format(i + 1, e - s, t_error))
        else:
            t_error = "N/A"
            logger.info("[Frame {} ({:.3f} s)]".format(i + 1, e - s))

        # Store pose in TUM dataset format 'tx ty tz qx qy qz qw'
        pose_tum_fmt = np.empty(7, dtype=np.float32)
        pose_tum_fmt[:3] = accumulated_transform[:3, 3]
        pose_tum_fmt[3:] = rotmat_to_quaternion(accumulated_transform[:3, :3])
        poses.append(pose_tum_fmt.tolist())

        # Store relative transform in TUM dataset format 'tx ty tz qx qy qz qw'
        transform_tum_fmt = np.empty(7, dtype=np.float32)
        transform_tum_fmt[:3] = transform[:3, 3]
        transform_tum_fmt[3:] = rotmat_to_quaternion(transform[:3, :3])
        transforms.append(transform_tum_fmt.tolist())

        # Store ground truth transform in TUM dataset format 'tx ty tz qx qy qz qw'
        gt_tum_fmt = np.empty(7, dtype=np.float32)
        gt_tum_fmt[:3] = gt_transform[:3, 3]
        gt_tum_fmt[3:] = rotmat_to_quaternion(gt_transform[:3, :3])
        gt_transforms_tum_fmt.append(gt_tum_fmt.tolist())

        # Store translational error
        errors.append(t_error)

    # Dump results
    report = {
        "estimated_poses": poses,
        "estimated_transforms": transforms,
        "ground_truth_transforms": gt_transforms_tum_fmt,
        "errors": errors
    }
    report.update(additional_info)

    output_file = Path(__file__).resolve().parent / "report.json"

    with output_file.open("w") as fp:
        json.dump(report, fp, indent=3)

    logger.info("Dumped report at '{}'".format(str(output_file)))

    if visualize:
        visualize_trajectory(poses, gt_transforms_tum_fmt)


if __name__ == "__main__":
    main()
