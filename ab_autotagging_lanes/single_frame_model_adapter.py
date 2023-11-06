import argparse
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import yaml
import sys


class SFLaneDetectionModelParams():
    def __init__(self) -> None:

        with open('ab_autotagging_lanes/conf/single_frame_model_conf.yaml') as file:
            conf = yaml.safe_load(file)

        with open('ab_autotagging_lanes/conf/repos_path.yaml') as file:
            repos_paths = yaml.safe_load(file)

        self.cfg = conf['cfg']
        self.weights = conf['weights']
        self.model_type = conf['model_type']
        self.multiprocess = conf['multiprocess']
        self.inference_conf = str(
            Path(repos_paths['autobrains_networks']).joinpath(conf['inference_conf']))
        
        if str(repos_paths['autobrains_networks']) not in sys.path:
            sys.path.append(str(repos_paths['autobrains_networks'])) 
        if str(repos_paths['cartica_utils']) not in sys.path:
            sys.path.append(str(repos_paths['cartica_utils'])) 


class SFLaneDetectionInfereResults():
    left_boundaries_ys: List[float]
    right_boundaries_ys: List[float]


class LaneDetectionModel():
    def __init__(self, params=None):
        self.params = params if params is not None else SFLaneDetectionModelParams()
        pass

    @staticmethod
    def read_and_parse_results(results_path: str):
        import pandas as pd

        lanes_summary_path = Path(results_path).joinpath(
            'summary/FinalResults_lanes.txt')
        df_lanes = pd.read_csv(lanes_summary_path, sep='\t', header=0)
        df_lanes = df_lanes[['name', 'x_center',
                             'y_center', 'conf', 'anchor_idx', 'angle']]
        df_lanes.rename(columns={'conf': 'score'}, inplace=True)

        rbs_summary_path = Path(results_path).joinpath(
            'summary/FinalResults_rbs.txt')
        df_rbs = pd.read_csv(rbs_summary_path, sep='\t', header=0)
        df_rbs = df_rbs[['name', 'x_center',
                         'y_center', 'conf', 'anchor_idx', 'angle']]
        df_rbs.rename(columns={'conf': 'score'}, inplace=True)

        return df_lanes, df_rbs

    @staticmethod
    def _plot_inferenced_images(results_path: str, max=20):

        images_dir_path = Path(results_path).joinpath('images')

        from PIL import Image

        images_names = os.listdir(images_dir_path)
        images_names = images_names[:min(max, len(images_names))]
        for image_name in images_names:
            with Image.open(images_dir_path.joinpath(image_name)) as im:
                plt.figure(figsize=(16, 8))
                plt.imshow(im)
                plt.show()

    def infer(self, output_dir: str, input_dirs: List[str], plot=False):
        """
        Apply lane detection infer on given input dirs, and store the results in the output dir.
        """
        from autobrains_networks.lane_detection import inference as LD_inference

        assert len(input_dirs) > 0, "Currently working with one input directory."

        lane_detection_args = vars(self.params)
        lane_detection_args['input_dirs'] = input_dirs
        lane_detection_args['output_dir'] = output_dir
        LD_inference._set_inference_args(lane_detection_args)

        if lane_detection_args['multiprocess']:
            import multiprocessing
            multiprocessing.set_start_method("spawn")

        for input_dir in lane_detection_args['input_dirs']:
            print(f'Inference on directory: {input_dir}')
            lane_detection_args['input_dir'] = input_dir
            LD_inference.lane_detection_inference(lane_detection_args)

        dir_name = Path(input_dirs[0]).parent.name
        results_path = Path(output_dir).joinpath(
            f'{dir_name}/0.75/3_680_1360_8_234_1344_352')  # TODO: build this

        if plot:
            self._plot_inferenced_images(results_path=results_path)

        df_lanes, df_rbs = self.read_and_parse_results(results_path=results_path)
        return df_lanes, df_rbs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str,
                        default='/home/ubuntu/workspace/datasets/small_dataset/images', required=True)
    parser.add_argument('--output_dir', default=str(Path(os.getcwd()
                                                         ).joinpath('tmp/single_frame/output')), type=str, required=False)
    args = parser.parse_args()

    import autobrains_networks
    autobrains_networks_path = Path(autobrains_networks.__file__).parent
    inference_conf_path = autobrains_networks_path.joinpath(
        'lane_detection/inference_conf.json')

    params = SFLaneDetectionModelParams(
        cfg=args.cfg_path,
        weights=args.weights_path,
        inference_conf=str(inference_conf_path))

    ld_model = LaneDetectionModel(params=params)
    ld_model.infer(input_dirs=[args.input_dir],
                   output_dir=args.output_dir, plot=False)
