import pandas as pd
from typing import List
from pathlib import Path
import os

from ab_autotagging_lanes.single_frame_model_adapter import LaneDetectionModel
from ab_autotagging_lanes.road_boundaries_filter import RoadBoundaryFilter, RoadBoundaryFilterParams


def apply_filter_on_detections(detections_df: pd.DataFrame,
                               road_boundaries_df: pd.DataFrame,
                               images_path: str = None,
                               detections_key=lambda df, name: df.name == name,
                               plot=False):

    rbs_filter = RoadBoundaryFilter(
        params=RoadBoundaryFilterParams(images_path=images_path))

    result_df = detections_df.copy()
    result_df['is_inside_road_boundaries'] = None

    for image_name in road_boundaries_df.name.unique():
        df_rbs_single = road_boundaries_df[road_boundaries_df.name == image_name]
        detections_df_single = detections_df[detections_key(
            detections_df, image_name)]

        is_inside_road_boundary, _ = rbs_filter.apply_filter(
            detections=detections_df_single, df_rbs_single=df_rbs_single, plot=plot, image_name=image_name)

        if is_inside_road_boundary is not None:
            result_df.loc[detections_key(
                detections_df, image_name), 'is_inside_road_boundaries'] = is_inside_road_boundary

    return result_df


def run_ld_model_and_apply_filter_on_detections(detections_df: pd.DataFrame,
                                                images_path: str,
                                                output_dir=str(
                                                    Path(os.getcwd()).joinpath('tmp/output')),
                                                detections_key=lambda df, name: df.name == name,
                                                plot=False,
                                                delete_output_dir=False):

    lane_detection_model = LaneDetectionModel()

    _, df_rbs = lane_detection_model.infer(
        input_dirs=images_path, output_dir=output_dir)

    results_df = apply_filter_on_detections(detections_df=detections_df,
                                            road_boundaries_df=df_rbs,
                                            images_path=images_path,
                                            detections_key=detections_key,
                                            plot=plot)

    if delete_output_dir:
        import shutil
        shutil.rmtree(output_dir)

    return results_df


if __name__ == '__main__':
    input_dirs = ['/home/tamir/datasets/small_3']
    output_dir = str(Path(os.getcwd()).joinpath('tmp/output'))
    PREDICTION_PATH = '/home/tamir/workspace/AB_AutoTagging_Lanes/tmp/b2b_pred_with_rbs.tsv'

    results = run_ld_model_and_apply_filter_on_detections(detections_df=pd.read_csv(PREDICTION_PATH, sep='\t'),
                                                          images_path=input_dirs,
                                                          plot=False,
                                                          delete_output_dir=False)

    import numpy as np
    print(f'Total {len(results)}, {np.sum(results.is_inside_road_boundaries.isna())} are None" \
          " / {np.sum(results.is_inside_road_boundaries == True)} inside / {np.sum(results.is_inside_road_boundaries == False)} outside')
