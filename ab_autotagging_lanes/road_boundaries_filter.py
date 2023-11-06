import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from pathlib import Path
import yaml

from sklearn.linear_model import LinearRegression

from ab_autotagging_lanes.single_frame_model_adapter import LaneDetectionModel


class RoadBoundaryFilterParams:
    def __init__(self, images_path=None) -> None:

        self.image_dataset_path = images_path

        with open('ab_autotagging_lanes/conf/rbs_filter_conf.yaml') as file:
            yaml_data = yaml.safe_load(file)

        self.min_points_in_curve = yaml_data['min_points_in_curve']
        self.fit_score_ths = yaml_data['fit_score_ths']
        self.dist_from_curve_ths = yaml_data['dist_from_curve_ths']
        self.minimum_curves = yaml_data['minimum_curves']
        self.maximum_curves_to_use = yaml_data['maximum_curves_to_use']


class RoadBoundaryFilter:
    def __init__(self, params: RoadBoundaryFilterParams) -> None:
        self.params = params

    def _are_points_under_curve(self, points_xs, points_ys, curve: LinearRegression):
        # are_under = curve.predict(points_xs.reshape(
        #     (-1, 1))).squeeze() - points_ys < self.params.dist_from_curve_ths
        are_under = curve(points_xs) - points_ys  < self.params.dist_from_curve_ths
        return are_under

    # def _fit_curves_to_rbs(self, df_rbs_single: pd.DataFrame) -> List[Tuple[int, LinearRegression, int, float]]:
    def _fit_curves_to_rbs(self, df_rbs_single: pd.DataFrame) -> Tuple[List[LinearRegression], List[Tuple[int, int, float]]]:
        rbs_groups = df_rbs_single.groupby('anchor_idx')

        curves = []
        meta = []
        for group_idx, group in rbs_groups:
            # xs, ys = np.array(group.x_center).reshape(
            #     (-1, 1)), np.array(group.y_center).reshape((-1, 1))
            xs, ys = np.array(group.x_center), np.array(group.y_center)

            # If group contains less than {min_points} points, it is too short to be considered.
            group_size = xs.shape[0]
            if group_size < self.params.min_points_in_curve:
                continue

            # curve = LinearRegression().fit(xs, ys)
            # score = curve.score(xs, ys)
            polyfit_coefs, ssr, _, _, _ = np.polyfit(xs, ys, deg=1, full=True)
            curve = np.polynomial.Polynomial(polyfit_coefs[::-1])
            sst = np.sum(np.square(ys - np.mean(ys)))
            score = 1 - ssr[0]/sst

            # If fit score less than {min_points} points it wont be considered.
            if score < self.params.fit_score_ths:
                continue

            curves.append(curve)
            meta.append((group_idx, group_size, score))

        return curves, meta

    def _is_detection_inside_road_boundary(self, detections: pd.DataFrame, curves: List[LinearRegression], fit_meta: List[Tuple[int, int, float]]):
        # If found less than {minimum_curves} rbs curves, wont activate dont care logic.
        if len(curves) < self.params.minimum_curves:
            return np.ones(len(detections))

        # If found more than 2 rbs curves, take into consideration only the 2 'biggest'.
        if len(curves) > self.params.maximum_curves_to_use:
            curves = [c for c, _ in sorted(zip(
                curves, fit_meta), key=lambda pair: -pair[1][1])][:self.params.maximum_curves_to_use]

        are_detections_under_curve_arr = []
        for curve in curves:
            bottom_y = (detections.y_center + detections.height / 2).values
            bottom_left_corner_x = (
                detections.x_center - detections.width / 2).values
            bottom_right_corner_x = (
                detections.x_center + detections.width / 2).values

            bottom_left_under_curve = self._are_points_under_curve(
                points_xs=bottom_left_corner_x, points_ys=bottom_y, curve=curve)
            bottom_right_under_curve = self._are_points_under_curve(
                points_xs=bottom_right_corner_x, points_ys=bottom_y, curve=curve)

            are_detections_under_curve = np.logical_or(
                bottom_left_under_curve, bottom_right_under_curve)
            are_detections_under_curve_arr.append(are_detections_under_curve)

        is_inside_road_boundary = np.prod(
            are_detections_under_curve_arr, axis=0)

        return is_inside_road_boundary

    def apply_filter(self, detections: pd.DataFrame, df_rbs_single: pd.DataFrame, plot=False, image_name=None):

        if len(detections) == 0 or len(df_rbs_single) == 0:
            return None, None

        assert np.all([c in detections.columns for c in ['x_center', 'y_center', 'height', 'width']]
                      ), "Invalid DataFrame. Should contain columns: 'x_center', 'y_center', 'height', 'width'."

        curves, fit_meta = self._fit_curves_to_rbs(df_rbs_single)
        is_inside_road_boundary = self._is_detection_inside_road_boundary(
            detections=detections, curves=curves, fit_meta=fit_meta)

        if plot:
            from ab_autotagging_lanes import visualizations

            assert image_name is not None, 'image_name should be provided'

            image_path = Path(
                self.params.image_dataset_path).joinpath(image_name)
            try:
                img = Image.open(image_path)
            except:
                print(f'No image found in {image_path}')
                return

            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            ax.set_title(image_name)
            ax.imshow(img)

            visualizations.draw_road_boundaries(
                ax, df_rbs_single=df_rbs_single)
            visualizations.draw_fitted_curves(
                ax, df_rbs_single=df_rbs_single, curves=curves, fit_meta=fit_meta)
            visualizations.draw_object_detections_with_in_boundary_indication(
                ax, detections, is_inside_road_boundary=is_inside_road_boundary)
            visualizations.draw_in_boundaries_area(
                ax, image_size=img.size, curves=curves)

            plt.show()

        return is_inside_road_boundary, fit_meta
