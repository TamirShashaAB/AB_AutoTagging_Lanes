
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple

from sklearn.linear_model import LinearRegression

def draw_road_boundaries(ax, df_rbs_single: pd.DataFrame):
    for _, g in df_rbs_single.groupby('anchor_idx'):
        ax.scatter(x=g.x_center, y=g.y_center)

def draw_fitted_curves(ax, df_rbs_single: pd.DataFrame, curves: List[LinearRegression], fit_meta: List[Tuple[int, int, float]]):
    for curve, curve_meta in zip(curves, fit_meta):
        xs, _ = df_rbs_single[df_rbs_single.anchor_idx == curve_meta[0]][['x_center', 'y_center']].values.T
        
        # xs_pred = np.arange(int(xs.min()), int(xs.max()) + 1).reshape((-1, 1))
        # ys_pred = curve.predict(xs_pred)
        xs_pred = np.arange(int(xs.min()), int(xs.max()) + 1)
        ys_pred = curve(xs_pred)

        ax.plot(xs_pred.squeeze(), ys_pred.squeeze(), color='black')

def draw_object_detections_with_in_boundary_indication(ax, df_obj_detections: pd.DataFrame, is_inside_road_boundary: List[bool]):
    for i, row in df_obj_detections.reset_index().iterrows():
        x = row.x_center - row.width / 2
        y = row.y_center - row.height / 2
        color = 'darkgreen' if is_inside_road_boundary[i] else 'red'
        linestyle = 'solid' if is_inside_road_boundary[i] else 'dashed'
        ax.add_patch(Rectangle((x, y), row.width, row.height, alpha=1, fill=None, edgecolor=color, linestyle=linestyle))

def draw_in_boundaries_area(ax, image_size: Tuple, curves: List[LinearRegression]):
    if len(curves) == 0:
        return
    
    ts = np.arange(0, image_size[0], 50)
    # curves_boundaries = [c.predict(ts.reshape(-1, 1)).squeeze() for c in curves]
    curves_boundaries = [c(ts).squeeze() for c in curves]
    boundaries = np.clip(np.max(curves_boundaries, axis=0), 0, image_size[1])
    ax.vlines(ts, [0], boundaries, alpha=0.5)

