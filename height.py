import cv2
import io
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from libs import remove_border_labels
from libs import merge_periodic_labels
from libs import remove_minmax_area
from libs import watermark_style


@st.cache_data(show_spinner=False)
def make_figure_heightdependence_png(height, quantity, use_log_area, disp_h0, xlabel, ylabel, show_text, color):
    fig_num_size, ax1 = plt.subplots(figsize=(10, 5))
    # ax1.plot(height, quantity, 'r-o')
    ax1.plot(height, quantity, color=color, marker="o")
    # ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel(xlabel)
    # ax1.set_ylabel(ylabel, color='r')
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(bottom=0)

    if use_log_area:
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1.0)

    ax1.axvline(disp_h0, color='black', linestyle='--', linewidth=2)
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)
    fig_num_size.text(s=show_text, **watermark_style)
    fig_num_size.tight_layout()
    buf = io.BytesIO()
    fig_num_size.savefig(buf, format="png")
    plt.close(fig_num_size)

    return buf.getvalue()


def height_analysis(img, h_val, min_a, max_a, s_xy, full_image_area_nm2, mode, tresh_type):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, mask = cv2.threshold(blur, h_val, 255, tresh_type)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels = cv2.connectedComponents(mask)
    num, labels = remove_minmax_area(num, labels, min_a, max_a)

    mask = np.array(mask, dtype=np.uint8)
    if mode == "periodic_conditions":
        num, labels = merge_periodic_labels(labels)

    elif mode == "exclude_border":
        labels = remove_border_labels(labels)
        labels = np.array(labels, dtype=np.uint8)
        num, labels = cv2.connectedComponents((labels > 0).astype(np.uint8))

    else:
        labels = np.array(labels, dtype=np.uint8)
        num, labels = cv2.connectedComponents((labels > 0).astype(np.uint8))

    data_list = []
    centroids = []
    radii = []
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl == 0: continue
        grain_mask = (labels == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = cnts[0]
        area_px = cv2.contourArea(cnt)
        area_nm2 = area_px * (s_xy ** 2)
        perimeter_nm = cv2.arcLength(cnt, True) * s_xy
        equiv_d = np.sqrt(4 * area_nm2 / np.pi)
        circularity = (4 * np.pi * area_nm2) / (perimeter_nm ** 2) if perimeter_nm > 0 else 0

        if len(cnt) >= 5:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            major = max(MA, ma) * s_xy
            minor = min(MA, ma) * s_xy
            aspect = major / minor if minor > 0 else 0
            theta_folded = np.minimum(angle, 180 - angle)
        else:
            major = np.nan
            minor = np.nan
            aspect = np.nan
            theta_folded = np.nan
            angle = np.nan

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centroids.append([(M["m10"] / M["m00"]) * s_xy, (M["m01"] / M["m00"]) * s_xy])
            radii.append(equiv_d / 2)

        data_list.append({
            "Area (nm^2)": area_nm2,
            "Diameter (nm)": equiv_d,
            "Perimeter (nm)": perimeter_nm,
            "Circularity": circularity,
            "Major axis of elips (nm)": major,
            "Minor axis of elips (nm)": minor,
            "Aspect Ratio": aspect,
            "Angle of elips": angle,
            "Fold angle": theta_folded,
        })

    df = pd.DataFrame(data_list)
    mean_values = df.copy()
    if not df.empty:
        mean_values = df.mean(numeric_only=True)
        num_grains = len(centroids)
        if num_grains > 1:
            tree = cKDTree(centroids)
            nnd, idx = tree.query(centroids, k=num_grains)
            edge_distances = []
            center_distances = []
            for i in range(num_grains):
                for k in range(1, num_grains):
                    j = idx[i, k]
                    center_dist = nnd[i, k]
                    edge_dist = center_dist - (radii[i] + radii[j])

                    if edge_dist > 0.1:
                        center_distances.append(center_dist)
                        edge_distances.append(edge_dist)
                        break

            mean_nn = np.mean(center_distances)
            mean_nn_grains = np.mean(edge_distances)
            rho = num_grains / full_image_area_nm2
            R_centroids = mean_nn / (0.5 / np.sqrt(rho))
            R_grains = mean_nn_grains / (0.5 / np.sqrt(rho))

        else:
            mean_nn = np.nan
            mean_nn_grains = np.nan
            rho = num_grains / full_image_area_nm2
            R_centroids = np.nan
            R_grains = np.nan

        mean_values["Nearest Neighbor (nm)"] = mean_nn
        mean_values["Edge Distance (nm)"] = mean_nn_grains
        mean_values["R_centroids"] = R_centroids
        mean_values["R_grains"] = R_grains
        mean_values["rho"] = rho * 1e14
        mean_values["num_grains"] = num_grains

        # print(mean_values)
        # df1 = pd.DataFrame(mean_values)
        # print(df1)

    return mean_values
    # return pd.DataFrame(mean_values)


