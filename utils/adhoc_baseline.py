import pandas as pd
import argparse
import os
import numpy as np
import math
import random


def lookup_table_coefficient(p, AR):
    """
    Lookup the empirical correction coefficient alpha based on the pin count (p)
    and aspect ratio (AR) using a two-dimensional lookup table.

    The lookup table 'table' is defined as follows (derived from TABLE V in TCAD99):

        table = {
            p_value: {AR_value: alpha, ...},
            ...
        }

    When p and AR are between two grid points, bilinear interpolation is used.
    If p or AR is outside the range of the table, the boundary value is used.

    Additionally, if p <= 3, the function directly returns 1.0.
    """
    # If p is less than or equal to 3, directly return 1.0
    if p <= 3:
        return 1.0

    # Sample lookup table: keys are p values (e.g., 4-pin and 8-pin),
    # and each value is a dictionary where the key is AR and the value is the corresponding alpha.
    table = {
        4: {1: 1.08, 2: 1.05, 4: 1.03, 10: 1.01},
        5: {1: 1.15, 2: 1.11, 4: 1.07, 10: 1.03},
        6: {1: 1.22, 2: 1.16, 4: 1.11, 10: 1.05},
        8: {1: 1.34, 2: 1.27, 4: 1.18, 10: 1.08},
        10: {1: 1.45, 2: 1.36, 4: 1.25, 10: 1.12},
        15: {1: 1.69, 2: 1.59, 4: 1.41, 10: 1.21},
        20: {1: 1.89, 2: 1.78, 4: 1.57, 10: 1.29},
        30: {1: 2.23, 2: 2.10, 4: 1.84, 10: 1.45}
    }

    # Ensure AR is at least 1
    AR = max(AR, 1.0)

    # Get all p grid points in the table (sorted)
    p_grid = sorted(table.keys())  # e.g., [4, 8]
    # Assume that the AR grid is the same for all p values; use one of them
    ar_grid = sorted(next(iter(table.values())).keys())  # e.g., [1, 2, 4, 8]

    # Determine the lower and upper bounds for the p dimension
    if p <= p_grid[0]:
        p_lower = p_upper = p_grid[0]
    elif p >= p_grid[-1]:
        p_lower = p_upper = p_grid[-1]
    else:
        for i in range(len(p_grid) - 1):
            if p_grid[i] <= p <= p_grid[i + 1]:
                p_lower = p_grid[i]
                p_upper = p_grid[i + 1]
                break

    # Determine the lower and upper bounds for the AR dimension
    if AR <= ar_grid[0]:
        ar_lower = ar_upper = ar_grid[0]
    elif AR >= ar_grid[-1]:
        ar_lower = ar_upper = ar_grid[-1]
    else:
        for i in range(len(ar_grid) - 1):
            if ar_grid[i] <= AR <= ar_grid[i + 1]:
                ar_lower = ar_grid[i]
                ar_upper = ar_grid[i + 1]
                break

    # Get the alpha values at the four grid points:
    # (p_lower, ar_lower), (p_lower, ar_upper), (p_upper, ar_lower), (p_upper, ar_upper)
    f_ll = table[p_lower][ar_lower]  # lower p, lower AR
    f_lu = table[p_lower][ar_upper]  # lower p, upper AR
    f_ul = table[p_upper][ar_lower]  # upper p, lower AR
    f_uu = table[p_upper][ar_upper]  # upper p, upper AR

    # If p is at a single grid point, perform linear interpolation only in the AR direction
    if p_lower == p_upper:
        if ar_lower == ar_upper:
            return f_ll  # Exact match
        else:
            t = (AR - ar_lower) / (ar_upper - ar_lower)
            return f_ll + t * (f_lu - f_ll)

    # If AR is at a single grid point, perform linear interpolation only in the p direction
    if ar_lower == ar_upper:
        t = (p - p_lower) / (p_upper - p_lower)
        return f_ll + t * (f_ul - f_ll)

    # Bilinear interpolation:
    # First, interpolate in the p direction for the two AR grid points
    t = (p - p_lower) / (p_upper - p_lower)
    f_lower = f_ll + t * (f_ul - f_ll)  # alpha corresponding to ar_lower
    f_upper = f_lu + t * (f_uu - f_lu)  # alpha corresponding to ar_upper

    # Then interpolate in the AR direction
    u = (AR - ar_lower) / (ar_upper - ar_lower)
    return f_lower + u * (f_upper - f_lower)


def estimate_net_wirelength_and_buffer(dbu, pin_locs, sink_caps=None,
                                       wireCapPerUnit=0.001,  # fF per micron
                                       bufferMaxCap=5.0,  # fF per buffer
                                       bufferArea=10.0,  # um^2 per buffer
                                       wireDerate=1.25,  # additional wirelength factor (e.g., +25%)
                                       bufferAreaDerate=1.20,  # additional buffer area factor (e.g., +20%)
                                       expandFactor=0.10):  # expansion factor for the bounding box for buffer placement
    """
    Estimate the net wirelength and buffer requirements using the combined approach:
      1. Compute the bounding box of the net's pin locations.
      2. For nets with 2 or 3 pins, use the bounding box half-perimeter directly.
      3. For nets with 4 or more pins, estimate the Steiner tree length as:
            estimated_length = 0.5 * (W + H) * alpha
         where alpha is obtained from an empirical CKMMZ table based on pin count and aspect ratio (AR).
      4. Apply a wire derate factor to account for placement inefficiencies.
      5. Estimate the wire capacitance based on the (derated) length.
      6. Sum the sink capacitances and compute the total capacitance.
      7. Estimate the number of buffers required (totalCap / bufferMaxCap).
      8. Compute the buffer area (and apply an additional derate factor).
      9. Expand the original bounding box by a given expansion factor for buffer "smearing".

    Parameters:
      pin_locs: List of tuples (x, y) for each pin's position.
      sink_caps: List of sink capacitance values for each pin. If None, assume zero for all.
      wireCapPerUnit: Capacitance per unit wire length.
      bufferMaxCap: Maximum capacitance a single buffer can drive.
      bufferArea: Area cost of a single buffer.
      wireDerate: Factor to increase estimated wirelength for pessimism.
      bufferAreaDerate: Factor to increase buffer area for additional overhead.
      expandFactor: Factor to expand the bounding box dimensions for buffer placement.

    Returns:
      A dictionary with keys:
        'estimatedWirelength': Final derated estimated wirelength.
        'wireCap': Estimated wire capacitance.
        'sinkCap': Total sink capacitance.
        'totalCap': Combined capacitance (wire + sink).
        'numBuffers': Estimated number of buffers (may be fractional).
        'bufferArea': Final estimated buffer area.
        'expandedBoundingBox': Tuple (expandedW, expandedH) of the bounding box.
    """
    if not pin_locs:
        return {}

    p = len(pin_locs)

    # Compute the bounding box from pin locations
    xs = [pt[0] for pt in pin_locs]
    ys = [pt[1] for pt in pin_locs]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    W = x_max - x_min
    H = y_max - y_min

    # Estimate the Steiner tree length using the CKMMZ method
    if p <= 3:
        # For 2-pin or 3-pin nets, the bounding box half-perimeter is sufficient.
        estSteinerLength = (W + H)
    else:
        # Calculate the aspect ratio (AR); ensure AR >= 1 by using max/min.
        if W > 0:
            AR = max(W, H) / min(W, H)
        else:
            AR = 1.0
        # Retrieve the empirical correction coefficient alpha
        alpha = lookup_table_coefficient(p, AR)
        # Estimated Steiner length: half-perimeter multiplied by alpha.
        estSteinerLength = (W + H) * alpha

    # Apply the wire derate factor to account for placement inefficiencies.
    estSteinerLength = estSteinerLength / (dbu * 1e+6)
    wireDerate = 3
    deratedLength = estSteinerLength * wireDerate

    # Compute the wire capacitance based on the derated length.
    wireCap = wireCapPerUnit * deratedLength

    # Sum the sink capacitances; if none provided, assume 0.
    if sink_caps is None:
        sinkCap = 0.0
    else:
        sinkCap = sum(sink_caps)

    # Total capacitance = wire capacitance + sink capacitance.
    totalCap = wireCap + sinkCap

    # Estimate the number of buffers required.
    numBuffers = totalCap / bufferMaxCap
    # Optionally, round up if whole buffers are required:
    # numBuffers = math.ceil(numBuffers)

    # Compute the buffer area and apply an additional derating factor.
    rawBufferArea = numBuffers * bufferArea
    finalBufferArea = rawBufferArea * bufferAreaDerate

    # Expand the bounding box dimensions for buffer smearing.
    expandedW = W * (1 + expandFactor)
    expandedH = H * (1 + expandFactor)

    # Return a dictionary containing all the estimated values.
    result = {
        "estimatedWirelength": deratedLength,
        "wireCap": wireCap,
        "sinkCap": sinkCap,
        "totalCap": totalCap,
        "numBuffers": numBuffers,
        "bufferArea": finalBufferArea,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "expandedBoundingBox": (expandedW, expandedH)
    }
    return result


def run_hacky_baseline():
    parser = argparse.ArgumentParser(description="Run hacky baseline.")
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')

    args = parser.parse_args()
    buf_info_file = '/home/fetzfs_projects/MLBuf/flows/OR_branch_integration/buf_data_oneType.csv'
    buf_info_df = pd.read_csv(buf_info_file)
    net_info_df = pd.read_csv(args.input)
    wireCapPerUnit = 8.88758e-11
    dbu = 2000
    bufferRow = buf_info_df[buf_info_df["buf_type"] == "BUF_X2"]
    bufferMaxCap = bufferRow["max_capacitance"].values[0]
    bufferArea = bufferRow["area"].values[0]

    all_lines = []

    # grouped by netname
    for net_name, net_info in net_info_df.groupby("Net Name"):
        pin_positions = np.stack([
            net_info["X"].values,
            net_info["Y"].values,
        ], axis=1)
        drvr_pos = pin_positions[0]
        pin_positions = tuple(map(tuple, pin_positions.tolist()))
        sink_capacities = net_info["Input Cap"].iloc[1:].tolist()

        estimate = estimate_net_wirelength_and_buffer(dbu, pin_positions, sink_capacities,
                                                      wireCapPerUnit=wireCapPerUnit,
                                                      bufferMaxCap=bufferMaxCap,
                                                      bufferArea=bufferArea,
                                                      wireDerate=2,
                                                      bufferAreaDerate=1.20,
                                                      expandFactor=0.10)
        net_lines = generate_buffer_lines(estimate, net_name)
        all_lines.extend(net_lines)

        # print("Estimated Net Wirelength and Buffer Requirements:")
        # for key, value in estimate.items():
        #     print(f"{key}: {value}")
        baseline_buf_count = estimate["numBuffers"]
        save_buffer(estimate, args.output)
    # After processing all nets, write the accumulated lines to the output files.
    # with open(args.output, 'w') as f:
    #     for line in all_lines:
    #         f.write(line)
    # with open('hacky_buffer_save0414.csv', 'w') as f2:
    #     for line in all_lines:
    #         f2.write(line)


def generate_buffer_lines(estimate, net_name):
    """
    Generate buffer bounding box lines for the given net based on the estimation result.
    Returns a list of strings, each representing a line to write to the output file.
    """
    lines = []
    # Round the estimated number of buffers to an integer
    float_buf_count = estimate["numBuffers"]
    final_buf_count = int(round(float_buf_count))
    # print(f"[INFO] Rounded buffer count = {final_buf_count}")

    # Calculate the bounding box for the net based on the estimation results
    x_min = estimate["x_min"]
    x_max = estimate["x_max"]
    y_min = estimate["y_min"]
    y_max = estimate["y_max"]
    expandedW, expandedH = estimate["expandedBoundingBox"]

    # Calculate center and expanded bounding box coordinates
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    halfW_expanded = 0.5 * expandedW
    halfH_expanded = 0.5 * expandedH

    lx = cx - halfW_expanded
    ly = cy - halfH_expanded
    ux = cx + halfW_expanded
    uy = cy + halfH_expanded
    # print(f"[INFO] Expanded BBox corners = (lx={lx}, ly={ly}), (ux={ux}, uy={uy})")

    # Define the base buffer dimensions
    bufferWidth = 0.76
    bufferHeight = 1.4
    # Apply a scaling factor (approximately 10% larger area) to the buffer bounding box
    scale = math.sqrt(1.1)  # Approx. 1.0488
    scaledW = bufferWidth * scale
    scaledH = bufferHeight * scale

    # Randomly assign positions for each buffer within the expanded bounding box
    for i in range(final_buf_count):
        # Randomly select a point ensuring the scaled buffer fits inside the bounding box
        rand_x = random.uniform(lx, ux - scaledW)
        rand_y = random.uniform(ly, uy - scaledH)

        # Adjust to center the base buffer inside the scaled bounding box
        bbox_lx = rand_x - 0.5 * (scaledW - bufferWidth)
        bbox_ly = rand_y - 0.5 * (scaledH - bufferHeight)
        bbox_ux = bbox_lx + scaledW
        bbox_uy = bbox_ly + scaledH

        bbox_area = (bbox_ux - bbox_lx) * (bbox_uy - bbox_ly)

        # Format the output line for the current buffer
        line = f"{bbox_lx},{bbox_ly},{bbox_ux},{bbox_uy},{bbox_area}\n"
        lines.append(line)

    return lines


def save_buffer(estimate, output_file):
    # -------------- 1. save int numBuffers  ----------------
    float_buf_count = estimate["numBuffers"]
    final_buf_count = int(round(float_buf_count))
    # print(f"[INFO] Rounded buffer count = {final_buf_count}")

    # -------------- 2. calculate xy coor of net bounding box  -------------
    x_min = estimate["x_min"]
    x_max = estimate["x_max"]
    y_min = estimate["y_min"]
    y_max = estimate["y_max"]
    expandedW, expandedH = estimate["expandedBoundingBox"]

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    halfW_expanded = 0.5 * expandedW
    halfH_expanded = 0.5 * expandedH

    lx = cx - halfW_expanded
    ly = cy - halfH_expanded
    ux = cx + halfW_expanded
    uy = cy + halfH_expanded

    # print(f"[INFO] Expanded BBox corners = (lx={lx}, ly={ly}), (ux={ux}, uy={uy})")

    # -------------- 3. randomly locate buffers ----------------
    bufferWidth = 0.76
    bufferHeight = 1.4

    # construct buffer bounding box，~ 10%
    scale = math.sqrt(1.1)  # ~1.0488
    scaledW = bufferWidth * scale
    scaledH = bufferHeight * scale

    with open(output_file, 'a') as f:
        for i in range(final_buf_count):
            # randomly locate in [lx, ux - bufferWidth]
            rand_x = random.uniform(lx, ux - scaledW)
            rand_y = random.uniform(ly, uy - scaledH)

            bbox_lx = rand_x - 0.5 * (scaledW - bufferWidth)
            bbox_ly = rand_y - 0.5 * (scaledH - bufferHeight)
            bbox_ux = bbox_lx + scaledW
            bbox_uy = bbox_ly + scaledH

            bbox_area = (bbox_ux - bbox_lx) * (bbox_uy - bbox_ly)

            f.write(f"{bbox_lx},{bbox_ly},{bbox_ux},{bbox_uy},{bbox_area}\n")
            # f2.write(f"{bbox_lx},{bbox_ly},{bbox_ux},{bbox_uy},{bbox_area}\n")

    # print(f"[INFO] Saved {final_buf_count} buffers' bounding boxes to {output_file}")


if __name__ == '__main__':
    run_hacky_baseline()
