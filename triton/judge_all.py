import os
import torch
import numpy as np
from tabulate import tabulate
from dataclasses import dataclass


@dataclass
class GroundTruth:
    reference_in: bool
    triton_in: bool


def test_in(a, left, right):
    a = a.cpu().numpy().reshape(left.shape)
    out_of_range = (a < left) | (a > right)
    count = np.sum(out_of_range)
    all_count = np.prod(a.shape)
    in_range = count == 0
    return in_range, f"{count}/{all_count}"


def test_dir(dir):
    reference_dir = f"./{dir}/reference.pt"
    interval_dir = f"./{dir}/interval.npy"
    if not os.path.exists(reference_dir) or not os.path.exists(interval_dir):
        return None  # Special case for missing files

    triton_reference = torch.load(reference_dir, weights_only=True)
    reference = triton_reference["torch_output"]
    triton_output = triton_reference["triton_output"]

    with open(interval_dir, "rb") as f:
        left = np.load(f, allow_pickle=True)
        right = np.load(f, allow_pickle=True)

    triton_ok, triton_count = test_in(triton_output, left, right)
    reference_ok, reference_count = test_in(reference, left, right)

    return triton_ok, reference_ok, triton_count, reference_count


cases = {
    "1924": GroundTruth(reference_in=True, triton_in=True),
    "5990": GroundTruth(reference_in=True, triton_in=True),
    "3017": GroundTruth(reference_in=True, triton_in=False),
    "5895": GroundTruth(reference_in=True, triton_in=True),
    "4551": GroundTruth(reference_in=True, triton_in=True),
    "1190": GroundTruth(reference_in=True, triton_in=True),
    "2843": GroundTruth(reference_in=True, triton_in=True),
    "1960": GroundTruth(reference_in=True, triton_in=True),
    "2680b": GroundTruth(reference_in=True, triton_in=True),
    "376": GroundTruth(reference_in=True, triton_in=True),
    "5065": GroundTruth(reference_in=True, triton_in=True),
    "1840": GroundTruth(reference_in=True, triton_in=True),
    "1666": GroundTruth(reference_in=True, triton_in=False),
    "1937": GroundTruth(reference_in=True, triton_in=True),
    "1671": GroundTruth(reference_in=True, triton_in=True),
    "1821": GroundTruth(reference_in=True, triton_in=False),
    "4701": GroundTruth(reference_in=True, triton_in=False),
    "1808": GroundTruth(reference_in=True, triton_in=True),
    "1578": GroundTruth(reference_in=True, triton_in=True),
    "6227": GroundTruth(reference_in=True, triton_in=True),
}

# 表格数据收集
table = []
for case, gt in cases.items():
    result = test_dir(case)
    if result is None:
        table.append([case, "-", "-", "-", "-", "-", "-", "-"])
        continue
    triton_ok, ref_ok, triton_count, ref_count = result
    triton_str = "✓ In Range" if triton_ok else "✗ Out of Range"
    ref_str = "✓ In Range" if ref_ok else "✗ Out of Range"

    triton_gt_str = "✓ Expected In" if gt.triton_in else "✗ Expected Out"
    ref_gt_str = "✓ Expected In" if gt.reference_in else "✗ Expected Out"

    triton_match = "✔" if triton_ok == gt.triton_in else "✘"
    ref_match = "✔" if ref_ok == gt.reference_in else "✘"

    table.append(
        [
            case,
            triton_str,
            triton_gt_str,
            triton_match,
            ref_str,
            ref_gt_str,
            ref_match,
            f"{triton_count} / {ref_count}",
        ]
    )

# 打印表格
headers = [
    "Issue ID",
    "Triton Output",
    "Triton GT",
    "Triton Match",
    "Reference Output",
    "Reference GT",
    "Ref Match",
    "Out-of-Range Count (Triton / Ref)",
]

print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

import numpy as np
import torch

print("==== Max Error Info Per Case ====")
print(
    f"{'Case':<8} {'MaxErrIdx':<10} {'Output':>14} {'Lower':>14} {'Upper':>14} {'Center':>14} {'AbsError':>14} {'OutOfRange':>12}"
)

for case, gt in cases.items():
    result = test_dir(case)
    if result is None:
        continue

    reference_dir = f"./{case}/reference.pt"
    interval_dir = f"./{case}/interval.npy"

    triton_reference = torch.load(reference_dir, weights_only=True)
    triton_output = triton_reference["triton_output"]
    with open(interval_dir, "rb") as f:
        left = np.load(f, allow_pickle=True)
        right = np.load(f, allow_pickle=True)

    triton_output = triton_output.cpu().numpy().reshape(left.shape)
    center = (left + right) / 2
    diff = np.abs(triton_output - center)

    idx = np.unravel_index(np.argmax(diff), diff.shape)
    output_val = triton_output[idx]
    left_val = left[idx]
    right_val = right[idx]
    center_val = center[idx]
    abs_err = diff[idx]
    out_of_range = not (left_val <= output_val <= right_val)

    print(
        f"{case:<8} {str(idx):<10} {output_val:14.7e} {left_val:14.7e} {right_val:14.7e} {center_val:14.7e} {abs_err:14.7e} {str(out_of_range):>12}"
    )
