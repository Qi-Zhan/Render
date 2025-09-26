import os
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class GroundTruth:
    reference_in: bool
    triton_in: bool


def test_in(a, left, right):
    out_of_range = (a < left) | (a > right)
    count = np.sum(out_of_range)
    all_count = np.prod(a.shape)
    in_range = count == 0
    return in_range, f"{count}/{all_count}"


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


def gex_max(our_left, our_right, torch_output, rq3_left, rq3_right):
    center = (rq3_left + rq3_right) / 2
    index = np.unravel_index(
        np.argmax(np.abs(torch_output - center)), torch_output.shape
    )
    return (
        our_left[index],
        our_right[index],
        torch_output[index],
        rq3_left[index],
        rq3_right[index],
    )


datas = {}

for case, gt in cases.items():
    if not gt.triton_in:
        continue
    reference_dir = f"./{case}/reference.pt"
    interval_dir = f"./{case}/interval.npy"
    interval_rq3_dir = f"./{case}/interval_RQ3.npy"

    triton_reference = torch.load(reference_dir, weights_only=True)
    reference = triton_reference["torch_output"]
    triton_output = triton_reference["triton_output"]

    with open(interval_dir, "rb") as f:
        left = np.load(f, allow_pickle=True)
        right = np.load(f, allow_pickle=True)
    if not os.path.exists(interval_rq3_dir):
        continue
    with open(interval_rq3_dir, "rb") as f:
        left_rq3 = np.load(f, allow_pickle=True)
        right_rq3 = np.load(f, allow_pickle=True)

    triton_output = triton_output.cpu().numpy().reshape(left.shape)
    torch_output = reference.cpu().numpy().reshape(left.shape)

    print(f"Case {case}:")
    triton_in = test_in(triton_output, left_rq3, right_rq3)
    torch_in = test_in(torch_output, left_rq3, right_rq3)
    if (not triton_in[0]) or (not torch_in[0]):
        if not triton_in[0]:
            print("Triton output is not in the interval")
        if not torch_in[0]:
            print("Torch output is not in the interval")
        # find the first element that is out of the range
        for index, (
            torch_o,
            triton_o,
            our_left,
            our_right,
            rq3_left,
            rq3_right,
        ) in enumerate(
            zip(
                torch_output.flatten(),
                triton_output.flatten(),
                left.flatten(),
                right.flatten(),
                left_rq3.flatten(),
                right_rq3.flatten(),
            )
        ):
            if not (rq3_left <= triton_o <= rq3_right) or not (
                rq3_left <= torch_o <= rq3_right
            ):
                print(f"Index: {index}")
                print(f"our: [{our_left}, {our_right}]")
                print(f"rq3: [{rq3_left}, {rq3_right}]")
                print(f"torch: {torch_o}")
                print(f"triton: {triton_o}")
                datas[case] = {
                    "our_left": our_left,
                    "our_right": our_right,
                    "rq3_left": rq3_left,
                    "rq3_right": rq3_right,
                    "torch": torch_o,
                    "triton": triton_o,
                    "overflow": True,
                }
                break
            # else:
            #     assert False, "The output is not in the interval"
    else:
        our_left, our_max, out, rq3_left, rq3_right = gex_max(
            left, right, torch_output, left_rq3, right_rq3
        )
        print(f"our: [{our_left}, {our_max}]")
        print(f"rq3: [{rq3_left}, {rq3_right}]")
        print(f"torch: {out}")
        datas[case] = {
            "our_left": our_left,
            "our_right": our_max,
            "rq3_left": rq3_left,
            "rq3_right": rq3_right,
            "torch": out,
            "triton": out,
            "overflow": False,
        }

# 所有值都 abs
# for case, data in datas.items():
#     data["our_left"] = abs(data["our_left"])
#     data["our_right"] = abs(data["our_right"])
#     data["rq3_left"] = abs(data["rq3_left"])
#     data["rq3_right"] = abs(data["rq3_right"])
#     data["torch"] = abs(data["torch"])
#     data["triton"] = abs(data["triton"])
import matplotlib.pyplot as plt

plot_data = []

# 遍历所有的数据
for case, data in datas.items():
    # 将每个case的数据存入列表
    plot_data.append(
        {
            "case": case,
            "our_left": data["our_left"],
            "our_right": data["our_right"],
            "rq3_left": data["rq3_left"],
            "rq3_right": data["rq3_right"],
            "torch": data["torch"],
            "triton": data["triton"],
            "overflow": data["overflow"],
            "our_left_relative_error": (data["our_left"] - data["triton"])
            / abs(data["triton"]),
            "our_right_relative_error": (data["our_right"] - data["triton"])
            / abs(data["triton"]),
            "rq3_left_relative_error": (data["rq3_left"] - data["triton"])
            / abs(data["triton"]),
            "rq3_right_relative_error": (data["rq3_right"] - data["triton"])
            / abs(data["triton"]),
        }
    )
# 如果relative_error > 100, < -100, 除以 100
# 如果relative_error < 10, < -10, 除以 10
for row in plot_data:
    for key in [
        "our_left_relative_error",
        "our_right_relative_error",
        "rq3_left_relative_error",
        "rq3_right_relative_error",
    ]:
        if row[key] > 100:
            row[key] /= 1000
        if row[key] < -100:
            row[key] /= 1000
        if row[key] > 9:
            row[key] /= 100
        if row[key] < -9:
            row[key] /= 100
        if row[key] > 0.9:
            row[key] /= 10
        if row[key] < -0.9:
            row[key] /= 10
        if -0.01 < row[key] < 0.01:
            row[key] *= 100
        if -0.1 < row[key] < 0.1:
            row[key] *= 10
            

# 将数据转换成 pandas DataFrame，便于使用 seaborn 进行可视化
import pandas as pd
import matplotlib.font_manager as fm
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
prop = fm.FontProperties(fname=font_path)

df = pd.DataFrame(plot_data)

# 绘制图形
fig, ax = plt.subplots(figsize=(8, 4))
# use symlog scale for y axis
ax.set_yscale("symlog")

for i, row in df.iterrows():
    our_left, our_right = row["our_left_relative_error"], row["our_right_relative_error"]
    rq3_left, rq3_right = row["rq3_left_relative_error"], row["rq3_right_relative_error"]
    # 放 text

    plt.text(i-0.1, our_left-0.05, f"{our_left:.2e}", fontsize=4, ha="center", va="bottom", fontproperties=prop)
    plt.text(i-0.1, our_right+0.05, f"{our_right:.2e}", fontsize=4, ha="center", va="top", fontproperties=prop)
    if row["case"] != "1808" and row["case"] != "1578":
        plt.text(i+0.1, rq3_left-0.05, f"{rq3_left:.2e}", fontsize=4, ha="center", va="bottom", fontproperties=prop)
        plt.text(i+0.1, rq3_right+0.05, f"{rq3_right:.2e}", fontsize=4, ha="center", va="top", fontproperties=prop)
    ax.boxplot([our_left, our_right], positions=[i-0.15], widths=0.2)
    ax.boxplot([rq3_left, rq3_right], positions=[i+0.15], widths=0.2)
    # plt.plot(
    #     [i-0.1, i-0.1],
    #     [our_left, our_right],
    #     color="blue",
    #     marker="_",
    #     markersize=2,
    #     label="Our Bound" if i == 0 else "",
    #     lw=0.5,
    # )
    # plt.plot(
    #     [i + 0.1, i + 0.1],
    #     [rq3_left, rq3_right],
    #     color="green",
    #     marker="_",
    #     markersize=2,
    #     label="w/o Mixed Precision" if i == 0 else "",
    #     # label="RQ3 Bound" if i == 0 else "",
    #     lw=0.5,
    # )
    if row["overflow"]:
        plt.scatter(i, 0, color="red", zorder=5, marker="x", s=5)
    else:
        plt.scatter(i, 0, color="blue", zorder=5, marker="D", s=5)

plt.xticks(df.index, df["case"])

ax.set_yticklabels([])
ax.set_xlabel("Case", fontsize=12, fontproperties=prop)
ax.legend()

# 优化布局
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图表
plt.show()
plt.savefig("RQ3.pdf")
