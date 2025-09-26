import matplotlib.pyplot as plt
from matplotlib import rcParams

from matplotlib import font_manager as fm

font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
prop = fm.FontProperties(fname=font_path)
rcParams["font.size"] = 14


data = [
    1.0785e2,
    -7.4000e1,
    -4.7770e0,
    -1.3523e1,
    2.0415e3,
    2.8772e2,
    -3.0283e1,
    -1.6775e2,
    7.01882e1,
    3.1499e1,
    6.8300e2,
    -5.1979e1,
    -6.0693e-1,
    -5.1979e1,
    5.3287e-2,
    2.3565e-1,
    -2.9727,
    -1.5188e2,
    8.115862,
    -2.6883e-2,
]

our_approach_intervals = [
    [1.0689e2, 1.0896e2],
    [-7.4043e1, -7.3894e1],
    [-3.57528e1, -3.57526e1],
    [-1.3581e1, -1.3501e1],
    [2.0372e3, 2.0492e3],
    [2.8706e2, 2.8876e2],
    [-3.0457e1, -3.0162e1],
    [-1.6799e2, -1.6766e2],
    [7.0087e1, 7.0226e1],
    [3.1381e1, 3.1666e1],
    [-5.2072e3, 7.1272e3],
    [-5.2307e1, -5.1735e1],
    [1.0001e1, 1.0079e1],
    [-5.2307e1, -5.1735e1],
    [1.5106e-3, 3.4247e-1],
    [3.2001e-1, 3.6569e-1],
    [1.4960, 1.5210],
    [-1.5364e2, -1.5047e2],
    [8.11586, 8.115863],
    [-2.8440e-2, -2.5480e-2],
]

satire_intervals = [
    [-1.3604e1, 2.2945e2],
    [-1.4e2, -7.9],
    [-3.57530e1, -3.57524e1],
    [-1.3755e1, -1.3308e1],
    [1.4661e3, 1.9145e3],
    [0, 0],
    [-3.0309e2, -3.0308e2],
    [-1.7009e2, -1.6555e2],
    [2.3049e1, 1.1726e2],
    [2.5709e2, 3.7343e2],
    [0, 0],
    [-6.7900e1, -3.6140e1],
    [9.922e1, 1.0157e1],
    [-6.7900e1, -3.6140e1],
    [0, 0],
    [0, 0],
    [1.4425e1, 1.5744e1],
    [0, 0],
    [8.115860, 8.115863],
    [0, 0],
]

cases = {
    "1924": 23.41 * 60,
    "5990": 180 * 60,
    "3017": 106.27,
    "5895": 0.6,
    "4551": 250.26,
    "1190": 0,
    "2843": 23.92,
    "1960": 0.529,
    "2680": 171.94,
    "376": 17.68,
    "5065": 0,
    "1840": 630.4319379329681,
    "1666": 9.80,
    "1937": 682.1305477619171,
    "1671": 0,
    "1821": 0,
    "4701": 2.1113882064819336,
    "1808": 0,
    "1578": 3.37,
    "6227": 0,
}
keys = cases.keys()
labels = [f"#{key}" for key in keys]

# remove the cases that are not in the interval
for i in range(len(our_approach_intervals) - 1, -1, -1):
    if not (our_approach_intervals[i][0] <= data[i] <= our_approach_intervals[i][1]):
        del our_approach_intervals[i]
        del satire_intervals[i]
        del labels[i]
        del data[i]
our_left = [interval[0] for interval in our_approach_intervals]
our_right = [interval[1] for interval in our_approach_intervals]
satire_left = [interval[0] for interval in satire_intervals]
satire_right = [interval[1] for interval in satire_intervals]

our_left_error = [abs(left - truth) for left, truth in zip(our_left, data)]
our_right_error = [abs(right - truth) for right, truth in zip(our_right, data)]
satire_left_error = [abs(left - truth) for left, truth in zip(satire_left, data)]
satire_right_error = [abs(right - truth) for right, truth in zip(satire_right, data)]
our_left_relative_error = [
    abs((left - truth) / truth) for left, truth in zip(our_left, data)
]
our_right_relative_error = [
    abs((right - truth) / truth) for right, truth in zip(our_right, data)
]
satire_left_relative_error = [
    abs((left - truth) / truth) for left, truth in zip(satire_left, data)
]
satire_right_relative_error = [
    abs((right - truth) / truth) for right, truth in zip(satire_right, data)
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

for ax in axes:
    ax.set_xscale("log")
    ax.grid(True, zorder=1)
    ax.set_facecolor("#eaeaf1")  # Light background for the axes
    ax.set_yticks(range(len(labels)), labels=labels)
    # ax.set_xlim(1e-6, 1e1)
    ax.invert_xaxis()
    ax.set_clip_on(True)

for slr, our, index, x in zip(
    satire_left_error, our_left_error, range(len(labels)), satire_left
):
    if x == 0:
        axes[0].scatter(our, index, color="#327787", zorder=3, s=30, marker="*")
    elif slr == our:
        axes[0].scatter(
            x=slr,
            y=index,
            color="black",
            zorder=3,
            s=30,
            marker="x",
            linewidth=2,
        )
    else:
        axes[0].scatter(
            x=slr,
            y=index,
            color="black",
            zorder=3,
            s=30,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[0].annotate(
            "",
            xytext=(slr, index),
            xy=(our, index),
            arrowprops=dict(arrowstyle="->", color="#327787", lw=2),
            zorder=2,
        )

for sla, our ,index, x in zip(
    satire_left_relative_error, our_left_relative_error, range(len(labels)), satire_left
):
    if x == 0:
        axes[1].scatter(our, index, color="#327787", zorder=3, s=30, marker="*")
    elif sla == our:
        axes[1].scatter(
            x=sla,
            y=index,
            color="black",
            zorder=3,
            s=30,
            marker="x",
            linewidth=2,
        )
    else:
        axes[1].scatter(
            x=sla,
            y=index,
            color="black",
            zorder=3,
            s=30,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].annotate(
            "",
            xytext=(sla, index),
            xy=(our, index),
            arrowprops=dict(arrowstyle="->", color="#327787", lw=2),
            zorder=2,
        )

for srr, our, index, x in zip(
    satire_right_error, our_right_error, range(len(labels)), satire_right
):
    if x == 0:
        axes[2].scatter(our, index, color="#327787", zorder=3, s=30, marker="*")
    elif srr == our:
        axes[2].scatter(
            x=srr,
            y=index,
            color="black",
            zorder=3,
            s=30,
            marker="x",
            linewidth=2,
        )
    else:
        axes[2].scatter(
            x=srr,
            y=index,
            color="black",
            zorder=3,
            s=30,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[2].annotate(
            "",
            xytext=(srr, index),
            xy=(our, index),
            arrowprops=dict(arrowstyle="->", color="#327787", lw=2),
            zorder=2,
        )

for sra, our, index, x in zip(
    satire_right_relative_error, our_right_relative_error, range(len(labels)), satire_right
):
    if x == 0:
        axes[3].scatter(our, index, color="#327787", zorder=3, s=30, marker="*")
    elif sra == our:
        axes[3].scatter(
            x=sra,
            y=index,
            color="black",
            zorder=3,
            s=30,
            marker="x",
            linewidth=2,
        )
    else:
        axes[3].scatter(
            x=sra,
            y=index,
            color="black",
            zorder=3,
            s=30,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[3].annotate(
            "",
            xytext=(sra, index),
            xy=(our, index),
            arrowprops=dict(arrowstyle="->", color="#327787", lw=2),
            zorder=2,
        )


axes[0].set_xlabel("Maximum Absolute Error (left interval)", fontproperties=prop, fontsize=14)
axes[1].set_xlabel("Maximum Relative Error (left interval)", fontproperties=prop, fontsize=14)
axes[2].set_xlabel("Maximum Absolute Error (right interval)", fontproperties=prop, fontsize=14)
axes[3].set_xlabel("Maximum Relative Error (right interval)", fontproperties=prop, fontsize=14)

plt.tight_layout()
plt.show()
plt.savefig("RQ1.pdf")
