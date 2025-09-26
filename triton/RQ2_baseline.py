import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm

# 设置 IEEE 风格字体
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
prop = fm.FontProperties(fname=font_path)
rcParams["font.size"] = 10

our = [
    11.818557739257812,
    2.238508701324463,
    5.040207862854004,
    4.734781980514526,
    32.31048798561096,
    4.707355499267578,
    2.2574766635894775,
    11.389878988265991,
    5.291883707046509,
    44.36,
    8.843079328536987,
    8.464456558227539,
    10.175003290176392,
    831.1217892169952,
    18.88 * 60,
    8.173716068267822,
    312.22004556655884,
    2.2405998706817627,
    21,
    11.410336971282959,
]

cases = {
    "#5990": 12.97,
    "#3017": 106.27,
    "#5895": 0.6,
    "#4551": 250.26,
    "#1190": 0,
    "#2843": 23.92,
    "#1960": 0.529,
    "#2680": 171.94,
    "#376": 17.68,
    "#5065": 0,
    "#1840": 630.4319379329681,
    "#1666": 9.80,
    "#1937": 682.1305477619171,
    "#1671": 0,
    "#1821": 0,
    "#4701": 2.1113882064819336,
    "#1808": 0,
    "#1578": 3.37,
    "#6227": 0,
    "#1924": 23.41 * 60,
}

baseline = np.array(list(cases.values()))
x_labels = list(cases.keys())
bar_width = 1
x_index = np.arange(len(x_labels)) * 3
plt.figure(figsize=(8, 4))
plt.bar(
    x_index, our, label="Our Approach", color="#1f4e79", edgecolor="black", linewidth=1
)

# Bar chart for 'cases' data
plt.bar(
    x_index + bar_width,
    baseline,
    label="SATIRE",
    color="#76b0d6",
    edgecolor="black",
    linewidth=1,
    alpha=0.7,
)

for i, value in enumerate(baseline):
    if value == 0:
        plt.bar(
            x_index[i] + bar_width,
            3600,
            color="#76b0d6",
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
        )

plt.yscale("symlog", linthresh=1, linscale=1)
plt.ylim(0, 3600)

plt.xlabel("Test Case", fontproperties=prop)
plt.ylabel("Execution Time (seconds)", fontproperties=prop)
plt.yticks([0, 1, 10, 100, 1000, 3600], ["0", "1", "10", "100", "1000", "3600"])
plt.legend(fontsize=15)
x_labels = list(cases.keys())
plt.xticks(x_index + 0.5, x_labels, rotation=30)
# ax.spines['bottom'].set_position('zero')
# plt.gca().spines["bottom"].set_position("zero")
plt.tight_layout()
plt.savefig("RQ2_baseline.pdf")

# average speedup
# filter out the 0 values
new_baseline = []
new_our = []
for i in range(len(baseline)):
    if baseline[i] != 0 and our[i] != 0:
        new_baseline.append(baseline[i])
        new_our.append(our[i])
baseline = np.array(baseline)
our = np.array(our)
speedup = np.mean(baseline / our)
print("Our speedup: ", speedup)
