import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from matplotlib import font_manager as fm

font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
prop = fm.FontProperties(fname=font_path)
rcParams["font.size"] = 10

cases = [
    "#1924",
    "#5990",
    "#3017",
    "#5895",
    "#4551",
    "#1190",
    "#2843",
    "#1960",
    "#2680",
    "#376",
    "#5065",
    "#1840",
    "#1666",
    "#1937",
    "#1671",
    "#1821",
    "#4701",
    "#1808",
    "#1578",
    "#6227",
]

t1 = [
    2.8088924884796143,
    3.2178518772125244,
    2.2077012062072754,
    2.346226692199707,
    2.1676509380340576,
    3.563875675201416,
    2.2054829597473145,
    2.2399049758911133,
    5.332653999328613,
    2.4750232696533203,
    15.647299528121948,
    4.977992057800293,
    8.399636268615723,
    6.1992902755737305,
    95.83924055099487,
    375.47,
    8.162198066711426,
    310.0457820892334,
    2.222644329071045,
    6.91,
]
t2 = [
    11.410336971282959,
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
]


t1 = np.array(t1)
t2 = np.array(t2)

extra = t2 - t1
x = np.arange(1, len(t1) + 1)

plt.figure(figsize=(8, 4))
plt.bar(x, t1, label="Standard", color="#1f4e79", edgecolor='black', linewidth=1)
plt.bar(x, extra, bottom=t1, label="Overhead (Our approach)", color="#76b0d6", edgecolor='black', linewidth=1)
plt.yscale("symlog", linthresh=1, linscale=1)

for i in range(len(x)):
    rel_diff = (t2[i] - t1[i]) / t1[i]
    if rel_diff < 0.1:
        plt.text(
            x[i],
            t2[i] * 1.03,
            f"{t2[i]:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontproperties=prop,
        )
    else:
        if t1[i] > 1000 or t2[i] > 1000:
            plt.text(
                x[i],
                t1[i] * 1.03,
                f"{t1[i]:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontproperties=prop,
            )
            plt.text(
                x[i],
                t2[i] * 1.03,
                f"{t2[i]:.4g}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontproperties=prop,
            )
        else:
            plt.text(
                x[i],
                t1[i] * 1.03,
                f"{t1[i]:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontproperties=prop,
            )
            plt.text(
                x[i],
                t2[i] * 1.03,
                f"{t2[i]:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontproperties=prop,
            )

plt.xlabel("Test Case", fontproperties=prop)
plt.ylabel("Execution Time (seconds)", fontproperties=prop)
# plt.title('Runtime Overhead of Our approach', fontproperties=prop)
plt.legend(fontsize=15)
x_labels = cases
plt.xticks(x, x_labels, rotation=30)
# plt.xticks(x)
plt.yticks([0, 1, 10, 100, 1000, 3600], ["0", "1", "10", "100", "1000", "3600"])
plt.tight_layout()
plt.savefig("RQ2.pdf")

overhead = t2 / t1
print(f"Max Overhead: {np.max(overhead):.4f}x")
print(f"Min Overhead: {np.min(overhead):.4f}x")

print(f"Average Overhead: {np.mean(overhead):.4f}x")

