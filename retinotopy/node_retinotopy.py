import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

node_info = json.load(open("node_info_nodist.json"))
xcutoff = -50
ecc_band_width = 20
occ_nodes = [n for n in node_info if n["location"][0]<xcutoff]
locs = [n["location"] for n in occ_nodes]
locs_array = np.array(locs)

ecc_bins = [(xcutoff - ecc_band_width*i) for i in range(4)][::-1]
ecc_node_bins = np.digitize([l[0] for l in locs], ecc_bins)
for ni, n in enumerate(occ_nodes):
	n["ecc_band"] = ecc_node_bins[ni]

ymin = locs_array[:, 1].min()
ymax = locs_array[:, 1].max()
ang_band_width = (ymax-ymin)/12

ang_assignments = [3, 2, 3, 5, 4, 5, 7, 6, 7, 1, 0, 1]
ang_bins = [(ymin + ang_band_width*i) for i in range(12)]
ang_node_bins = np.digitize([l[1] for l in locs], ang_bins)
for ni, n in enumerate(occ_nodes):
	n["ang_band"] = ang_assignments[ang_node_bins[ni]-1]

colors_hex = ["#FF001A", "#FFA500", "#9AFF00", "#00FF25", "#00FFE5", "#005AFF", "#6500FF", "#FF00D9"]
colors_rgb255 = [(255, 0, 26), (255, 165, 0), (154, 255, 0), (0, 255, 37), (0, 255, 229), (0, 90, 255), (101, 0, 255), (255, 0, 217)]
colors_rgb = [tuple(np.array(c)/255.) for c in colors_rgb255]

for ni, n in enumerate(occ_nodes):
	n["color"] = colors_rgb[n["ang_band"]]+(0.25*(n["ecc_band"]+1),)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([p["location"][0] for p in occ_nodes], [p["location"][1] for p in occ_nodes], [p["location"][2] for p in occ_nodes], c = [n["color"] for n in occ_nodes])


f1 = plt.figure()
ax1 = f1.add_subplot(111, polar=True)

for i in xrange(8*4):
    color = colors_rgb[i % 8] + (0.25*(i/8 + 1), )
    ax1.bar(i * 2 * np.pi / 8, 1, width=2 * np.pi / 8, bottom=i / 8,
           color=color, edgecolor = color)
ax1.set_yticks([])
plt.show()

