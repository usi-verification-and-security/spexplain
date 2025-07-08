from matplotlib import pyplot as plt
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import os

import numpy as np
from matplotlib import pyplot as plt

r1 = [1.77,
60.4,
780,
2300]

r2 = [255.355,
376.139,
366.744,
479.93]


models = ['MNIST 1x50',
'MNIST 2x50',
'MNIST 3x50',
'MNIST 4x50',]
plt.rcParams['text.usetex'] = True

plt.figure(figsize=(7, 4))
plt.plot(models, r1, label=r'\textsc{S\textsubscript{p}EX\textsubscript{pl}AI\textsubscript{n}} (Generalize)', marker='o',linewidth=2.5, color='blue')
plt.plot(models, r2, label=r'\textsc{VeriX} (Abductive) ', marker='o',linewidth=2.5, color='red')

# Add labels, title, and legend
plt.xlabel('Model', fontsize=16)
plt.ylabel('Runtime (s)', fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=14)  # Adjust the font size for major ticks
# plt.tick_params(axis='both', which='minor', labelsize=12)  # Adjust the font size for minor ticks
# plt.title('Runtime Comparison')
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks(np.arange(0, 2000, 500))
plt.grid(alpha=0.3)
plt.legend(fontsize=14)
plt.ylim(0, 1800)
# Show the plot
plt.tight_layout()

plt.savefig(f"plots/0--plot_layers.pdf", format='pdf')
plt.show()
# plt.plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i])

# plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i])



r1 = [24.46,
69.94,
88.88,
118.26,
172.63]

r2 = [5.371,93.469,
589.178,
1589.322,
6357.288]

xs = [100,
400,
900,
1600,
2500]

plt.rcParams['text.usetex'] = True

plt.figure(figsize=(7, 4))
plt.plot(xs, r1, label=r'\textsc{S\textsubscript{p}EX\textsubscript{pl}AI\textsubscript{n}} (Generalize)', marker='o', linewidth=2.5, color='blue')
plt.plot(xs, r2, label=r'\textsc{VeriX} (Abductive) ', marker='o',  linewidth=2.5, color='red')

# Add labels, title, and legend
plt.xlabel('Input Size', fontsize=16)
plt.ylabel('Runtime (s)', fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=14)  # Adjust the font size for major ticks
# plt.tick_params(axis='both', which='minor', labelsize=12)  # Adjust the font size for minor ticks
# plt.title('Runtime Comparison')
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks(np.arange(0, 2000, 500))
plt.grid(alpha=0.3)
plt.legend(fontsize=14)
plt.ylim(0, 1800)
# Show the plot
plt.tight_layout()

plt.savefig(f"plots/0--plot_input.pdf", format='pdf')
plt.savefig(f"plots/0--plot_input.png", format='png')
plt.show()
# plt.plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i])

# plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i])