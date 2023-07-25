import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
plt.subplots_adjust(wspace=0.25, hspace=0.55)

for case_idx in range(5):
    random_scheduler_results = np.zeros(shape=(5000))
    hybrid_scheduler_results = np.zeros(shape=(5000))
    our_scheduler_results = np.zeros(shape=(5000))
    for repetition in range(3):
        with open('data/random_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'rb') as f:
            random_scheduler_results += np.load(f) / 3

        with open('data/hybrid_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'rb') as f:
            hybrid_scheduler_results += np.load(f) / 3

        with open('data/our_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'rb') as f:
            our_scheduler_results += np.load(f) / 3

    assert(len(random_scheduler_results) == len(hybrid_scheduler_results))
    assert(len(random_scheduler_results) == len(our_scheduler_results))

    data=[]
    for i in range(5000):
        data.append([i, random_scheduler_results[i], 'Random'])
        data.append([i, hybrid_scheduler_results[i], 'Hybrid'])
        data.append([i, our_scheduler_results[i], 'Ours'])

    df=pd.DataFrame(data, columns=['trial', 'cost', 'scheduler'])
    ax=sns.lineplot(ax=axes[case_idx // 3, case_idx % 3], hue_order=['Ours', 'Hybrid', 'Random'], linewidth=2,
                      palette=['tab:green', 'tab:orange', 'tab:blue'],
                      data=df, x="trial", y="cost", hue='scheduler')
    ax.lines[2].set_linestyle("--")

    ax.set_xlabel(None)
    ax.set_ylabel(None)
    if case_idx == 0:
        ax.set(ylim=(0, 20))
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_yticklabels([0, 5, 10, 15, 20], fontsize=15)
    elif case_idx == 1:
        ax.set(ylim=(0, 40))
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_yticklabels([0, 10, 20, 30, 40], fontsize=15)
    elif case_idx == 2:
        ax.set(ylim=(0, 40))
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_yticklabels([00, 10, 20, 30, 40], fontsize=15)
    elif case_idx == 3:
        ax.set(ylim=(0, 60))
        ax.set_yticks([0, 20, 40, 60])
        ax.set_yticklabels([0, 20, 40, 60], fontsize=15)
    elif case_idx == 4:
        ax.set(ylim=(0, 200))
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_yticklabels([0, 50, 100, 150, 200], fontsize=15)

    ax.set_xticks([0, 2500, 5000])
    ax.set_xticklabels([0, 2500, 5000], fontsize=15)

    if case_idx == 4:
        ax.get_legend().set_title(None)
        handles, labels=ax.get_legend_handles_labels()
        handles[2].set_linestyle('--')
        handles=[handles[2], handles[1], handles[0]]
        labels=[labels[2], labels[1], labels[0]]
        ax.legend(handles, labels, ncol=1, handletextpad=0.3,
                  loc='upper left', bbox_to_anchor=(1.2, 1), fontsize=15)
    else:
        ax.get_legend().remove()
axes[1, 2].remove()
plt.savefig("convergence.pdf", dpi=1000)
