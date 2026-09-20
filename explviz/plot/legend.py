"""Building one shared legend for a figure."""

from __future__ import annotations


def collect_handles(axes_list):
    """Collect (label -> handle) across axes, keeping first occurrence."""
    found = {}
    for ax in axes_list:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            found.setdefault(l, h)
    return found


def build_legend(fig, axes_list, order=None, fontsize=18, width_frac=0.25):
    """Place a single deduplicated legend in its own axes on the right.

    ``order`` optionally lists labels in the order they should appear; labels
    in ``order`` that no artist produced are **skipped**.  The original code
    appended ``handles_labels.get(label)`` unconditionally, so a missing label
    inserted a ``None`` handle and broke the legend.
    """
    found = collect_handles(axes_list)
    if not found:
        return None

    if order:
        labels = [l for l in order if l in found]
        labels += [l for l in found if l not in order]
    else:
        labels = list(found)

    handles = [found[l] for l in labels]

    fig.subplots_adjust(right=1.0 - width_frac - 0.05)
    legend_ax = fig.add_axes([1.0 - width_frac - 0.02, 0.05, width_frac, 0.9])
    legend_ax.axis("off")
    return legend_ax.legend(handles, labels, fontsize=fontsize,
                            loc="center", frameon=True)
