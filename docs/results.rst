=======
Results
=======

.. plot::
   :context: close-figs
   :include-source:

    import os.path
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns

    from zalando_classification.utils import make_plot_data

.. plot::
   :context: close-figs
   :include-source:

    summary_dir = "../logs"

    names = ["default", "adam", "adam-beta-0.5"]
    pretty_names = ["RMSProp", r"Adam ($\beta_1=0.9$)", r"Adam ($\beta_1=0.5$)"]
    pretty_name_mapping = dict(zip(names, pretty_names))

    n_splits = 3
    splits = range(n_splits)

    data = make_plot_data(names, splits, summary_dir, pretty_name_mapping)

    g = sns.relplot(x="epoch", y="accuracy", hue="name",
                    row="partition", ci="sd",
                    height=5, aspect=golden_ratio,
                    data=new_data, kind="line")
    g.set(yscale="log")
