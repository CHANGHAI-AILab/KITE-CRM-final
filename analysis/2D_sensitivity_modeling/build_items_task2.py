"""Thin adapter exposing the Task 2 split/labels to the 2D sensitivity scripts.

The 2D arm reuses EXACTLY the official Task 2 Training/Validation split and case labels
used by the released 3D classifier, so that the 3D-vs-2D comparison is on identical cases.

Provide two symbols:

    build_items(task_id) -> (train_items, val_items)
        each a list of (path, label), where `path`'s basename starts with the case id
        (so `norm(basename.split('_')[0])` yields the normalized case id) and `label`
        is the integer gold-standard class.

    norm(id) -> normalized case id (string)

Implementation options (pick one for your environment):

  (A) If you run inside the released classifier package, import its loader directly:
          from classification_tasks2_4.train_LE import build_items, norm
      and delete the fallback below.

  (B) Otherwise, build the two lists from your label table (see sensitivity_paths.LABELS_XLSX):
      read the id / dataset_name / Task-2 gold columns, keep rows with dataset_name in
      {Training, Validation} and a 0/1 label, and point each item at that case's 3D
      decoder-feature .npz (or any per-case file whose name starts with the id).

The fallback raises until wired, to avoid silently using a wrong split.
"""


def norm(b):
    b = str(b).split('_')[0].split('.')[0].lstrip('0')
    return b or '0'


def build_items(task_id):
    raise NotImplementedError(
        "Wire build_items() to the official Task 2 split — see this module's docstring "
        "(option A: import from classification_tasks2_4.train_LE; option B: build from "
        "sensitivity_paths.LABELS_XLSX)."
    )
