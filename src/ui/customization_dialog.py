"""Customization dialog: exposes the post-processing (PSP) and ROI hyperparameters so users can
tweak them per project. Defaults match the engine's config defaults (the empirically chosen AMAP
values), so a fresh project behaves exactly as before. The deep-learning model and its
resolution/patch settings are intentionally NOT exposed (model adaptation is done by fine-tuning in
the AMAP repository and loading the resulting checkpoint via the checkpoint selector).

ROI-algorithm parameters are disabled when the *old* ROI algorithm is selected, because that
algorithm uses its own fixed internal values; the post-processing parameters remain editable since
they apply to both ROI algorithms.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QSpinBox, QDialogButtonBox, QLabel, QPushButton,
)

# key -> (label, default, min, max, step, is_psp)
#   is_psp=True  -> post-processing, applies to BOTH ROI algorithms (always editable)
#   is_psp=False -> only used by the new ROI algorithm (disabled when old ROI is selected)
PARAMS = [
    ("min_fp_pixels",        "Min foot-process size (px)",     25,   1,   100_000,    1,  True),
    ("roi_contour_min_area", "ROI contour min area (px)",      4000, 0,   10_000_000, 50, True),
    ("roi_dilation_iters",   "ROI dilation iterations",        25,   1,   500,        1,  False),
    ("roi_erosion_iters",    "ROI erosion iterations",         8,    0,   500,        1,  False),
    ("roi_min_area",         "ROI min component area (px)",    5000, 0,   10_000_000, 50, False),
]
DEFAULTS = {key: default for key, _, default, *_ in PARAMS}


class CustomizationDialog(QDialog):
    def __init__(self, configs, is_old_roi, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize processing parameters")
        self._spins = {}

        layout = QVBoxLayout(self)
        if is_old_roi:
            note = QLabel("Old ROI algorithm selected — ROI parameters are fixed by that algorithm; "
                          "only the post-processing parameters below are editable.")
            note.setWordWrap(True)
            layout.addWidget(note)

        form = QFormLayout()
        for key, label, default, lo, hi, step, is_psp in PARAMS:
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(int(configs.get(key, default)))
            if is_old_roi and not is_psp:
                spin.setEnabled(False)
            self._spins[key] = spin
            form.addRow(label, spin)
        layout.addLayout(form)

        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset_defaults)
        layout.addWidget(reset)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _reset_defaults(self):
        for key, spin in self._spins.items():
            if spin.isEnabled():
                spin.setValue(DEFAULTS[key])

    def values(self):
        """Return {config_key: int value} for all parameters (call after exec() returns accepted)."""
        return {key: spin.value() for key, spin in self._spins.items()}
