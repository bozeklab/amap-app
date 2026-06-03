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

# key -> (label, default, min, max, step, is_psp, description)
#   is_psp=True  -> post-processing, applies to BOTH ROI algorithms (always editable)
#   is_psp=False -> only used by the new ROI algorithm (disabled when old ROI is selected)
# Pixel thresholds are on images resampled to the fixed 0.0227 µm/px grid (1 px² ≈ 5.15e-4 µm²),
# so a pixel value corresponds to a fixed physical size (given below).
PARAMS = [
    ("min_fp_pixels",        "Min foot-process size (px)",     25,   1,   100_000,    1,  True,
     "Smallest foot-process kept (pixels). Connected components below this size in the instance "
     "segmentation are discarded as noise. At 0.0227 µm/px, 25 px ≈ 0.013 µm²."),
    ("roi_contour_min_area", "ROI contour min area (px)",      4000, 0,   10_000_000, 50, True,
     "Smallest ROI outline kept when drawing the region of interest (pixels). Smaller contours are "
     "ignored. ≈ 2.06 µm²."),
    ("roi_dilation_iters",   "ROI dilation iterations",        25,   1,   500,        1,  False,
     "Morphological dilation iterations that merge slit-diaphragm structures into one continuous "
     "ROI. Higher = larger, more merged region. (New ROI algorithm only.)"),
    ("roi_erosion_iters",    "ROI erosion iterations",         8,    0,   500,        1,  False,
     "Morphological erosion iterations applied after dilation to tighten/smooth the ROI boundary. "
     "Higher = tighter ROI. (New ROI algorithm only.)"),
    ("roi_min_area",         "ROI min component area (px)",    5000, 0,   10_000_000, 50, False,
     "Smallest connected ROI region kept (pixels); smaller regions are removed as noise. "
     "≈ 2.58 µm². (New ROI algorithm only.)"),
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
        for key, label, default, lo, hi, step, is_psp, desc in PARAMS:
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(int(configs.get(key, default)))
            # Wrap as fixed-width rich text so the tooltip word-wraps instead of rendering
            # as one long unreadable line.
            tip = f"<div style='width: 260px; white-space: normal;'>{desc}</div>"
            spin.setToolTip(tip)             # hover description over the value
            if is_old_roi and not is_psp:
                spin.setEnabled(False)
            row_label = QLabel(label)
            row_label.setToolTip(tip)        # ...and over the label
            self._spins[key] = spin
            form.addRow(row_label, spin)
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
