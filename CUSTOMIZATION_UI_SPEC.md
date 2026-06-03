# Wiring the "Customize" parameters dialog into the UI

The backend is done (branch `expose-roi-psp-params`): the engine/morph read these keys from the
project config with the AMAP defaults, so behaviour is unchanged until a user edits them:
`min_fp_pixels=25`, `roi_contour_min_area=4000`, `roi_dilation_iters=25`, `roi_erosion_iters=8`,
`roi_min_area=5000`. The dialog (`src/ui/customization_dialog.py`) is ready. Three small edits add
the button and wire it. (Cannot be GUI-verified here — please launch the app once after wiring.)

## 1. `src/ui/ui_mainwindow.py` — add the button (free cell: row 3, col 1, next to "Old ROI")
In `setupUi`, after the `check_old_roi` block (~line 282):
```python
        self.button_customize = QPushButton(self.frame_projects)
        self.button_customize.setObjectName(u"button_customize")
        self.button_customize.setEnabled(False)
        self.grid_configs.addWidget(self.button_customize, 3, 1, 1, 1)
```
In `retranslateUi`, near the other `setText` calls:
```python
        self.button_customize.setText(QCoreApplication.translate("MainWindow", u"Customize…", None))
```
(`QPushButton` is already imported in this file.)

## 2. `src/ui/main_window.py` — reference + connect + open handler
Top of file:
```python
from src.ui.customization_dialog import CustomizationDialog
```
(ensure `QPushButton` is imported from `PySide6.QtWidgets`.)

Where the other widgets are grabbed (next to `self.check_use_gpu = self.findChild(...)`, ~line 217):
```python
        self.button_customize = self.findChild(QPushButton, "button_customize")
        self.button_customize.clicked.connect(self.open_customization_dialog)
```

Add the handler (mirrors `checkbox_use_gpu_change`):
```python
    def open_customization_dialog(self):
        if self.is_disabled or self.is_loading:
            return
        project_name = self.list_projects.currentItem().text()
        path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        configs = self.load_project_configuration(path)
        dlg = CustomizationDialog(configs, configs.get('is_old_roi', False), self)
        if dlg.exec():                      # accepted
            configs.update(dlg.values())
            self.save_project_configuration(path, configs)
```

## 3. Enable/disable logic
In the project-load code (where `self.check_use_gpu.setChecked(...)` / `setEnabled(...)` are set,
~lines 296 & 318), add:
```python
        self.button_customize.setEnabled(not project_configs['is_morphometry_finished'])
```
And in `checkbox_old_roi_change` (after the save, ~line 385) keep the button usable for both
algorithms (the dialog itself greys the ROI-only params when old ROI is selected) — no extra line
needed if you follow the recommended design below.

## Design decision (the part you were unsure about)
- **Recommended (implemented in the dialog):** the **Customize button stays enabled** for any
  active project; inside the dialog the **ROI-algorithm params are greyed out when Old ROI is
  selected** (that algorithm uses its own fixed values), while the **post-processing params
  (`min_fp_pixels`, `roi_contour_min_area`) stay editable** because they apply to *both* ROI
  algorithms. This is the most correct behaviour and needs no special button logic.
- **Your first instinct (simpler):** if you'd rather fully disable customization under Old ROI,
  replace the enable line above with
  `self.button_customize.setEnabled(not project_configs['is_old_roi'] and not project_configs['is_morphometry_finished'])`
  and add the same toggle in `checkbox_old_roi_change`. Defaults are then always used under Old ROI.

## Verify after wiring
1. Launch the app, load a project → "Customize…" button appears and is enabled.
2. Click it → dialog opens with the 5 params at their defaults; edit one, OK → check `conf.json` got
   the new value; reopen → value persists.
3. Toggle "Old ROI" → ROI params are greyed (recommended design); PSP params stay editable.
4. Run the project → confirm a changed param visibly affects the ROI/SD output.
