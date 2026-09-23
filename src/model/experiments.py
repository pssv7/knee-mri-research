"""Pre-registered, single-change experiments. No preset implies a score gain."""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Experiment:
    name: str = "baseline"
    backbone: str = "convnext_tiny.fb_in22k_ft_in1k"
    slices: int = 16
    max_series: int = 5
    series_selection: str = "uniform"
    resample: bool = False
    classwise_pool: bool = False
    slice_transformer: bool = False
    spatial_encoder: bool = False
    plane_adapters: bool = False

    def to_dict(self):
        result = asdict(self)
        # Preserve existing experiment contracts and checkpoint sidecars.
        for name in ("slice_transformer", "spatial_encoder", "plane_adapters"):
            if not getattr(self, name):
                result.pop(name)
        return result


BASELINE = Experiment()
PRESETS = {
    "baseline": BASELINE,
    "resample": replace(BASELINE, name="resample", resample=True),
    "protocol": replace(BASELINE, name="protocol", series_selection="protocol"),
    "classwise": replace(BASELINE, name="classwise", classwise_pool=True),
    "convnextv2": replace(BASELINE, name="convnextv2",
                          backbone="convnextv2_tiny.fcmae_ft_in22k_in1k"),
    "slice_transformer": replace(BASELINE, name="slice_transformer",
                                 series_selection="protocol", slice_transformer=True),
    # The 24-slice control the spatial candidates are screened against. Every
    # difference from the incumbent -- protocol selection, 24 slices -- is in
    # this preset too, so the only thing the comparison varies is the 3D path.
    "spatial_control": replace(BASELINE, name="spatial_control",
                               series_selection="protocol", slices=24),
    "spatial": replace(BASELINE, name="spatial", series_selection="protocol",
                       slices=24, spatial_encoder=True),
    # Screened only after the spatial blocks earn it: two architecture changes
    # in one arm cannot be attributed to either.
    "spatial_planes": replace(BASELINE, name="spatial_planes",
                              series_selection="protocol", slices=24,
                              spatial_encoder=True, plane_adapters=True),
}


def get_experiment(name):
    if name not in PRESETS:
        raise ValueError(f"unknown experiment {name!r}; choose {sorted(PRESETS)}")
    return PRESETS[name]
