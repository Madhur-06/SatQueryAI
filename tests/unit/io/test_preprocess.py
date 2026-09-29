import numpy as np
import pytest

from satquery.io.preprocess import optical, sar, temporal
from satquery.io.raster import inspect_raster, read_raster
from satquery.types import BandRole

ALL = np.ones((64, 64), dtype=bool)


def test_db_roundtrip_and_floor() -> None:
    x = np.array([1e-3, 0.1, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(sar.to_linear(sar.to_db(x)), x, rtol=1e-5)
    assert sar.to_db(np.array([0.0]))[0] == sar.DB_FLOOR


def test_detect_scale() -> None:
    assert sar.detect_scale(np.array([[-10.0, 2.0]]), np.ones((1, 2), bool), "float32") == "db"
    assert sar.detect_scale(np.array([[0.1, 0.2]]), np.ones((1, 2), bool), "float32") == "linear"
    assert sar.detect_scale(np.array([[10, 20]]), np.ones((1, 2), bool), "uint16") == "dn"


@pytest.mark.parametrize("looks", [1, 4, 8])
def test_estimate_looks_on_gamma_field(looks: int) -> None:
    field = np.random.default_rng(0).gamma(looks, 1 / looks, (256, 256))
    valid = np.ones_like(field, bool)
    assert sar.estimate_looks(field, valid) == pytest.approx(looks, rel=0.1)
    field[:, :128] *= 0.03  # two classes: the estimate must stay robust to the edge
    assert sar.estimate_looks(field, valid) == pytest.approx(looks, rel=0.1)


def test_lee_filter_reduces_speckle_and_keeps_mean(sar_linear) -> None:
    data, valid = read_raster(inspect_raster(sar_linear.path))
    vv = data[0]
    land = ~sar_linear.truth["water"]
    filtered = sar.lee_filter(vv, valid, size=7)

    def cv(x):
        return x[land].std() / x[land].mean()

    assert cv(filtered) < 0.6 * cv(vv)
    assert filtered[land].mean() == pytest.approx(vv[land].mean(), rel=0.05)


def test_lee_filter_rejects_even_window() -> None:
    with pytest.raises(ValueError, match="odd"):
        sar.lee_filter(np.ones((8, 8)), np.ones((8, 8), bool), size=4)


def test_sar_composite() -> None:
    vv = np.full((4, 4), -10.0)
    rgb = sar.composite(vv, vv - 6, np.ones((4, 4), bool))
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    grey = sar.composite(vv, None, np.ones((4, 4), bool))
    assert (grey[..., 0] == grey[..., 2]).all()


def test_percentile_stretch_range() -> None:
    band = np.random.default_rng(0).normal(1000, 200, (64, 64))
    out = optical.percentile_stretch(band, ALL)
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert optical.percentile_stretch(np.zeros((4, 4)), np.zeros((4, 4), bool)).max() == 0


def test_true_and_false_colour(optical_4band) -> None:
    data, valid = read_raster(inspect_raster(optical_4band.path))
    bands = {BandRole.BLUE: 0, BandRole.GREEN: 1, BandRole.RED: 2, BandRole.NIR: 3}
    tc = optical.true_colour(data, bands, valid)
    fc = optical.false_colour(data, bands, valid)
    assert tc is not None
    assert fc is not None
    assert tc.shape == fc.shape == (128, 128, 3)
    # Vegetation is bright red in false colour; water is dark.
    land = ~optical_4band.truth["water"]
    assert fc[land, 0].mean() > fc[~land, 0].mean() + 100
    assert optical.true_colour(data, {BandRole.RED: 2}, valid) is None


def test_normalized_difference_handles_invalid() -> None:
    a = np.array([[3.0, 0.0, 1.0]])
    b = np.array([[1.0, 0.0, 1.0]])
    valid = np.array([[True, True, False]])
    nd = optical.normalized_difference(a, b, valid)
    assert nd[0, 0] == pytest.approx(0.5)
    assert np.isnan(nd[0, 1])  # zero denominator
    assert np.isnan(nd[0, 2])  # invalid


def test_histogram_match_removes_gain(temporal_pair) -> None:
    t1, t2 = temporal_pair
    d1, v1 = read_raster(inspect_raster(t1.path))
    d2, v2 = read_raster(inspect_raster(t2.path))
    stable = ~t2.truth["builtup_gain"]
    before = abs(d2[3][stable].mean() - d1[3][stable].mean())
    matched = temporal.histogram_match(d2, d1, v2, v1)
    after = abs(matched[3][stable].mean() - d1[3][stable].mean())
    assert after < 0.5 * before
