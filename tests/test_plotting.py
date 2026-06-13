import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def test_plot(synthetic_submerged_timeseries):
    """Test the plot method of the Timeseries class."""
    ts = synthetic_submerged_timeseries

    fig, ax = ts.plot()

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_title() == "Pressure at Station A (Sensor 1)"
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "pressure (cmh2o)"

    line = ax.lines[0]
    assert (
        list(line.get_xdata())
        == pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC").tolist()
    )
    assert list(line.get_ydata()) == [1313.00, 1312.00, 1310.00]

    plt.close(fig)


def test_dataset_plot(synthetic_dataset):
    """Test the plot method of the Dataset class."""
    dataset = synthetic_dataset

    fig, axes = dataset.plot()

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert all(isinstance(ax, plt.Axes) for ax in axes)

    assert len(axes) == len({ts.variable for ts in dataset.timeseries})

    for ax in axes:
        assert "Timeseries for" in ax.get_title()
        assert ax.get_xlabel() == "Time"

    plt.close(fig)


def test_dataset_plot_facet_location(synthetic_dataset):
    """facet='location' returns a figure per variable, one panel per location."""
    result = synthetic_dataset.plot(facet="location", ncols=5)

    # synthetic_dataset has a single variable (pressure) -> one figure
    assert isinstance(result, dict)
    assert set(result) == {ts.variable for ts in synthetic_dataset.timeseries}

    fig, axes = result["pressure"]
    assert isinstance(fig, plt.Figure)
    locations = synthetic_dataset.get_locations()
    # every location gets a (titled) panel; the rest of the grid is hidden
    visible = [ax for ax in axes if ax.get_visible()]
    assert len(visible) == len(locations)
    assert {ax.get_title() for ax in visible} == set(locations)

    for fig, _ in result.values():
        plt.close(fig)


def test_dataset_plot_facet_location_legend_by_sensor(synthetic_submerged_timeseries):
    """A panel with >1 sensor shows a legend labelled by serial; single-series panels don't."""
    from gensor.core.dataset import Dataset

    ts_a1 = synthetic_submerged_timeseries  # location 'Station A', sensor 'Sensor 1'
    ts_a2 = ts_a1.model_copy(update={"sensor": "Sensor X"})  # same location, 2nd sensor
    ts_b = ts_a1.model_copy(update={"location": "Station B", "sensor": "Sensor 2"})
    ds = Dataset(timeseries=[ts_a1, ts_a2, ts_b])

    fig, axes = ds.plot(facet="location")["pressure"]
    by_loc = {ax.get_title(): ax for ax in axes if ax.get_visible()}

    # Station A has two sensors -> legend by serial; Station B has one -> no legend
    leg_a = by_loc["Station A"].get_legend()
    assert leg_a is not None
    assert {t.get_text() for t in leg_a.get_texts()} == {"Sensor 1", "Sensor X"}
    assert by_loc["Station B"].get_legend() is None

    plt.close(fig)


def test_dataset_plot_facet_location_sharex(synthetic_dataset):
    """sharex=True aligns every location panel to the same (full) x-range."""
    fig, axes = synthetic_dataset.plot(facet="location", sharex=True)["pressure"]
    drawn = [ax for ax in axes if ax.get_visible() and ax.lines]
    assert len(drawn) >= 2
    xlims = {ax.get_xlim() for ax in drawn}
    assert len(xlims) == 1  # all panels share identical x-limits
    plt.close(fig)
