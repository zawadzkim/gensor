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
