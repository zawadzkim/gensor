"""
!!! warning

    Whenever Timeseries objects are created via read_from_csv and use a parser (e.g.,
    'vanessen'), the timestamps are localized and converted to UTC. Therefore, if the
    user creates his own timeseries outside the read_from_csv, they should ensure that
    the timestamps are in UTC format.
"""

VARIABLE_TYPES_AND_UNITS = {
    "temperature": ["degc"],
    "pressure": ["cmh2o", "mmh2o"],
    "conductivity": ["ms/cm"],
    "flux": ["m/s"],
    "head": ["m asl"],
    "depth": ["m"],
}
