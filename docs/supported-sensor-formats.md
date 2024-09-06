# Supported sensor formats

Each company producing sensors has their own data formats used for data exchange. Sometimes they also include common formats, like CSV, as export method, but still they come uniquely structured. Gensor, as of the first release, has a `read_from_csv()` function used to read timeseries from data exported from sensors which requires to indicate which format is being used. Based on the user choice, an appropriate parser is selected, which is preconfigured for a particular brand of sensors.

As of now, the only pre-configured parser implemented is one for van Essen instruments. The developers hope, that with the contributions from the community, we can include more prewritten formats with tests and examples, so not everybody has to parse the files themselves each time.

## van Essen format

#### What do the van Essen Instruments loggers measure?

The loggers measure the abolute pressure at the membrane of the sensor, therefore the datasets require compensation for barometric pressure.

#### How the data from van Essen instruments is collected?

If the loggers are not equipped with some kind of telemetric system (which is the case most of the times), it is necessary to go to the field once in a while and collect the data. That is done either with a laptop (using DiverOffice or DiverField software) or a small handheld device called DiverMate and an Android device. Each time the data is collected, a timeseries is generated (that includes old records that may have been already collected before and are stored in another CSV file).
Normall, in Project Grow we collect the data every two months with DiverMate and then share the CSV files to Google Drive. Each time we export the data, we end up with a new CSV file that need to be merged to obtain a consistent timeseries.
