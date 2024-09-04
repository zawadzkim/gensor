# gensor

[![Release](https://img.shields.io/github/v/release/zawadzkim/gensor)](https://img.shields.io/github/v/release/zawadzkim/gensor)
[![Build status](https://img.shields.io/github/actions/workflow/status/zawadzkim/gensor/main.yml?branch=main)](https://github.com/zawadzkim/gensor/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/zawadzkim/gensor)](https://img.shields.io/github/commit-activity/m/zawadzkim/gensor)
[![License](https://img.shields.io/github/license/zawadzkim/gensor)](https://img.shields.io/github/license/zawadzkim/gensor)

Library for handling groundwater sensor data.

# Van Essen Instruments

Loggers produced by van Essen instruments are a popular choise in Reasearch projects due to their long battery life, small size and relatively good price. Recently, the data can be easily collected in the field with a peripheral device called DiverMate. From that device, the data can be shared to, e.g., Google Drive or OneDrive cloud storate as a CSV file.

Firstly the data is read from a chosen data source (a local file or a cloud source) and parsed. The CSV files have a uniform structure; the first 50 rows are metadata about the device and its settings and the remaining rows are data rows. Therefore, it is easy to set up a function that parses the CSV files into the metadata and data.

Secondly, the user chooses whether to send data to their own cloud storage (e.g., relational database, web application) or store locally as a sqlite database. Preferably, the data is always stored as raw measurements

Lastly, the package contains functions to process the data into groundwater levels and perform simple plotting.
