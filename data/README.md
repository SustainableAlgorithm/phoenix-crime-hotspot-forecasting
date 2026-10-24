# Data

Place the source incident CSV on your local machine and pass its path to the pipeline. Do **not** commit raw incident-level records to this repository. Confirm the dataset's license, terms of use, and redistribution permissions before sharing any data-derived material.

The pipeline expects columns named `OCCURRED ON` and `GRID`. It aggregates usable records between January 2020 and December 2024 into month-by-grid counts. Incident identifiers, street addresses, and ZIP codes are not used in the aggregate panel.

The generated aggregate data can still carry sensitive implications when mapped or interpreted. Review privacy, disclosure risk, and intended use before publication.
