# Info
This module is used to generate some random data on the corridor of interest. We used SUMO, and more specifically, tools/osmWebWizard.py, tools/randomTrips.py (with --validate option in order to get rid of Error: No route for vehicle 'vehicle_id'), in order to generate the network and some random trips.

command to work with osmWebWizard.py
```
python $SUMO_HOME/tools/osmWebWizard.py
```
Python should be installed for this command. SUMO repository should be cloned and its path should be exported into system's environmental variables using:
```
export SUMO_HOME=path/to/sumo/folder
```
For generating the trips use:
```
pyhton $SUMO_HOME/tools/randomTrips.py -e 36000 -n "path/to/net.xml" -o "path/to/output.trips.xml" --validate
```