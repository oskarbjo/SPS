

import pytimber as pt
import matplotlib.pyplot as plt
import numpy as np

  


name='MKI.867.IPOC.CPU.%'
db=pt.LoggingDB()

START_TIMESTAMP = '2020-10-09 19:37:00'
STOP_TIMESTAMP = '2020-10-09 19:38:00'

data=db.get(name,START_TIMESTAMP,STOP_TIMESTAMP)

data[list(data.keys())[0]]

print('')