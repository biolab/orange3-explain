# a no-op workaround so that table unlocking does not crash with older Orange
# remove when the minimum supported version is 3.31
from contextlib import nullcontext

from Orange.data import Table

if not hasattr(Table, "unlocked"):
    Table.unlocked = nullcontext

# temporary disable due to https://github.com/biolab/orange3/issues/5746
Table.LOCKING = False
