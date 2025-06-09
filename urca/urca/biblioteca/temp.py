#%%
import pandas as pd
from db_api import UrcaDb, BcbDb, Local, AmpereDB


#%%
for t in Local().tables:
    print(t)

#%%
with open("database.txt", "w") as f:
    for t in Local().tables:
        f.write(t+"\n")
        data = Local().get_table(t)
        f.write(data.describe().to_string())
        f.write("\n---------------------------------------\n")

#%%
from pprint import pprint
for db in [UrcaDb(), BcbDb(), Local(), AmpereDB()]:
    pprint(db.tables)