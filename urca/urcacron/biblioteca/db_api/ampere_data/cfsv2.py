# +
# This file download the CFSV2 data from Ampere

import os
import shutil
import zipfile
import pandas as pd
# -

from .utils import *


def selectFile2Download(date, base_dir, model, flux, flux_files):
    date_str = dateTimestamp2Str(date)
    yesterday_str = dateTimestamp2Str(date-pd.Timedelta(days=1))
    consolidation = [ # from more consolidated to less
        ("ACOMPH"+date_str,      date_str+"-PSAT"),
        ("ACOMPH"+date_str,      date_str),
        ("ACOMPH"+yesterday_str, date_str)
    ]
    for acomph, date_forecast in consolidation:
        if acomph not in flux_files:
            continue
        if date_forecast not in flux_files[acomph]:
            continue
            
        if model+".zip" in flux_files[acomph][date_forecast]:
            return acomph, date_forecast
    else:
        raise ValueError(f"{date} was not found")

def download_CFSV2_INFORMES(date: pd.Timestamp, flux, base_dir="/share_lab/URCA/Dados/API_Ampere/ENA_PREV_CFSV2-INFORMES", flux_files=None):
    if flux_files is None:
        flux_files = flux.get_list()
    date_str = dateTimestamp2Str(date)
    model = 'CFSV2-INFORMES'
    target_file = 'CFSV2-INFORMES/ena_diaria_final_CFSV2-INFORMES.csv'
    
    acomph, date_forecast = selectFile2Download(date, base_dir, model, flux, flux_files)
    
    zip_file_name = os.path.join(base_dir, "_".join([acomph, date_forecast, '.temp.zip']))
    destination_file = os.path.join(base_dir, "_".join([
            date_str, acomph, date_forecast, os.path.split(target_file)[-1]]))
    
    if not os.path.exists(destination_file):
        a = flux.download(acomph, date_forecast, model, zip_file_name)
        if not os.path.exists(zip_file_name):
            raise ValueError(f"{zip_file_name} download failed")

        # extract the file of interest and delete the zip file
        with zipfile.ZipFile(zip_file_name) as z:
            with z.open(target_file) as zf, open(destination_file, 'wb') as f:
                shutil.copyfileobj(zf, f)
        os.remove(zip_file_name)
    #else:
    #    print("já baixado")
    
    data = pd.read_csv(destination_file, sep=';', skipinitialspace=True)
    data.DATA = dateStr2Timestamp(data.DATA)
    return data

def update_CFSV2_INFORMES(flux, base_dir="/share_lab/URCA/Dados/API_Ampere/ENA_PREV_CFSV2-INFORMES"):
    flux_files = flux.get_list()
    for date in flux.get_list():
        #print(date)
        try:
            download_CFSV2_INFORMES(dateStr2Timestamp(date[-8:]), flux, base_dir, flux_files=flux_files)
        except ValueError as e:
            print(e)

if __name__ == '__main__':
    from ee_ampere_consultoria.produtos.flux import FluxAutomatico
    import os
    from dotenv import load_dotenv

    base_dir = os.path.split("/share_lab/URCA")
    load_dotenv(os.path.join(*base_dir, 'config', '.env'))
    
    USERNAME = os.getenv("AMPERE_USER")
    MD5_PASSWORD_HASH = os.getenv("AMPERE_PASSWORD")
    USER_ACESS_TOKEN = os.getenv("AMPERE_TOKEN")
    
    flux = FluxAutomatico(USERNAME, MD5_PASSWORD_HASH, USER_ACESS_TOKEN)
    update_CFSV2_INFORMES(flux)
    
    from ampere_ena_prevista import get_avaliable_data
    print(get_avaliable_data("/share_lab/URCA/Dados/API_Ampere/ENA_PREV_CFSV2-INFORMES"))