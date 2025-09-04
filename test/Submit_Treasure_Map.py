# %%
import argparse
import requests
import urllib.parse
import os, sys, json
from astropy.table import Table

# %%
#set your api_token here:
def register_pts(gwid, infile, verbose=False):
    """

    7DT=94, KMTNet=95

    ra,dec,date-obs,filter,ul5_1,delt
    114.958,-32.264,2024-04-23T03:00:50.552,r,19.335,0.226

    """
    BASE = "https://treasuremap.space/api/v1"
    TARGET = "pointings"
    API_TOKEN = "WbNOm1dQr5cQvYRrpOA6EzULE2PdFjsmv59vJw" #-your-verified-account-api-token-"

    # Read the FIT file into a Table
    table = Table.read(infile)
    graceid = gwid

    # Loop over the rows in the Table
    for row in table:
        # Extract the data from the row
        ra = row['ra']
        dec = row['dec']
        time = row['date-obs'] # str(row['date-obs']) + 'T' + str(row['date-time'])
        depth = row['ul5_1']
        filte = row['filter']
        #t_m_t0 = row['t_m_to']
        #pos_angle = row['pos_angle']


        # Create the JSON data
        json_data = {
            "api_token":API_TOKEN,
            "graceid":graceid,
            "pointings":[
            {
                "position":"POINT({} {})".format(str(ra),str(dec)),
                "band":str(filte),
                "instrumentid":"94",
                "depth":str(depth),
                "depth_unit":"ab_mag",
                "time":str(time),
                "pos_angle":"0.0",
                "status":"completed",
            },
            ]
        }
        print(json_data)

        # Send a POST request to the API endpoint
        url = "{}/{}".format(BASE, TARGET)
        r = requests.post(url=url, json=json_data)
        print(r.text)
        #sys.exit() 

# %%
must_have_keys = ['ra', 'dec', 'date-obs', 'ul5_1', 'filter']

# %%
# gwid = input(f"GW ID (S250206dm):")
# infile = input(f'Path to Table:')

# %%
# intbl = Table.read(infile)
# intbl

# %%
if __name__ == "__main__":
# if (gwid != '') & (infile != ''):
    parser = argparse.ArgumentParser(description='Submitting completed pointings')
    parser.add_argument('gwid', help='GW superevent ID from LVK alert')
    parser.add_argument('infile', help='Input file path')
    sys.exit(register_pts(**vars(parser.parse_args())))  # raises if too few/many/unsupported args used
# else:
    # print(f"Wrong Input\n\tGW ID={gwid}\n\tInput File={infile}")


