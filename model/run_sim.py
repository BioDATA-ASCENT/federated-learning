from flwr.simulation import run_simulation

from track_a.client_app import app as client_app
from track_a.server_app import app as server_app
from track_a.task import client_files

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=len(client_files()),   # one node per client file
)
