import numpy as np
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    strategy = FedAvg(min_train_nodes=2, min_available_nodes=2)
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord([np.array([0.0], dtype=np.float32)]),
        num_rounds=int(context.run_config["num-server-rounds"]),
    )
    final_value = result.arrays.to_numpy_ndarrays()[0].item()
    print(f"Smoke test complete: final averaged value = {final_value}")
