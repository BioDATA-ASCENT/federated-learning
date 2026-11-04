import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    client_delta = float(context.node_config["client-delta"])
    updated_arrays = [arrays[0] + np.array([client_delta], dtype=arrays[0].dtype)]

    content = RecordDict(
        {
            "arrays": ArrayRecord(updated_arrays),
            "metrics": MetricRecord({"num-examples": 1}),
        }
    )
    return Message(content=content, reply_to=msg)
