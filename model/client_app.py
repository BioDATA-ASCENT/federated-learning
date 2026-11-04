"""what each lab (client) runs. Same shape as the quickstart's
client_app.py: a train handler and an evaluate handler."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from track_a.task import LogisticRegressionModel, get_labels, load_data, test, train

app = ClientApp()


@app.train()
def client_train(msg: Message, context: Context):
    # 1. Which client am I? Flower gives each node its partition-id.
    partition_id = context.node_config["partition-id"]
    X_train, y_train, _, _, _ = load_data(partition_id)
    n_labels, n_genes = len(get_labels()), X_train.shape[1]

    # 2. Build the model and load the global weights the server sent.
    model = LogisticRegressionModel(n_labels, n_genes)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # 3. Train locally for E epochs.
    epochs = msg.content["config"]["local-epochs"]
    train(model, X_train, y_train, epochs)

    # 4. Send back the new weights and how many cells were used.
    #    "num-examples" is what FedAvg uses to weight this client.
    content = RecordDict({
        "arrays": ArrayRecord(model.get_weights()),
        "metrics": MetricRecord({"num-examples": len(y_train)}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def client_evaluate(msg: Message, context: Context):
    # Same as above, but test instead of train, and send back only metrics.
    partition_id = context.node_config["partition-id"]
    _, _, X_test, y_test, name = load_data(partition_id)
    model = LogisticRegressionModel(len(get_labels()), X_test.shape[1])
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    f1, acc = test(model, X_test, y_test)
    print(f"[client {name}] macro F1 = {f1:.3f}  accuracy = {acc:.3f}  (n_test = {len(y_test)})")
    content = RecordDict({
        "metrics": MetricRecord({"macro-f1": f1, "accuracy": acc, "num-examples": len(y_test)}),
    })
    return Message(content=content, reply_to=msg)
