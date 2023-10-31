from dvclive import Live
import random
NUM_EPOCHS = 20


def train_model(i):
    print(f"Training epoch {i} of {NUM_EPOCHS}")


def evaluate_model():
    metrics = {}
    metrics["accuracy"] = random.randrange(1, 100)
    metrics["loss"] = random.randrange(1, 20)
    return metrics


with Live() as live:
    live.log_param("epochs", NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        train_model(epoch)
        metrics = evaluate_model()
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)
        live.next_step()

    live.log_artifact("model.pkl", type="model")
