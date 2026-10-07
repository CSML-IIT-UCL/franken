import json
import matplotlib.pyplot as plt


def plot_training_history(path):
    """
    Create a 2x2 panel plot:

        Energy RMSE (Train)      Energy RMSE (Validation)
        Forces RMSE (Train)      Forces RMSE (Validation)

    RFF and LES steps are shown separately.
    """
    json_file = path / "training_history.json"
    with open(json_file, "r") as f:
        history = json.load(f)

    cycles_rff = []
    cycles_les = []

    train_energy_rff = []
    train_energy_les = []

    val_energy_rff = []
    val_energy_les = []

    train_forces_rff = []
    train_forces_les = []

    val_forces_rff = []
    val_forces_les = []

    for entry in history:
        cycle = entry["cycle"]
        step = entry["step"]

        train = entry["metrics"]["train"]
        val = entry["metrics"]["validation"]

        if step == "rff":
            cycles_rff.append(cycle)

            train_energy_rff.append(train["energy_RMSE"])
            val_energy_rff.append(val["energy_RMSE"])

            train_forces_rff.append(train["forces_RMSE"])
            val_forces_rff.append(val["forces_RMSE"])

        elif step == "les":
            cycles_les.append(cycle)

            train_energy_les.append(train["energy_RMSE"])
            val_energy_les.append(val["energy_RMSE"])

            train_forces_les.append(train["forces_RMSE"])
            val_forces_les.append(val["forces_RMSE"])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # ------------------------------
    # Energy RMSE - Train
    # ------------------------------
    axs[0, 0].plot(
        cycles_rff,
        train_energy_rff,
        "o-",
        label="RFF",
        lw=2
    )

    axs[0, 0].plot(
        cycles_les,
        train_energy_les,
        "s-",
        label="LES",
        lw=2
    )

    axs[0, 0].set_title("Train Energy RMSE")
    axs[0, 0].set_ylabel("RMSE [meV/atom]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # ------------------------------
    # Energy RMSE - Validation
    # ------------------------------
    axs[0, 1].plot(
        cycles_rff,
        val_energy_rff,
        "o-",
        label="RFF",
        lw=2
    )

    axs[0, 1].plot(
        cycles_les,
        val_energy_les,
        "s-",
        label="LES",
        lw=2
    )

    axs[0, 1].set_title("Validation Energy RMSE")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # ------------------------------
    # Forces RMSE - Train
    # ------------------------------
    axs[1, 0].plot(
        cycles_rff,
        train_forces_rff,
        "o-",
        label="RFF",
        lw=2
    )

    axs[1, 0].plot(
        cycles_les,
        train_forces_les,
        "s-",
        label="LES",
        lw=2
    )

    axs[1, 0].set_title("Train Forces RMSE")
    axs[1, 0].set_xlabel("Cycle")
    axs[1, 0].set_ylabel("RMSE [meV/A]")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # ------------------------------
    # Forces RMSE - Validation
    # ------------------------------
    axs[1, 1].plot(
        cycles_rff,
        val_forces_rff,
        "o-",
        label="RFF",
        lw=2
    )

    axs[1, 1].plot(
        cycles_les,
        val_forces_les,
        "s-",
        label="LES",
        lw=2
    )

    axs[1, 1].set_title("Validation Forces RMSE")
    axs[1, 1].set_xlabel("Cycle")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(path / "History_training.png")
    plt.show()


