import logging
import torch
from dataloader import *
from emmental.data import EmmentalDataLoader, EmmentalDataset

logger = logging.getLogger(__name__)

def load_data(filename, opt, vocab, evaluation=False):
    # Load data from file
    data, label = load_tacred(filename, opt, vocab, evaluation)
    label = torch.LongTensor(label)
    return data, label


def create_dataloaders(task_name, dataset, batch_size, word2id, oov="~#OoV#~"):
    # Create dataloaders
    oov_id = word2id[oov]
    dataloaders = []

    for split in ["train", "valid", "test"]:
        split_x, split_y = dataset[split]
        split_x = [
            torch.LongTensor([word2id.get(w, oov_id) for w in seq]) for seq in split_x
        ]

        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={task_name: "label"},
                dataset=EmmentalDataset(
                    name=task_name,
                    X_dict={"feature": split_x},
                    Y_dict={"label": split_y},
                ),
                split=split,
                batch_size=batch_size,
                shuffle=True if split == "train" else False,
            )
        )
        logger.info(
            f"Loaded {split} for {task_name} containing {len(split_x)} samples."
        )

    return dataloaders
