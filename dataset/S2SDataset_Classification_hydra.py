from transformers import AutoTokenizer
from dataset.utils import dsets
from dataset.utils.datasetbase import DatasetBase


class S2SDataset_Classification(DatasetBase):
    NAME = "mcdataset"  # mutil-choice dataset
    task_info = {
        "winogrande_s": {
            "num_labels": 2,
        },
        "winogrande_m": {
            "num_labels": 2,
        },
        "boolq": {
            "num_labels": 2,
        },
        "obqa": {
            "num_labels": 4,
        },
        "ARC-Easy": {
            "num_labels": 5,
        },
        "ARC-Challenge": {
            "num_labels": 5,
        },
    }

    #def __init__(self, accelerator, args):
    def __init__(self, tokenizer, name='winogrande_s', testing_set='val', add_space=False, max_seq_len=300, batch_size=4, is_s2s=False, instruct=False):
        super().__init__()

        # self.args = args
        # self.accelerator = accelerator

        
        # accelerator.wait_for_everyone()
        self.tokenizer = tokenizer
        self.dataset = name
        self.testing_set = testing_set
        self.add_space = add_space
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.is_s2s = is_s2s
        self.instruct = instruct


        # self.tokenizer = AutoTokenizer.from_pretrained(
            # self.model, trust_remote_code=True
        # )
        self.tokenizer.padding_side = "left"
        if self.dataset in ["boolq", "winogrande_m", "winogrande_s"]:
            self.tokenizer.add_eos_token = True

        if self.tokenizer.pad_token is None:
            if self.tokenizer.bos_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.bos_token
        # self.tokenizer.pad_token = self.tokenizer.bos_token
        if self.dataset in self.task_info:
            self.num_labels = self.task_info[self.dataset]["num_labels"]
        elif self.dataset.startswith("MMLU"):
            self.num_labels = 4
        else:
            raise NotImplementedError

        if self.dataset.startswith("winogrande"):
            dset_class: dsets.ClassificationDataset = getattr(dsets, "winogrande")
            self.dset = dset_class(
                self.tokenizer,
                add_space=self.add_space,
                name=self.dataset,
                max_seq_len=self.max_seq_len,
                instruct=self.instruct,
            )
        elif self.dataset.startswith("ARC"):
            dset_class: dsets.ClassificationDataset = getattr(dsets, "arc")
            self.dset = dset_class(
                self.tokenizer,
                add_space=self.add_space,
                name=self.dataset,
                max_seq_len=self.max_seq_len,
                instruct=self.instruct,
            )
        elif self.dataset.startswith("MMLU"):
            dset_class: dsets.ClassificationDataset = getattr(dsets, "mmlu")
            self.dset = dset_class(
                self.tokenizer,
                add_space=self.add_space,
                name=self.dataset[5:],
                max_seq_len=self.max_seq_len,
                instruct=self.instruct,
            )
        else:
            dset_class: dsets.ClassificationDataset = getattr(dsets, self.dataset)
            self.dset = dset_class(
                self.tokenizer, add_space=self.add_space, max_seq_len=self.max_seq_len,
                instruct=self.instruct,
            )

        # if accelerator.is_local_main_process:
        if True:
            print("=====================================")
            print(f"Loaded {self.dataset} dataset.")
            print("=====================================")

    def get_loaders(self):
        """
        Returns the train and test data loaders.
        """

        self.target_ids = self.dset.target_ids

        if self.dataset.startswith("MMLU"):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.is_s2s,  # sequence to sequence model?
                batch_size=self.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count
            self.test_dataloader = self.dset.loader(
                is_s2s=self.is_s2s,  # sequence to sequence model?
                batch_size=self.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return

        self.train_dataloader = self.dset.loader(
            is_s2s=self.is_s2s,  # sequence to sequence model?
            batch_size=self.batch_size,  # training batch size
            split="train",  # training split name in dset
            subset_size=-1,  # train on subset? (-1 = no subset)
        )
        total_data_count = 0
        for batch in self.train_dataloader:
            total_data_count += batch[1].size(0)
        self.num_samples = total_data_count

        self.test_dataloader = self.dset.loader(
            is_s2s=self.is_s2s,  # sequence to sequence model?
            batch_size=self.batch_size,  # training batch size
            split="validation",  # training split name in dset
            subset_size=-1,  # train on subset? (-1 = no subset)
        )

        if self.testing_set != "val":
            raise NotImplementedError("Only validation set is supported for now.")
