import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from transformers import RobertaTokenizer
from transformers import AutoTokenizer

from model import SupConModel
from data_loader import ContrastiveFunctionDataset, SingleFunctionDataset
from trainer import Trainer
from supcon_loss import PairSupConLoss
from trainer import logger
from paircollator import PairCollatorPool
from utils import make_val_splits

import argparse
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # === CLI ARGs ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # === Load Config ===
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.get('seed', 42))

    # === Model & Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = SupConModel(pretrained_model="microsoft/codebert-base")

    # === Optimizer & Scheduler ===
    lr = float(config.get('lr', 2e-5))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = None  # placeholder, add your own scheduler if needed

    # === Loss Function ===
    supcon_loss = PairSupConLoss(temperature=config.get('temperature', 0.07))

    # === Datasets ===
    if args.mode == 'train':
        # Read mode from configuration (default func+pair)
        # val_mode = config.get("val_split_mode", "func+pair")  # "func" or "func+pair"
        # train_split_csv, val_split_csv = make_val_splits(
        #     train_csv=config["train_csv"],
        #     val_ratio=config.get("val_ratio", 0.1),
        #     mode=val_mode,
        #     seed=config.get("seed", 42),
        #     out_dir=config.get("split_cache_dir", ".cache_splits")
        # )

        train_data = ContrastiveFunctionDataset(
            csv_path=config["train_csv"],
            max_pairs_per_type=config.get('max_pairs_per_type', 10000),
            balance_cls_pairs=config.get('balance_cls_pairs', True),
            balance_all_pairs=config.get('balance_all_pairs', True),
            cwe_filter=config.get("cwe_filter", None)
        )
        val_data = ContrastiveFunctionDataset(
            csv_path=config["val_csv"],
            max_pairs_per_type=config.get('max_pairs_per_type_val', config.get('max_pairs_per_type', 3000)),
            balance_cls_pairs=config.get('balance_cls_pairs', True),
            balance_all_pairs=config.get('balance_all_pairs', True),
            cwe_filter=config.get("cwe_filter", None)
        )
    

    # === Dataloaders ===
    collator = PairCollatorPool(tokenizer)
    if args.mode == 'train':
        train_loader = DataLoader(train_data, batch_size=config.get('batch_size', 16), shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_data, batch_size=config.get('batch_size', 16), shuffle=False, collate_fn=collator)

    # === Trainer ===
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        optimizer=optimizer,
        supcon_loss_fn=supcon_loss,
        scheduler=scheduler,
        config_path=args.config
    )
    
    if args.mode == 'train':
        best_auprc, bad_epochs = 0.0, 0
        patience = config.get('patience', 3)
        for epoch in range(config.get('epochs', 10)):
            logger.info(f"===== Epoch {epoch} =====")
            trainer.train_epoch(train_loader)
            val_result = trainer.evaluate(val_loader, name="Val")
            auprc = val_result.get('single_auprc', val_result.get('pair_auprc', 0.0))
            if auprc > best_auprc + 1e-6:
                best_auprc, bad_epochs = auprc, 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info(f"⏹ Early stop at epoch {epoch}, best_auprc={best_auprc:.4f}")
                    break

    if args.mode == 'test' or config['test_csv']:
        test_data = SingleFunctionDataset(csv_path=config['test_csv'], tokenizer=tokenizer, cwe_filter=config.get("cwe_filter", None))
        test_loader = DataLoader(test_data, batch_size=config.get('batch_size', 16), shuffle=False)
        logger.info("✅ Loading best model for final test")
        model.load_state_dict(torch.load(config.get('checkpoint_path', 'best_model.pt'), map_location=device))
        test_result = trainer.test_single(test_loader, test_path=config['test_csv'])

if __name__ == "__main__":
    main()
