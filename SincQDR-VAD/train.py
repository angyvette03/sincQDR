import os
# os.environ["WANDB_MODE"] = "offline" 
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import wandb
from tqdm import tqdm

from metrics import (
    merge_intervals,
    compute_time_metrics_multi,
    time_based_pr_roc
)

from scipy.ndimage import median_filter

from dataset import SCF, AVA
from model.sincqdrvad import SincQDRVAD
from function.util import WarmupHoldDecayScheduler, save_best_k_model_with_auroc, median_smoothing_filter, metrics_calculation
from function.loss import QDRLoss
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from torch.utils.data import Subset
from sklearn.metrics import roc_curve


# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW_SIZE = 0.63
SINC_CONV = True
QDR_LOSS_WEIGHT = 0.25
QDR_LOSS_TYPE = 'psq'

# BATCH_SIZE = 256
BATCH_SIZE = 8
OVERLAP = 0.875
PATCH_SIZE = 8
MEDIAN_KERNEL_SIZE = 7

EPOCHS = 8

FRAME_SHIFT = WINDOW_SIZE * (1 - OVERLAP)  # seconds per frame
THRESHOLD = 0.5

EXP_NAME = 'sinc_qdr_vad'

exp_dir = "/home/users/ntu/angy0091/scratch/SincQDR-VAD-All"
os.makedirs(exp_dir, exist_ok=True)

# Initialize wandb
# wandb.init(project="SincQDR-VAD", name=EXP_NAME,config={
#     "seed": 42,
#     "epochs": 150,
#     "batch_size": BATCH_SIZE,
#     "max_lr": 0.01,
#     "momentum": 0.9,
#     "weight_decay": 0.001,
#     "warmup_ratio": 0.05,
#     "hold_ratio": 0.45,
#     "min_lr": 0.001,
#     "augment": True,
#     "window_size": WINDOW_SIZE,
#     "sinc_conv": SINC_CONV,
#     "qdr_loss_weight": QDR_LOSS_WEIGHT,
#     "qdr_loss_type": QDR_LOSS_TYPE,
# })
# config = wandb.config
config={
    "seed": 42,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.001,
    "warmup_ratio": 0.05,
    "hold_ratio": 0.45,
    "min_lr": 0.001,
    "augment": True,
    "window_size": WINDOW_SIZE,
    "sinc_conv": SINC_CONV,
    "qdr_loss_weight": QDR_LOSS_WEIGHT,
    "qdr_loss_type": QDR_LOSS_TYPE,
}

torch.manual_seed(config["seed"])

# Setup logging
log_file = os.path.join(exp_dir, 'train_allValFiles_8Epochs_29Jan.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# Load datasets
train_manifests = [
    # f'./data/manifest/{WINDOW_SIZE}/balanced_background_training_manifest.json', 
    "/home/users/ntu/angy0091/scratch/balanced_speech_training_manifest.json"]

logging.info('Loading training set ...')
train_dataset = SCF(
    manifest_files=train_manifests,
    sample_duration=config["window_size"],
    augment=config["augment"],
    feature_extraction=(not config["sinc_conv"]),
    )
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True) 
logging.info(f'Training set size: {len(train_loader)}')

logging.info('Loading validation set...')
# ----- actual -----
val_dir = '/home/users/ntu/angy0091/scratch/SincQDR_Val'
# ----- end of actual -----

# ----- debugging -----
# val_dir = '/home/users/ntu/angy0091/scratch/SincQDR_Val_Debug'
# ----- end of debugging

val_dataset = AVA(
    root_dir=val_dir,
    sample_duration=config["window_size"],
    overlap=OVERLAP,
    feature_extraction=(not config["sinc_conv"]),
)

# ----- debugging purposes only ------
# val_subset = Subset(val_dataset, [0])
# val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
# ----- debugging purposes only end ------

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
logging.info(f'Validation set size: {len(val_loader)}')

logging.info('Finish loading dataset!')
print('------------------------------')

# Initialize model, loss function, and optimizer
model = SincQDRVAD(1, 32, 64, PATCH_SIZE, 2, config["sinc_conv"]).to(device)
bce_criterion = nn.BCEWithLogitsLoss()
qdr_criterion = QDRLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=config["max_lr"],
    momentum=config["momentum"],
    weight_decay=config["weight_decay"]
)
scheduler = WarmupHoldDecayScheduler(
    optimizer=optimizer,
    total_steps=len(train_loader) * config["epochs"],
    warmup_ratio=config["warmup_ratio"],
    hold_ratio=config["hold_ratio"],
    min_lr=config["min_lr"],
    max_lr=config["max_lr"],
)

# ----- Training loop -----
top_3_val_auroc = []
print(f"Number of samples in dataset: {len(train_dataset)}")
logging.info(f"Number of samples in dataset: {len(train_dataset)}")

for epoch in range(config["epochs"]):
    model.train()
    running_loss = 0.0
    # val_probs_list, val_labels_list = [], []
    logging.info(f"Epoch {epoch}")

    train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch + 1}/{config['epochs']}] Training")

    for batch_idx, batch in train_progress_bar:
        inputs, labels = batch[0].to(device), batch[1].float().unsqueeze(1).to(device)
        # print(f'Inputs: {inputs.shape}, Labels: {labels}')
        print(f"Epoch {epoch+1}/{config['epochs']} - Batch {batch_idx+1}/{len(train_loader)}")
        print(f"  Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
        print(f"  First input sample min/max: {inputs[0].min().item()}/{inputs[0].max().item()}")
        print(f"  First label sample: {labels[0].item()}")
        
        logging.info(f"Epoch {epoch+1}/{config['epochs']} - Batch {batch_idx+1}/{len(train_loader)}")
        # logging.info(f"  Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
        # logging.info(f"  First input sample min/max: {inputs[0].min().item()}/{inputs[0].max().item()}")
        # logging.info(f"  First label sample: {labels[0].item()}")

        optimizer.zero_grad()

        try:
            outputs = model(inputs)
            logging.info("Forward pass OK")
            bce_loss = bce_criterion(outputs, labels)
            qdr_loss = qdr_criterion(outputs, labels)
            loss = (1 - QDR_LOSS_WEIGHT) * bce_loss + QDR_LOSS_WEIGHT * qdr_loss
            loss.backward()
            logging.info("Backward pass OK")
            optimizer.step()
            scheduler.step()
            logging.info("Step OK")
            logging.info(f"Output shape = {outputs.shape}, Labels shape = {labels.shape}")

        except Exception as e:
            logging.exception(f"Error at batch {batch_idx+1}: {e}")
            break

        if batch_idx % 10 == 0:
            logging.info(
                f"Epoch [{epoch + 1}/{config['epochs']}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}, "
                f"BCE Loss: {(1 - QDR_LOSS_WEIGHT) * bce_loss.item():.4f}, "
                f"QDR Loss: {QDR_LOSS_WEIGHT * qdr_loss:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    logging.info(f"Epoch [{epoch + 1}/{config['epochs']}] Train Loss: {avg_train_loss:.4f}")
    

    # wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_train_loss})

    
    # logging.info(f"Epoch [{epoch + 1}/{config['epochs']}] AUROC: {auroc}, FPR: {fpr}, FNR: {fnr}, F2-score: {f2_score}")
    # wandb.log({"auroc": auroc, "fpr": fpr, "fnr": fnr, "f2-score": f2_score})

    # Save the best 3 models
    # save_best_k_model_with_auroc(exp_dir, model, epoch, auroc, top_3_val_auroc, k=3)
# ----- end of Training loop -----


val_probs_list, val_labels_list = [], []

gt_list = []
preds_with_scores_list = []
total_lengths = []

# Validation step
# checkpoint_path = "/home/users/ntu/angy0091/scratch/SincQDR-VAD-All/model_last_epoch.ckpt"
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# with torch.no_grad():
#     for batch in tqdm(val_loader, desc='Validating'):
#         if config["sinc_conv"]:
#             val_inputs = [item[0].to(device) for item in batch]
#         else:
#             val_inputs = [item[1].to(device) for item in batch]
#         val_labels = [item[2].to(device).float().unsqueeze(1) for item in batch]
#         val_inputs = torch.cat(val_inputs, dim=0)
#         val_labels = torch.cat(val_labels, dim=0)

#         val_probs = model.predict(val_inputs)

#         # Apply median smoothing filter
#         val_probs_list, val_labels_list = median_smoothing_filter(val_probs, val_labels, val_probs_list, val_labels_list, MEDIAN_KERNEL_SIZE, device)

# ----- trying out new metrics -----

with torch.no_grad():
    for idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
        try:
            segments = batch  # list of (wave, mel, label)

            waves  = [seg[0] for seg in segments]
            mels   = [seg[1] for seg in segments]
            labels = [seg[2] for seg in segments]
            
            logging.info(
                f"[VAL] idx={idx} | "
                f"waves={len(waves)} | "
                f"frames={waves[0].shape[0]} | "
                f"labels={len(labels)}"
            )

            # ----- inputs -----
            if config["sinc_conv"]:
                val_inputs = torch.cat(waves, dim=0).to(device)
            else:
                val_inputs = torch.cat(mels, dim=0).to(device)

            val_labels = torch.cat(labels, dim=0).float().to(device)

            # logging.info(
            #     f"Input shape={val_inputs.shape}, "
            #     f"Labels shape={val_labels.shape}"
            # )

            # ----- forward -----
            probs = torch.sigmoid(model(val_inputs)).squeeze(1)
            
            # ---- frame-level storage ----
            # val_probs_list.append(probs.unsqueeze(1))
            # val_labels_list.append(val_labels.unsqueeze(1))
            val_probs_list.append(probs.detach().cpu())
            val_labels_list.append(val_labels.detach().cpu())


            # # for time based predictions
            # # ----- median smoothing -----
            # probs = torch.from_numpy(
            #     median_filter(probs.cpu().numpy(), size=MEDIAN_KERNEL_SIZE)
            # ).to(val_labels.device).float()

            # # ---- time-based GT ----
            # gt_segments = []
            # for i, y in enumerate(val_labels):
            #     if y == 1:
            #         start = i * FRAME_SHIFT
            #         end = start + FRAME_SHIFT
            #         gt_segments.append((start, end))

            # gt_segments = merge_intervals(gt_segments)
            # gt_list.append(gt_segments)
            # total_lengths.append(len(val_labels) * FRAME_SHIFT)

            # # ---- time-based predictions ----
            # pred_intervals_with_scores = []
            # for i, p in enumerate(probs):
            #     if p >= THRESHOLD:
            #         start = i * FRAME_SHIFT
            #         end = start + FRAME_SHIFT
            #         pred_intervals_with_scores.append((start, end, float(p)))

            # preds_with_scores_list.append(pred_intervals_with_scores)

        except Exception:
            logging.exception(f"[VAL] Error at idx={idx}")

logging.info(
    f"[VAL DONE] "
    # f"files={len(gt_list)} | "
    # f"total_frames={sum(v.shape[0] for v in val_labels_list)}"
)

# --- test test ---

print(f"len(val_labels_list): {len(val_labels_list)}, len(val_probs_list): {len(val_probs_list)}")


# Concatenate results

val_labels_cat = torch.cat(val_labels_list, dim=0).cpu().numpy()
val_probs_cat = torch.cat(val_probs_list, dim=0).cpu().numpy()

# val_labels_cat = torch.cat(val_labels_list, dim=0).cpu().numpy().ravel()
# val_probs_cat  = torch.cat(val_probs_list, dim=0).cpu().numpy().ravel()

# Compute ROC
roc_fpr, roc_tpr, thresholds = roc_curve(val_labels_cat, val_probs_cat)
roc_fnr = 1 - roc_tpr  # FNR = 1 - TPR

# w = 0.48  
# score = w * roc_fpr + (1 - w) * roc_fnr

# # Pick threshold that minimizes weighted score
# best_idx = np.argmin(score)
# THRESHOLD = thresholds[best_idx]
THRESHOLD = 0.80

logging.info(f"Chosen threshold (weighted FPR/FNR): {THRESHOLD:.4f}")
# logging.info(f"FPR={roc_fpr[best_idx]:.4f}, FNR={roc_fnr[best_idx]:.4f}")


# Metrics calculation
auroc, fpr, fnr, tpr, tnr, f2_score = metrics_calculation(val_labels_cat, val_probs_cat, THRESHOLD)
# threshold = 0.5
val_preds = (val_probs_cat >= THRESHOLD).astype(int)
val_true = val_labels_cat.astype(int)

# Compute confusion matrix: TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(val_true, val_preds).ravel()

# Compute other metrics
precision = precision_score(val_true, val_preds, zero_division=0)
recall = recall_score(val_true, val_preds, zero_division=0)
accuracy = accuracy_score(val_true, val_preds)

# # ----- time based metrics -----
# time_stats = compute_time_metrics_multi(
#     gt_list,
#     [[(a, b) for (a, b, _) in preds] for preds in preds_with_scores_list],
#     total_lengths
# )

# logits_list = [p.squeeze(1).cpu().numpy() for p in val_probs_list]

# curves_t = time_based_pr_roc(
#     gt_list,
#     logits_list,
#     total_lengths
# )

# logging.info(
#     f"[TIME] P={time_stats['precision']:.4f} | "
#     f"R={time_stats['recall']:.4f} | "
#     f"TP={time_stats['TP']:.4f} | "
#     f"FP={time_stats['FP']:.4f} | "
#     f"FN={time_stats['FN']:.4f} | "
#     f"TN={time_stats['TN']:.4f} | "
#     f"Acc={time_stats['accuracy']:.4f} | "
#     f"AUC={curves_t['AUROC']:.4f}"
# )
# # ----- time based metrics -----

# logging.info(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
# logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

# logging.info(f"AUROC: {auroc}, FPR: {fpr}, FNR: {fnr}, F2-score: {f2_score}")

logging.info(
    f"[FRAME] AUROC={auroc:.4f} | "
    f"P={precision:.4f} | R={recall:.4f} | "
    f"Acc={accuracy:.4f} | F2={f2_score:.4f} | "
    f"FPR={fpr:.4f} | FNR={fnr:.4f} | TPR={tpr:.4f} | TNR={tnr:.4f}"
)

# After last epoch, save final model
final_checkpoint = os.path.join(exp_dir, f'model_last_epoch_29Jan.ckpt')
torch.save(model.state_dict(), final_checkpoint)
logging.info(f'Final model saved to {final_checkpoint}')

logging.info('Training complete!')
# wandb.finish()
