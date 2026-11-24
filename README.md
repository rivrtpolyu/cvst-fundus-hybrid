# Fundus Hybrid Classification (ConvNeXt V2 + DINOv3)

Patient-disjoint retinal fundus classifier (`Normal` vs. `CVST Abnormal`) with optic-disc-aware ROI extraction, four architectural variants (`global`, `disc`, `dual`, and `hybrid` cross-attention fusion), baseline benchmarks, and gradient-based explainability (`rollout`, `grad-rollout`, `grad-cam`).

---

## Repository Structure

```text
cvst-fundus-hybrid/
├─ configs/
│  ├─ base.yaml                 # Default training hyperparameters (AMP, grad accumulation, checkpointing)
│  ├─ global.yaml               # Full-fundus DINOv3 stream
│  ├─ disc.yaml                 # Optic-disc ROI ConvNeXt V2 stream
│  ├─ dual.yaml                 # Single-backbone dual-view fusion
│  ├─ hybrid.yaml               # Proposed Dual-Stream Hybrid (ConvNeXt V2 + DINOv3 Cross-Attention)
│  └─ explain.yaml              # Explainability settings
├─ data/
│  ├─ normal/                   # [patient_id]-[age]-[gender]/*.JPG
│  └─ abnormal/                 # [patient_id]-[YY-MM-DD]/*.JPG
├─ weights/
│  ├─ dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
│  └─ convnextv2_base_1k_224_fcmae.pt
├─ dinov3/                      # Local DINOv3 repository for offline torch.hub loading
├─ outputs/                     # Auto-generated manifests, splits, checkpoints, predictions, figures & logs
├─ core.py                      # Config, I/O, seed/device, AMP helpers, and evaluation metrics
├─ anatomy.py                   # OD/OS-aware optic disc localization, FOV crop, and ROI masking
├─ backbones.py                 # DINOv3, ConvNeXt V2, CrossAttentionFusion, and model definitions
├─ data.py                      # Manifest generation, patient-disjoint splits, transforms, and Datasets
├─ explain.py                   # Attention rollout, grad-rollout, and Grad-CAM CLI
├─ train.py                     # Training entrypoint (AMP + gradient accumulation + checkpointing)
├─ evaluate.py                  # Evaluation, model comparison, and ablation summary CLI
├─ prepare_data.py              # Dataset audit, manifest, and split generation CLI
├─ requirements.txt
└─ README.md
