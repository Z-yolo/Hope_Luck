# Anonymous reproduction package

**Main source code for RCC-ICLR2027.**

## Evaluate supplied checkpoints

Due to the size limit of anonymous GitHub, it is not possible to upload checkpoints.

```bash
python rcc_guard.py eval \
  --dataset cremad \
  --data-root <CREMAD_ROOT> \
  --checkpoint ckpt/cremad/best.pth \
  --output outputs/cremad

python rcc_guard.py eval \
  --dataset avsbench \
  --data-root <AVSBENCH_ROOT> \
  --checkpoint ckpt/avsbench/best.pth \
  --output outputs/avsbench

```
## Train from scratch

```bash
python rcc_guard.py train \
  --dataset cremad \
  --data-root <CREMAD_ROOT> \
  --output runs/cremad \
  --gpu-ids 0

python rcc_guard.py train \
  --dataset avsbench \
  --data-root <AVSBENCH_ROOT> \
  --output runs/avsbench \
  --gpu-ids 0
```

