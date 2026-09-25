# RCC: anonymous reproduction package

**Main source code for Reliability Configuration Calibration (RCC).**

## Evaluate supplied checkpoints


```bash
python rcc_guard.py eval \
  --dataset cremad \
  --data-root <CREMAD_ROOT> \
  --checkpoint checkpoints/cremad/best.pth \
  --output outputs/cremad

python rcc_guard.py eval \
  --dataset avsbench \
  --data-root <AVSBENCH_ROOT> \
  --checkpoint checkpoints/avsbench/best.pth \
  --output outputs/avsbench

```

