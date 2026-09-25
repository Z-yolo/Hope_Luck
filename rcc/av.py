"""Shared AV RCC loss, calibration, training and evaluation implementation."""

import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .metrics import build_evaluation_summary


def random_different_label_indices(labels):
    """Sample one different-label source per example, preserving the RCC RNG order."""
    values = labels.detach().cpu().long().tolist()
    chosen = []
    for i, label in enumerate(values):
        candidates = [j for j, other in enumerate(values) if j != i and other != label]
        chosen.append(random.choice(candidates) if candidates else i)
    return torch.tensor(chosen, device=labels.device, dtype=torch.long)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def unpack_av_batch(batch):
    """Support the three-field and six-field released dataset batches.

    CREMA-D returns:
        (spectrogram, images, label)

    AVSBench returns:
        (spectrogram, images, audio_noise, visual_noise, label, idx)

    RCC_guard only needs spectrogram/images/label, so we safely ignore the
    optional noise/index fields.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            spec, image, label = batch
            return (spec, image, label)
        if len(batch) >= 5:
            spec, image, label = (batch[0], batch[1], batch[4])
            return (spec, image, label)
    raise ValueError(
        f"Unsupported AV batch format: type={type(batch)}, len={(len(batch) if hasattr(batch, '__len__') else 'NA')}"
    )


def resolve_num_classes_from_name(dataset_name: str) -> int:
    """Centralized class-number mapping used by main/eval and extension datasets."""
    mapping = {
        "CREMAD": 6,
        "cremad": 6,
        "avsbench": 23,
        "AVSBench": 23,
    }
    if dataset_name not in mapping:
        raise NotImplementedError("Incorrect dataset name {}".format(dataset_name))
    return mapping[dataset_name]


def get_num_classes_runtime(args) -> int:
    n = int(getattr(args, "num_classes", -1))
    if n > 0:
        return n
    return resolve_num_classes_from_name(str(args.dataset))


def supervised_margin_torch(logits, labels):
    """
    logits: [B, C]
    labels: [B]
    return logit_y - max_{c != y} logit_c
    """
    labels = labels.long()
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), -1000000000.0)
    comp_logits = masked.max(dim=1).values
    return true_logits - comp_logits


def _get_fusion_module(model):
    return (
        model.module.fusion_module if hasattr(model, "module") else model.fusion_module
    )


def _get_active_rcc_probs(args):
    """Return currently active RCC probabilities (audio->visual, visual->audio)."""
    pa = getattr(args, "_rcc_p_a2v", None)
    pv = getattr(args, "_rcc_p_v2a", None)
    if pa is None:
        pa = args.rcc_prob_a2v if float(args.rcc_prob_a2v) >= 0 else 0.0
    if pv is None:
        pv = args.rcc_prob_v2a if float(args.rcc_prob_v2a) >= 0 else 0.0
    pa = max(0.0, min(float(args.rcc_prob_max), float(pa)))
    pv = max(0.0, min(float(args.rcc_prob_max), float(pv)))
    return (pa, pv)


def _weighted_ce_for_selected(logits, labels, selected, gate=None):
    """Per-sample CE over selected samples, optionally weighted by a detached gate."""
    if selected.sum().item() == 0:
        return torch.zeros([], device=logits.device)
    ce = F.cross_entropy(logits, labels, reduction="none")
    if gate is None:
        return ce[selected].mean()
    gate = gate.detach().clamp(min=0.0, max=1.0)
    w = gate[selected]
    denom = w.sum()
    if denom.item() <= 1e-08:
        return torch.zeros([], device=logits.device)
    return (ce[selected] * w).sum() / (denom + 1e-08)


def compute_rcc_loss(args, model, feat_a, feat_v, labels, logits_a=None, logits_v=None):
    """
    RCC: Reliability-Configuration Calibration.

    audio -> visual direction:
        f(a_shuf, v) -> y
        It injects the missing configuration where audio is present but unreliable,
        while visual remains the target reliable modality.

    visual -> audio direction:
        f(a, v_shuf) -> y
        It injects the missing configuration where visual is present but unreliable,
        while audio remains the target reliable modality.

    Direction probabilities are not necessarily symmetric. In calibrated mode, they
    are estimated from held-out reliability-configuration gaps.
    """
    device = labels.device
    B = labels.size(0)
    if B <= 1 or float(args.lambda_rcc) <= 0:
        return torch.zeros([], device=device)
    p_a2v, p_v2a = _get_active_rcc_probs(args)
    fusion_module = _get_fusion_module(model)
    total = torch.zeros([], device=device)
    arange = torch.arange(B, device=device)
    if p_a2v > 0:
        idx_a = random_different_label_indices(labels)
        changed_a = idx_a != arange
        selected_a = (torch.rand(B, device=device) < p_a2v) & changed_a
        a_src = feat_a[idx_a]
        if not args.rcc_update_source:
            a_src = a_src.detach()
        _, _, out_a2v = fusion_module(a_src, feat_v)
        gate_v = None
        if args.rcc_gate == "margin" and logits_v is not None:
            delta_v = supervised_margin_torch(logits_v.detach(), labels)
            tau = max(float(args.rcc_gate_tau), 1e-06)
            gate_v = torch.sigmoid((delta_v - float(args.rcc_gate_eta_v)) / tau)
        loss_a2v = _weighted_ce_for_selected(
            out_a2v, labels, selected_a, gate=gate_v
        )
        total = total + loss_a2v
    if p_v2a > 0:
        idx_v = random_different_label_indices(labels)
        changed_v = idx_v != arange
        selected_v = (torch.rand(B, device=device) < p_v2a) & changed_v
        v_src = feat_v[idx_v]
        if not args.rcc_update_source:
            v_src = v_src.detach()
        _, _, out_v2a = fusion_module(feat_a, v_src)
        gate_a = None
        if args.rcc_gate == "margin" and logits_a is not None:
            delta_a = supervised_margin_torch(logits_a.detach(), labels)
            tau = max(float(args.rcc_gate_tau), 1e-06)
            gate_a = torch.sigmoid((delta_a - float(args.rcc_gate_eta_a)) / tau)
        loss_v2a = _weighted_ce_for_selected(
            out_v2a, labels, selected_v, gate=gate_a
        )
        total = total + loss_v2a
    return total


def get_guard_lambda(args):
    """Return the effective loss weight for the adoption-preserving guard."""
    if float(getattr(args, "lambda_guard", -1.0)) >= 0:
        return float(args.lambda_guard)
    return float(getattr(args, "guard_ratio", 0.25)) * float(
        getattr(args, "lambda_rcc", 0.0)
    )


def get_guard_start_epoch(args):
    start = int(getattr(args, "guard_start_epoch", -1))
    if start < 0:
        start = int(getattr(args, "rcc_start_epoch", 0))
    return start


def compute_adoption_guard_loss(
    args, model, feat_a, feat_v, labels, logits_a=None, logits_v=None
):
    """
    Adoption-Preserving Guard (APG).

    RCC restores missing take-over configurations, e.g. f(a_shuf, v)->y for
    audio-unreliable/visual-reliable cases. When this intervention is too strong,
    the originally reliable modality may be over-suppressed. APG adds a weak,
    reliability-gated counter-configuration that preserves that modality.

    If active RCC direction is a2v, APG protects audio adoption by training
        f(a, v_shuf) -> y
    with probability p_guard_a = guard_prob_ratio * p_a2v.

    If active RCC direction is v2a, APG protects visual adoption by training
        f(a_shuf, v) -> y
    with probability p_guard_v = guard_prob_ratio * p_v2a.

    This is not symmetric RCC: APG probabilities and weights are derived from
    calibrated RCC directions and kept weak by construction.
    """
    device = labels.device
    B = labels.size(0)
    guard_lambda = get_guard_lambda(args)
    if (
        B <= 1
        or guard_lambda <= 0
        or float(getattr(args, "guard_prob_ratio", 0.0)) <= 0
    ):
        return torch.zeros([], device=device)
    p_a2v, p_v2a = _get_active_rcc_probs(args)
    prob_ratio = max(0.0, float(getattr(args, "guard_prob_ratio", 0.25)))
    p_guard_a = max(
        0.0, min(float(getattr(args, "rcc_prob_max", 1.0)), prob_ratio * float(p_a2v))
    )
    p_guard_v = max(
        0.0, min(float(getattr(args, "rcc_prob_max", 1.0)), prob_ratio * float(p_v2a))
    )
    fusion_module = _get_fusion_module(model)
    total = torch.zeros([], device=device)
    arange = torch.arange(B, device=device)
    detach_conflict = not args.guard_update_conflict
    if p_guard_a > 0:
        idx_v = random_different_label_indices(labels)
        changed_v = idx_v != arange
        selected = (torch.rand(B, device=device) < p_guard_a) & changed_v
        v_src = feat_v[idx_v]
        if detach_conflict:
            v_src = v_src.detach()
        _, _, out_guard_a = fusion_module(feat_a, v_src)
        gate_a = None
        if getattr(args, "guard_gate", "margin") == "margin" and logits_a is not None:
            delta_a = supervised_margin_torch(logits_a.detach(), labels)
            tau = max(float(getattr(args, "guard_tau", 1.0)), 1e-06)
            gate_a = torch.sigmoid(
                (delta_a - float(getattr(args, "guard_eta_a", 0.0))) / tau
            )
        loss_guard_a = _weighted_ce_for_selected(
            out_guard_a, labels, selected, gate=gate_a
        )
        total = total + loss_guard_a
    if p_guard_v > 0:
        idx_a = random_different_label_indices(labels)
        changed_a = idx_a != arange
        selected = (torch.rand(B, device=device) < p_guard_v) & changed_a
        a_src = feat_a[idx_a]
        if detach_conflict:
            a_src = a_src.detach()
        _, _, out_guard_v = fusion_module(a_src, feat_v)
        gate_v = None
        if getattr(args, "guard_gate", "margin") == "margin" and logits_v is not None:
            delta_v = supervised_margin_torch(logits_v.detach(), labels)
            tau = max(float(getattr(args, "guard_tau", 1.0)), 1e-06)
            gate_v = torch.sigmoid(
                (delta_v - float(getattr(args, "guard_eta_v", 0.0))) / tau
            )
        loss_guard_v = _weighted_ce_for_selected(
            out_guard_v, labels, selected, gate=gate_v
        )
        total = total + loss_guard_v
    return total


@torch.no_grad()
def collect_reliability_config_stats(args, model, loader, device, max_batches=0):
    """Collect only the four rates required by calibration."""
    model.eval()
    total = 0
    both_correct = audio_only = visual_only = 0
    for bidx, batch in enumerate(loader):
        if max_batches and bidx >= max_batches:
            break
        spec, image, label = unpack_av_batch(batch)
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device).long().view(-1)
        _, out_a, out_v = model(spec.unsqueeze(1).float(), image.float())
        pred_a = out_a.argmax(dim=1)
        pred_v = out_v.argmax(dim=1)
        corr_a = pred_a == label
        corr_v = pred_v == label
        n = label.numel()
        total += int(n)
        both_correct += int((corr_a & corr_v).sum().item())
        audio_only += int((corr_a & ~corr_v).sum().item())
        visual_only += int((~corr_a & corr_v).sum().item())
    denom = max(total, 1)
    return {
        "p_a_wrong_v_right": float(visual_only / denom),
        "p_v_wrong_a_right": float(audio_only / denom),
        "p_v_right": float((both_correct + visual_only) / denom),
        "p_a_right": float((both_correct + audio_only) / denom),
    }


def estimate_rcc_probabilities_from_gaps(args, train_stats, hold_stats):
    """
    Estimate direction probabilities from held-out reliability-configuration gaps.

    audio -> visual should compensate missing (audio wrong, visual right):
        p_a2v = max(0, P_hold(a wrong, v right) - P_train(a wrong, v right)) / P_train(v right)

    visual -> audio should compensate missing (visual wrong, audio right):
        p_v2a = max(0, P_hold(v wrong, a right) - P_train(v wrong, a right)) / P_train(a right)
    """
    eps = 1e-08
    gap_a2v = max(
        0.0,
        float(hold_stats["p_a_wrong_v_right"])
        - float(train_stats["p_a_wrong_v_right"]),
    )
    gap_v2a = max(
        0.0,
        float(hold_stats["p_v_wrong_a_right"])
        - float(train_stats["p_v_wrong_a_right"]),
    )
    denom_a2v = max(float(train_stats["p_v_right"]), eps)
    denom_v2a = max(float(train_stats["p_a_right"]), eps)
    raw_p_a2v = gap_a2v
    raw_p_v2a = gap_v2a
    norm_p_a2v = gap_a2v / denom_a2v
    norm_p_v2a = gap_v2a / denom_v2a
    if getattr(args, "rcc_calib_gap_type", "normalized") == "raw":
        base_p_a2v = raw_p_a2v
        base_p_v2a = raw_p_v2a
    else:
        base_p_a2v = norm_p_a2v
        base_p_v2a = norm_p_v2a
    scale = float(getattr(args, "rcc_calib_scale", 1.0))
    p_a2v = scale * base_p_a2v
    p_v2a = scale * base_p_v2a
    p_a2v = max(float(args.rcc_prob_floor), min(float(args.rcc_prob_max), p_a2v))
    p_v2a = max(float(args.rcc_prob_floor), min(float(args.rcc_prob_max), p_v2a))
    if float(args.rcc_prob_a2v) >= 0 and args.rcc_mode == "manual":
        p_a2v = float(args.rcc_prob_a2v)
    if float(args.rcc_prob_v2a) >= 0 and args.rcc_mode == "manual":
        p_v2a = float(args.rcc_prob_v2a)
    return (p_a2v, p_v2a)


def maybe_update_rcc_calibration(args, model, train_loader, hold_loader, device, epoch):
    if not getattr(args, "use_rcc", False) or args.rcc_mode != "calibrated":
        return None
    if epoch < int(args.rcc_start_epoch):
        return None
    interval = int(args.rcc_calib_interval)
    already = bool(getattr(args, "_rcc_calibrated_once", False))
    if already and (
        interval <= 0 or (epoch - int(args.rcc_start_epoch)) % interval != 0
    ):
        return None
    max_batches = int(getattr(args, "rcc_calib_max_batches", 0))
    train_stats = collect_reliability_config_stats(
        args, model, train_loader, device, max_batches=max_batches
    )
    hold_stats = collect_reliability_config_stats(
        args, model, hold_loader, device, max_batches=max_batches
    )
    p_a2v, p_v2a = estimate_rcc_probabilities_from_gaps(
        args, train_stats, hold_stats
    )
    args._rcc_p_a2v = float(p_a2v)
    args._rcc_p_v2a = float(p_v2a)
    args._rcc_calibrated_once = True
    calibration = {"a2v": float(p_a2v), "v2a": float(p_v2a)}
    save_path = args.rcc_calib_json
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(calibration, f, ensure_ascii=False, indent=2)
    return calibration


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    if scheduler is not None:
        scheduler.step()
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(
        tqdm(dataloader, desc="Epoch {}/{}".format(epoch, args.epochs))
    ):
        spec, image, label = unpack_av_batch(batch)
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        need_features = args.use_rcc or args.use_adoption_guard
        if need_features:
            ret = model(spec.unsqueeze(1).float(), image.float(), return_features=True)
            if isinstance(ret, tuple) and len(ret) == 4:
                out, out_a, out_v, feat_dict = ret
            else:
                out, out_a, out_v = ret
                feat_dict = None
        else:
            out, out_a, out_v = model(spec.unsqueeze(1).float(), image.float())
            feat_dict = None
        logits_f = out
        logits_a = out_a
        logits_v = out_v
        loss_v = criterion(logits_v, label)
        loss_a = criterion(logits_a, label)
        loss_f = criterion(logits_f, label)
        loss_cls = loss_f + args.gamma * (loss_a + loss_v)
        loss = loss_cls
        if (
            getattr(args, "use_cf_anchor", False)
            and epoch >= args.cf_start_epoch
            and (args.lambda_cf > 0)
        ):
            loss_cf = loss_a + loss_v
            loss = loss + args.lambda_cf * loss_cf
        if (
            getattr(args, "use_rcc", False)
            and epoch >= args.rcc_start_epoch
            and (args.lambda_rcc > 0)
        ):
            if (
                feat_dict is None
                or "feat_a" not in feat_dict
                or "feat_v" not in feat_dict
            ):
                raise RuntimeError("RCC requires model(..., return_features=True).")
            loss_rcc = compute_rcc_loss(
                args=args,
                model=model,
                feat_a=feat_dict["feat_a"],
                feat_v=feat_dict["feat_v"],
                labels=label,
                logits_a=logits_a,
                logits_v=logits_v,
            )
            loss = loss + args.lambda_rcc * loss_rcc
        guard_start = get_guard_start_epoch(args)
        guard_lambda = get_guard_lambda(args)
        if (
            getattr(args, "use_adoption_guard", False)
            and epoch >= guard_start
            and (guard_lambda > 0)
        ):
            if (
                feat_dict is None
                or "feat_a" not in feat_dict
                or "feat_v" not in feat_dict
            ):
                raise RuntimeError(
                    "Adoption guard requires model(..., return_features=True)."
                )
            loss_guard = compute_adoption_guard_loss(
                args=args,
                model=model,
                feat_a=feat_dict["feat_a"],
                feat_v=feat_dict["feat_v"],
                labels=label,
                logits_a=logits_a,
                logits_v=logits_v,
            )
            loss = loss + guard_lambda * loss_guard
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def valid(args, model, device, dataloader):
    """Evaluate one full pass and return only the public accuracy summary."""
    labels = []
    predictions_audio = []
    predictions_visual = []
    predictions_fusion = []
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            spec, image, label = unpack_av_batch(batch)
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            out, out_a, out_v = model(spec.unsqueeze(1).float(), image.float())
            labels.append(label.detach().cpu().numpy())
            predictions_audio.append(out_a.argmax(dim=1).detach().cpu().numpy())
            predictions_visual.append(out_v.argmax(dim=1).detach().cpu().numpy())
            predictions_fusion.append(out.argmax(dim=1).detach().cpu().numpy())

    def concatenate(parts):
        return np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.int64)

    return build_evaluation_summary(
        concatenate(labels),
        concatenate(predictions_audio),
        concatenate(predictions_visual),
        concatenate(predictions_fusion),
    )
