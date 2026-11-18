"""Generate attention visualizations for the saved SHV video classifiers.

Attention highlights model token interactions; it is not a causal explanation.
"""

import argparse
import gc
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from train_transformers import DEFAULT_ROOT, MODEL_CONFIGS, read_frames


MODEL_NAMES = ("timesformer", "videomae")
CLASS_NAMES = ("Non_Harassment", "Harassment")
ALPHA = 0.4


def prepare_input(frames, processor, device):
    """Match the deterministic resize and normalization used by predict.py."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    x = v2.Resize((224, 224), antialias=True)(x).float() / 255.0
    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).to(device)


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    model = AutoModelForVideoClassification.from_pretrained(
        checkpoint_path, attn_implementation="eager"
    ).to(device).eval()
    return processor, model


def register_attention_hooks(model, model_name, captured):
    """Capture only the final layer's attention to limit memory use."""
    final_layer = model.base_model.encoder.layer[-1]

    def save_attention(output, key):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError(f"{key} attention did not return attention weights.")
        captured[key] = output[1].detach().float().cpu()

    handles = []
    if model_name == "timesformer":
        def request_attention(*hook_args):
            inputs = hook_args[1]
            if len(inputs) < 2:
                raise RuntimeError("Unexpected TimeSformer attention call signature.")
            return (inputs[0], True, *inputs[2:])

        spatial = final_layer.attention.attention
        temporal = final_layer.temporal_attention.attention
        handles.append(spatial.register_forward_pre_hook(request_attention))
        handles.append(temporal.register_forward_pre_hook(request_attention))
        handles.append(spatial.register_forward_hook(
            lambda *hook_args: save_attention(hook_args[-1], "spatial")
        ))
        handles.append(temporal.register_forward_hook(
            lambda *hook_args: save_attention(hook_args[-1], "temporal")
        ))
    elif model_name == "videomae":
        attention = final_layer.attention.attention
        handles.append(attention.register_forward_hook(
            lambda *hook_args: save_attention(hook_args[-1], "spatial")
        ))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return handles


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Attention contains non-finite values.")
    maximum = float(values.max())
    if maximum <= 0:
        raise RuntimeError("Attention map has no positive values.")
    return values / maximum


def attention_to_frames(captured, model_name, config):
    """Convert architecture-specific token attention into per-frame maps/scores."""
    grid = config.image_size // config.patch_size
    if model_name == "timesformer":
        spatial = captured.get("spatial")
        temporal = captured.get("temporal")
        if spatial is None or temporal is None:
            raise RuntimeError("TimeSformer spatial or temporal attention was not captured.")

        expected_tokens = grid * grid + 1
        if spatial.ndim != 4 or spatial.shape[0] != config.num_frames:
            raise RuntimeError(f"Unexpected TimeSformer spatial attention shape: {tuple(spatial.shape)}")
        if spatial.shape[-2:] != (expected_tokens, expected_tokens):
            raise RuntimeError(f"Unexpected TimeSformer spatial token shape: {tuple(spatial.shape)}")
        if temporal.ndim != 4 or temporal.shape[-2:] != (config.num_frames, config.num_frames):
            raise RuntimeError(f"Unexpected TimeSformer temporal attention shape: {tuple(temporal.shape)}")

        # Spatial CLS-query attention locates patches; temporal key attention
        # estimates which sampled frames receive attention across patch tracks.
        maps = spatial.mean(dim=1)[:, 0, 1:].reshape(config.num_frames, grid, grid)
        frame_scores = temporal.mean(dim=(0, 1, 2))
        frame_scores = frame_scores.numpy()
    else:
        spatial = captured.get("spatial")
        if spatial is None or spatial.ndim != 4:
            shape = None if spatial is None else tuple(spatial.shape)
            raise RuntimeError(f"Unexpected VideoMAE attention tensor shape: {shape}")
        tubelet = config.tubelet_size
        temporal_tokens = config.num_frames // tubelet
        expected_tokens = temporal_tokens * grid * grid
        if spatial.shape[-2:] != (expected_tokens, expected_tokens):
            raise RuntimeError(f"Unexpected VideoMAE token shape: {tuple(spatial.shape)}")

        # VideoMAE has no CLS token. Mean incoming attention per patch token,
        # then group the patch tokens into the model's temporal tubelets.
        token_scores = spatial[0].mean(dim=1).mean(dim=0)
        tubelet_maps = token_scores.reshape(temporal_tokens, grid, grid)
        tubelet_scores = token_scores.reshape(temporal_tokens, grid * grid).sum(dim=1)
        frame_scores = tubelet_scores.repeat_interleave(tubelet)
        maps = tubelet_maps.repeat_interleave(tubelet, dim=0)

    if len(maps) != config.num_frames or len(frame_scores) != config.num_frames:
        raise RuntimeError("Attention frame count does not match model input frame count.")
    scores = np.asarray(frame_scores, dtype=np.float32)
    if not np.isfinite(scores).all() or float(scores.sum()) <= 0:
        raise RuntimeError("Frame attention scores are invalid.")
    scores /= scores.sum()
    return [normalize_map(item.numpy()) for item in maps], scores


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")


def render_explanation(
    model_name, video_path, frames, maps, frame_scores, probabilities, output_dir
):
    predicted = int(probabilities.argmax())
    confidence = float(probabilities.max())
    original_dir = output_dir / "frames"
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    important = np.argsort(frame_scores)[::-1]

    fig, axes = plt.subplots(len(frames), 3, figsize=(12, max(3, len(frames) * 2.5)))
    if len(frames) == 1:
        axes = np.asarray([axes])

    for index, (frame, attention) in enumerate(zip(frames, maps)):
        height, width = frame.shape[:2]
        resized = cv2.resize(attention, (width, height), interpolation=cv2.INTER_CUBIC)
        resized = np.clip(resized, 0.0, 1.0)
        colored = cv2.applyColorMap(np.uint8(resized * 255), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frame, 1.0 - ALPHA, colored, ALPHA, 0)

        original_path = original_dir / f"frame_{index + 1:02d}_original.png"
        heatmap_path = heatmap_dir / f"frame_{index + 1:02d}_attention.png"
        overlay_path = overlay_dir / f"frame_{index + 1:02d}_overlay.png"
        write_image(original_path, frame)
        write_image(heatmap_path, colored)
        write_image(overlay_path, overlay)

        axes[index, 0].imshow(frame)
        axes[index, 0].set_ylabel(f"Frame {index + 1}\nscore={frame_scores[index]:.3f}")
        axes[index, 0].set_title("Original")
        axes[index, 1].imshow(colored)
        axes[index, 1].set_title("Attention heatmap")
        axes[index, 2].imshow(overlay)
        axes[index, 2].set_title("Overlay")
        for axis in axes[index]:
            axis.axis("off")

    fig.suptitle(
        f"{model_name} | {CLASS_NAMES[predicted]} ({confidence:.1%}) | "
        "Attention visualization, not a causal explanation",
        fontsize=12,
    )
    fig.tight_layout()
    montage_path = output_dir / "visualization.png"
    fig.savefig(montage_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "model": model_name,
        "video": str(video_path),
        "predicted_class": CLASS_NAMES[predicted],
        "confidence": confidence,
        "class_probabilities": {
            CLASS_NAMES[index]: float(value) for index, value in enumerate(probabilities)
        },
        "frame_attention_scores": [float(value) for value in frame_scores],
        "important_frames": [
            {"frame": int(index + 1), "score": float(frame_scores[index])}
            for index in important[: min(3, len(important))]
        ],
        "attention_method": (
            "Final-layer TimeSformer spatial CLS-to-patch attention with final-layer "
            "temporal key attention for frame scores."
            if model_name == "timesformer"
            else "Final-layer VideoMAE eager attention, averaged over heads and query tokens; "
                 "patch attention grouped by temporal tubelet."
        ),
        "interpretation_note": (
            "Attention highlights token interactions used by the model. It does not prove "
            "that a highlighted frame or region caused the prediction."
        ),
        "visualization": str(montage_path),
    }


def explain_video(model_name, video_path, processor, model, output_dir, device):
    frame_count = MODEL_CONFIGS[model_name]["num_frames"]
    frames = read_frames(str(video_path), frame_count)
    captured = {}
    handles = register_attention_hooks(model, model_name, captured)
    try:
        inputs = prepare_input(frames, processor, device)
        if tuple(inputs.shape) != (1, frame_count, 3, 224, 224):
            raise RuntimeError(f"Unexpected model input shape: {tuple(inputs.shape)}")
        with torch.inference_mode():
            logits = model(pixel_values=inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].float().cpu().numpy()
        if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"Unexpected prediction output: {probabilities}")
        maps, frame_scores = attention_to_frames(captured, model_name, model.config)
        result = render_explanation(
            model_name, video_path, frames, maps, frame_scores, probabilities, output_dir,
        )
        summary_path = output_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result, frames, maps, frame_scores
    finally:
        for handle in handles:
            handle.remove()


def save_comparison(video_path, results, output_dirs, output_dir):
    if len(results) != 2:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, model_name in enumerate(MODEL_NAMES):
        result = results[model_name]
        for column, frame_item in enumerate(result["important_frames"]):
            image_path = (
                output_dirs[model_name] / "overlays"
                / f"frame_{frame_item['frame']:02d}_overlay.png"
            )
            overlay = cv2.imread(str(image_path))
            if overlay is None:
                raise OSError(f"Could not read attention overlay: {image_path}")
            axes[row, column].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[row, column].set_title(
                f"Frame {frame_item['frame']} | score={frame_item['score']:.3f}"
            )
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(
            f"{model_name}\n{result['predicted_class']} ({result['confidence']:.1%})"
        )
    fig.suptitle(f"{video_path.name} | Top attended frames (attention is not causal)")
    fig.tight_layout()
    path = output_dir / f"{video_path.stem}_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def find_test_videos(data_root):
    videos = []
    for class_name in ("Harassment", "Non_Harassment"):
        folder = data_root / "test" / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Test class directory not found: {folder}")
        videos.extend(sorted(folder.glob("*.mp4")))
    if not videos:
        raise FileNotFoundError(f"No .mp4 test videos found under: {data_root / 'test'}")
    return videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to one video clip.")
    source.add_argument("--all-test", action="store_true", help="Explain all test-set clips.")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--model", choices=(*MODEL_NAMES, "both"), default="both")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Override the checkpoint path for a single selected model.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explainability"))
    parser.add_argument("--no-display", action="store_true", help="Save files without opening plots.")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    output_root = args.output_dir if args.output_dir.is_absolute() else project_dir / args.output_dir
    video_paths = find_test_videos(args.data_root) if args.all_test else [args.video]
    for path in video_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

    models_to_run = MODEL_NAMES if args.model == "both" else (args.model,)
    if args.checkpoint and len(models_to_run) != 1:
        parser.error("--checkpoint can only be used with --model timesformer or --model videomae.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Models: {', '.join(models_to_run)}")
    print("Attention visualizations are not causal explanations.")

    all_results = {path: {} for path in video_paths}
    output_dirs = {path: {} for path in video_paths}
    for model_name in models_to_run:
        default_checkpoint = project_dir / "outputs" / model_name / "best_model"
        checkpoint = args.checkpoint or default_checkpoint
        processor, model = load_model(checkpoint, device)
        try:
            for video_path in video_paths:
                model_output = output_root / model_name / video_path.stem
                result, _, _, _ = explain_video(
                    model_name, video_path, processor, model, model_output, device
                )
                all_results[video_path][model_name] = result
                output_dirs[video_path][model_name] = model_output
                print(
                    f"{video_path.name} | {model_name}: {result['predicted_class']} "
                    f"({result['confidence']:.1%}) | saved {result['visualization']}"
                )
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for video_path in video_paths:
        results = all_results[video_path]
        if len(results) == 2:
            comparison = save_comparison(
                video_path, results, output_dirs[video_path], output_root / "comparisons"
            )
            print(f"Comparison saved: {comparison}")
        if not args.no_display:
            for result in results.values():
                image = plt.imread(result["visualization"])
                plt.figure(figsize=(12, 8))
                plt.imshow(image)
                plt.axis("off")
                plt.show()


if __name__ == "__main__":
    main()
