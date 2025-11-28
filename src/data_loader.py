import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data_folders(data_root, limit=-1):
    data_path = Path(data_root).resolve()
    valid_items = []

    if not data_path.exists():
        logger.error(f"Data root {data_path} does not exist.")
        return []

    sorted_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()], key=lambda x: x.name)

    for dir_path in sorted_dirs:
        folder_id = dir_path.name

        possible_exts = ['.png', '.jpg', '.jpeg']
        image_path = None
        for ext in possible_exts:
            temp_path = dir_path / f"{folder_id}{ext}"
            if temp_path.exists():
                image_path = temp_path
                break

        gt_path = dir_path / f"{folder_id}.txt"

        if image_path:
            valid_items.append({
                "id": folder_id,
                "image_path": str(image_path),
                "gt_path": str(gt_path) if gt_path.exists() else None,
                "output_path": str(dir_path / f"{folder_id}_det.txt")
            })

    if limit != -1:
        valid_items = valid_items[:limit]

    logger.info(f"Loaded {len(valid_items)} data folders.")
    return valid_items
