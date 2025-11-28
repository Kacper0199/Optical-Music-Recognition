import yaml
import logging
import os
import time
from src.monitor import PerformanceMonitor
from src.data_loader import load_data_folders
from src.pipeline.model_client import get_model_client
from src.evaluation import evaluate_prediction

logger = logging.getLogger(__name__)


def run_pipeline(config_dir):
    config_dir = os.path.abspath(config_dir)

    with open(os.path.join(config_dir, "main_config.yaml")) as f:
        main_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "paths_config.yaml")) as f:
        paths_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "plans_config.yaml")) as f:
        plans_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "models_config.yaml")) as f:
        models_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "prompts.yaml")) as f:
        prompts_cfg = yaml.safe_load(f)

    active_plan_key = main_cfg['active_experiment_plan']
    active_plan = plans_cfg[active_plan_key]

    logger.info(f"Active plan: {active_plan_key}")

    monitor = PerformanceMonitor()
    monitor.start()

    try:
        model_cfg = models_cfg[active_plan['model_key']]
        prompt_text = prompts_cfg['prompts'][active_plan['prompt_key']]['text']

        client = get_model_client(model_cfg)
        data_items = load_data_folders(paths_cfg['data_root'], active_plan['data_limit'])

        total_accuracy_sum = 0.0
        scored_items = 0

        for item in data_items:
            logger.info(f"Processing image ID: {item['id']}")

            start_time = time.time()
            result_text = client.generate(item['image_path'], prompt_text)
            end_time = time.time()

            latency = end_time - start_time
            monitor.log_inference(latency, len(result_text))

            with open(item['output_path'], 'w', encoding='utf-8') as f:
                f.write(result_text)
            logger.info(f"Saved detection to: {item['output_path']}")

            if item['gt_path']:
                with open(item['gt_path'], 'r', encoding='utf-8') as f:
                    gt_text = f.read()

                accuracy, correct_sym, total_sym, errors = evaluate_prediction(gt_text, result_text)

                logger.info(f"METRICS -> Accuracy: {accuracy:.4f} | Correct symbols: {correct_sym}/{total_sym} | Errors: {errors}")

                total_accuracy_sum += accuracy
                scored_items += 1
            else:
                logger.info("No ground truth file found. Skipping evaluation for this item.")

        if scored_items > 0:
            avg_score = total_accuracy_sum / scored_items
            logger.info(f"Experiment finished. Average Accuracy: {avg_score:.4f}")
        else:
            logger.info("Experiment finished. No ground truth found for scoring.")

    finally:
        monitor.stop(paths_cfg['logs_root'])
