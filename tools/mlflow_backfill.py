#!/usr/bin/env python3
"""Backfill existing Hydra run outputs into MLflow and optionally watch for new runs."""
from __future__ import annotations

import argparse
import json
import os
import time
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from functools import lru_cache

from omegaconf import OmegaConf
import mlflow
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
WANDB_ROOT = ROOT / "wandb"
DEFAULT_HOST_MINIO_ENDPOINT = "http://127.0.0.1:17243"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mlflow_backfill.log')
    ]
)
logger = logging.getLogger(__name__)


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts/lists for MLflow parameters."""
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = ",".join(map(str, value))
        else:
            flat[full_key] = value
    return flat


def log_artifact_if_exists(path: Path, retry_count: int = 3, artifact_path: Optional[str] = None) -> bool:
    """Upload artifact if present, with retry mechanism and detailed error logging."""
    if not path.exists():
        logger.debug(f"Artifact {path} does not exist, skipping")
        return False

    # Directory upload path
    if path.is_dir():
        try:
            if not any(path.iterdir()):
                logger.warning(f"Artifact directory {path} is empty, skipping")
                return False
        except OSError as exc:
            logger.error(f"Cannot access artifact directory {path}: {exc}")
            return False
        target_artifact_path = artifact_path or path.name
    else:
        # Validate file before upload
        try:
            file_size = path.stat().st_size
            if file_size == 0:
                logger.warning(f"Artifact {path} is empty (0 bytes), skipping")
                return False
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"Artifact {path} is too large ({file_size} bytes), skipping")
                return False
            logger.debug(f"Artifact {path} size: {file_size} bytes")

            # Check file permissions
            if not os.access(path, os.R_OK):
                logger.error(f"Artifact {path} is not readable, skipping")
                return False

        except OSError as exc:
            logger.error(f"Cannot access artifact {path}: {exc}")
            return False
    
    for attempt in range(retry_count):
        try:
            if path.is_dir():
                logger.info(f"Uploading artifact directory: {path} (attempt {attempt + 1}/{retry_count})")
                mlflow.log_artifacts(str(path), artifact_path=target_artifact_path)
                logger.info(f"Successfully uploaded artifact directory: {path}")
            else:
                logger.info(f"Uploading artifact: {path} (attempt {attempt + 1}/{retry_count})")
                mlflow.log_artifact(str(path))
                logger.info(f"Successfully uploaded artifact: {path}")
            return True
        except MlflowException as exc:
            error_msg = str(exc)
            logger.warning(f"Failed to upload artifact {path} (attempt {attempt + 1}): {error_msg}")
            
            # Check for specific error types
            if "500" in error_msg or "Internal Server Error" in error_msg:
                logger.error(f"MLflow server error (500) for {path}. This may indicate server-side issues.")
            elif "403" in error_msg or "Forbidden" in error_msg:
                logger.error(f"Permission denied for {path}. Check MLflow server permissions.")
            elif "413" in error_msg or "Request Entity Too Large" in error_msg:
                logger.error(f"File too large for {path}. Consider compressing or splitting the file.")
            elif "Connection" in error_msg or "timeout" in error_msg.lower():
                logger.error(f"Connection issue for {path}. Check network connectivity and MLflow server status.")
            elif "S3" in error_msg or "bucket" in error_msg.lower():
                logger.error(f"S3/Minio issue for {path}. Check S3 credentials and bucket permissions.")
            
            if attempt < retry_count - 1:
                backoff_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                logger.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
            else:
                logger.error(f"Failed to upload artifact {path} after {retry_count} attempts: {exc}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                # Enhanced diagnostics
                try:
                    import requests
                    tracking_uri = mlflow.get_tracking_uri()
                    logger.info(f"MLflow tracking URI: {tracking_uri}")
                    
                    # Check MLflow server health
                    response = requests.get(f"{tracking_uri}/health", timeout=5)
                    logger.info(f"MLflow server health check: {response.status_code}")
                    
                    # Check S3/Minio connection if configured
                    s3_endpoint = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
                    if s3_endpoint:
                        logger.info(f"S3 endpoint: {s3_endpoint}")
                        try:
                            import boto3
                            s3_client = boto3.client(
                                's3',
                                endpoint_url=s3_endpoint,
                                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
                            )
                            buckets = s3_client.list_buckets()
                            logger.info(f"S3 connection test: Found {len(buckets['Buckets'])} buckets")
                        except Exception as s3_exc:
                            logger.error(f"S3 connection test failed: {s3_exc}")
                    
                except Exception as health_exc:
                    logger.error(f"Could not check MLflow server health: {health_exc}")
        except Exception as exc:
            logger.error(f"Unexpected error uploading artifact {path}: {exc}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            break
    
    return False


@lru_cache(maxsize=None)
def find_wandb_run_dir(run_id: str) -> Optional[Path]:
    """Locate the W&B run directory matching a given run id."""
    if not run_id or not WANDB_ROOT.exists():
        return None
    matches = sorted(WANDB_ROOT.glob(f"run-*-{run_id}"))
    return matches[0] if matches else None


@lru_cache(maxsize=None)
def load_wandb_metadata(run_id: str) -> Dict[str, Any]:
    """Return metadata for a W&B run, or an empty dict if unavailable."""
    run_dir = find_wandb_run_dir(run_id)
    if run_dir is None:
        return {}
    metadata_file = run_dir / "files" / "wandb-metadata.json"
    if not metadata_file.exists():
        return {}
    try:
        return json.loads(metadata_file.read_text())
    except json.JSONDecodeError as exc:
        print(f"[WARN] Could not parse W&B metadata for {run_id}: {exc}")
        return {}


def extract_wandb_project(run_id: str) -> Optional[str]:
    """Derive the W&B project name from metadata/args."""
    metadata = load_wandb_metadata(run_id)
    args = metadata.get("args", [])
    for arg in args:
        if isinstance(arg, str) and arg.startswith("wandb.project="):
            return arg.split("=", 1)[1]
    # Fallback to config.yaml if args are missing
    run_dir = find_wandb_run_dir(run_id)
    if run_dir is None:
        return None
    config_file = run_dir / "files" / "config.yaml"
    if not config_file.exists():
        return None
    try:
        import yaml

        config = yaml.safe_load(config_file.read_text())
        project = config.get("wandb", {}).get("value", {}).get("project")
        return project
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to extract W&B project for {run_id}: {exc}")
        return None


def extract_wandb_run_name(run_id: str) -> Optional[str]:
    """Grab the configured W&B run name if one exists."""
    run_dir = find_wandb_run_dir(run_id)
    if run_dir is None:
        return None
    config_file = run_dir / "files" / "config.yaml"
    if not config_file.exists():
        return None
    try:
        import yaml

        config = yaml.safe_load(config_file.read_text())
        return config.get("wandb", {}).get("value", {}).get("run_name")
    except Exception:
        return None


def load_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE env file."""
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def check_mlflow_server_health(tracking_uri: str) -> bool:
    """Check if MLflow server is healthy and accessible."""
    try:
        import requests
        import urllib.parse
        
        # Parse the tracking URI
        parsed_uri = urllib.parse.urlparse(tracking_uri)
        base_url = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
        
        # Try health endpoint first
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("MLflow server health check: OK")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Fallback to API endpoint
        try:
            response = requests.get(f"{base_url}/api/2.0/mlflow/experiments/search", timeout=10)
            if response.status_code == 200:
                logger.info("MLflow server API check: OK")
                return True
            else:
                logger.warning(f"MLflow server API returned status {response.status_code}")
        except requests.exceptions.RequestException as exc:
            logger.error(f"MLflow server API check failed: {exc}")
            
        return False
    except Exception as exc:
        logger.error(f"Error checking MLflow server health: {exc}")
        return False


def check_artifact_storage_config() -> bool:
    """Check if artifact storage is properly configured."""
    try:
        # Check if S3 endpoint is configured
        s3_endpoint = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
        if not s3_endpoint:
            logger.warning("MLFLOW_S3_ENDPOINT_URL not set - artifacts may not be stored properly")
            return False
        
        # Check AWS credentials
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if not aws_access_key or not aws_secret_key:
            logger.warning("AWS credentials not set - artifact storage may fail")
            return False
        
        logger.info(f"Artifact storage configured: S3 endpoint={s3_endpoint}")
        return True
    except Exception as exc:
        logger.error(f"Error checking artifact storage config: {exc}")
        return False


def check_existing_run(experiment_name: str, run_name: str) -> bool:
    """Check if a run with the same name already exists in the experiment."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.debug(f"Experiment {experiment_name} does not exist yet")
            return False
        
        # Search for runs with the same name in the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"run_name = '{run_name}'"
        )
        
        if not runs.empty:
            logger.info(f"Run '{run_name}' already exists in experiment '{experiment_name}'")
            return True
        
        logger.debug(f"Run '{run_name}' does not exist in experiment '{experiment_name}'")
        return False
        
    except Exception as exc:
        logger.error(f"Error checking existing run: {exc}")
        return False


def generate_unique_run_name(base_name: str, experiment_name: str) -> str:
    """Generate a unique run name by appending a counter if needed."""
    run_name = base_name
    counter = 1
    
    while check_existing_run(experiment_name, run_name):
        run_name = f"{base_name}_{counter}"
        counter += 1
        if counter > 1000:  # Safety limit
            logger.warning(f"Could not generate unique run name for {base_name}")
            break
    
    if counter > 1:
        logger.info(f"Generated unique run name: {run_name} (original: {base_name})")
    
    return run_name


def discover_artifacts_in_run(run_dir: Path) -> list:
    """Discover all potential artifacts in a run directory."""
    discovered_artifacts = []
    
    # 標準アーティファクトの候補
    standard_artifacts = [
        "cebra_model.pt",
        "cebra_embeddings.npy", 
        "confusion_matrix.png",
        "cebra_interactive_discrete.html",
        "cebra_interactive_discrete.svg",
        "static_PCA_plot.png",
        "static_UMAP_plot.png",
        "None.html",
        "main.log",
    ]
    
    # 標準アーティファクトをチェック
    for artifact_name in standard_artifacts:
        artifact_path = run_dir / artifact_name
        if artifact_path.exists():
            try:
                file_size = artifact_path.stat().st_size
                if file_size > 0 and os.access(artifact_path, os.R_OK):
                    discovered_artifacts.append(artifact_name)
            except OSError:
                pass
    
    # その他のファイルを探索
    for file_path in run_dir.iterdir():
        if file_path.is_file() and file_path.name not in standard_artifacts:
            # 特定の拡張子のファイルをアーティファクトとして追加
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.html', '.json', '.csv', '.txt', '.log', '.pt', '.pkl', '.npy', '.npz', '.svg']:
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 0 and os.access(file_path, os.R_OK):
                        discovered_artifacts.append(file_path.name)
                except OSError:
                    pass
    
    return discovered_artifacts


def validate_artifacts_before_upload(run_dir: Path, artifact_names: list, debug: bool = False) -> tuple[list, list]:
    """Validate artifacts before upload and return valid and invalid artifacts."""
    valid_artifacts = []
    invalid_artifacts = []
    
    logger.info(f"Validating {len(artifact_names)} artifacts in {run_dir}")
    
    for artifact_name in artifact_names:
        artifact_path = run_dir / artifact_name
        
        if not artifact_path.exists():
            logger.warning(f"Artifact {artifact_name} does not exist at {artifact_path}")
            invalid_artifacts.append(artifact_name)
            continue
            
        try:
            file_size = artifact_path.stat().st_size
            if file_size == 0:
                logger.warning(f"Artifact {artifact_name} is empty (0 bytes) at {artifact_path}")
                invalid_artifacts.append(artifact_name)
                continue
                
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"Artifact {artifact_name} is too large ({file_size} bytes) at {artifact_path}")
                invalid_artifacts.append(artifact_name)
                continue
                
            if not os.access(artifact_path, os.R_OK):
                logger.error(f"Artifact {artifact_name} is not readable at {artifact_path}")
                invalid_artifacts.append(artifact_name)
                continue
                
            valid_artifacts.append(artifact_name)
            logger.info(f"Artifact {artifact_name} is valid (size: {file_size} bytes)")
            
            if debug:
                # 詳細なファイル情報を表示
                stat_info = artifact_path.stat()
                logger.debug(f"  - Path: {artifact_path}")
                logger.debug(f"  - Size: {stat_info.st_size} bytes")
                logger.debug(f"  - Modified: {stat_info.st_mtime}")
                logger.debug(f"  - Permissions: {oct(stat_info.st_mode)}")
            
        except OSError as exc:
            logger.error(f"Cannot access artifact {artifact_name} at {artifact_path}: {exc}")
            invalid_artifacts.append(artifact_name)
    
    logger.info(f"Validation complete: {len(valid_artifacts)} valid, {len(invalid_artifacts)} invalid")
    return valid_artifacts, invalid_artifacts


def list_artifacts_in_runs(include_datasets: Optional[Iterable[str]], limit: Optional[int] = None) -> None:
    """List all artifacts in run directories for debugging purposes."""
    logger.info("Listing artifacts in run directories...")
    
    total_runs = 0
    total_artifacts = 0
    artifact_stats = {}
    
    for dataset_dir in iter_dataset_dirs(include_datasets):
        logger.info(f"Checking dataset: {dataset_dir}")
        run_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        
        for run_dir in run_dirs:
            total_runs += 1
            logger.info(f"\nRun: {run_dir}")
            
            # 標準アーティファクトのチェック
            artifact_names = [
                "cebra_model.pt",
                "cebra_embeddings.npy", 
                "confusion_matrix.png",
                "cebra_interactive_discrete.html",
                "static_PCA_plot.png",
                "static_UMAP_plot.png",
                "None.html",
                "main.log",
            ]
            
            for artifact_name in artifact_names:
                artifact_path = run_dir / artifact_name
                if artifact_path.exists():
                    try:
                        file_size = artifact_path.stat().st_size
                        readable = os.access(artifact_path, os.R_OK)
                        status = "OK" if readable and file_size > 0 else "INVALID"
                        logger.info(f"  {artifact_name}: {status} ({file_size} bytes)")
                        if status == "OK":
                            total_artifacts += 1
                            artifact_stats[artifact_name] = artifact_stats.get(artifact_name, 0) + 1
                    except OSError as exc:
                        logger.info(f"  {artifact_name}: ERROR ({exc})")
                else:
                    logger.info(f"  {artifact_name}: NOT FOUND")
            
            # その他のファイルをチェック
            other_files = [f for f in run_dir.iterdir() if f.is_file() and f.name not in artifact_names]
            if other_files:
                logger.info(f"  Other files: {[f.name for f in other_files]}")
                for other_file in other_files:
                    try:
                        file_size = other_file.stat().st_size
                        logger.info(f"    {other_file.name}: {file_size} bytes")
                    except OSError:
                        pass
            
            if limit and total_runs >= limit:
                break
        
        if limit and total_runs >= limit:
            break
    
    logger.info(f"\nSummary: {total_runs} runs checked, {total_artifacts} valid artifacts found")
    if artifact_stats:
        logger.info("Artifact statistics:")
        for artifact_name, count in sorted(artifact_stats.items()):
            logger.info(f"  {artifact_name}: {count} runs")


def ingest_run(run_dir: Path, skip_artifacts: bool = False, skip_duplicates: bool = True, debug: bool = False) -> bool:
    """Log a single Hydra run directory into MLflow with enhanced error handling."""
    logger.info(f"Processing run directory: {run_dir}")
    
    try:
        hydra_cfg = run_dir / ".hydra" / "config.yaml"
        if not hydra_cfg.exists():
            logger.warning(f"No Hydra config found in {run_dir}, skipping")
            return False

        logger.debug(f"Loading Hydra config from {hydra_cfg}")
        cfg = OmegaConf.load(hydra_cfg)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        dataset_name = cfg_dict["dataset"]["name"]
        base_run_name = run_dir.name
        wandb_file = run_dir / "wandb_run_id.txt"
        wandb_run_id = wandb_file.read_text().strip() if wandb_file.exists() else None
        wandb_project = extract_wandb_project(wandb_run_id) if wandb_run_id else None
        wandb_run_name = extract_wandb_run_name(wandb_run_id) if wandb_run_id else None

        # Use W&B project as the MLflow experiment when available
        if wandb_project:
            experiment_name = wandb_project
        elif dataset_name:
            experiment_name = f"CEBRA/{dataset_name}"
        else:
            experiment_name = "CEBRA/default"

        logger.info(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        
        # 重複チェック
        if skip_duplicates and check_existing_run(experiment_name, base_run_name):
            logger.info(f"Run '{base_run_name}' already exists in experiment '{experiment_name}', skipping")
            return True
        
        # ユニークなrun名を生成
        run_name = generate_unique_run_name(base_run_name, experiment_name)
        
        with mlflow.start_run(
            run_name=run_name,
            description=str(run_dir.relative_to(ROOT)),
        ):
            logger.info(f"Started MLflow run: {run_name}")
            
            # タグの設定
            if wandb_run_id:
                mlflow.set_tag("wandb_run_id", wandb_run_id)
                logger.debug(f"Set wandb_run_id tag: {wandb_run_id}")
            if wandb_project:
                mlflow.set_tag("wandb_project", wandb_project)
                logger.debug(f"Set wandb_project tag: {wandb_project}")
            if wandb_run_name:
                mlflow.set_tag("wandb_run_name", wandb_run_name)
                logger.debug(f"Set wandb_run_name tag: {wandb_run_name}")
            
            # パラメータのログ
            logger.info("Logging parameters...")
            mlflow.log_params(flatten_dict(cfg_dict["embedding"], "embedding"))
            mlflow.log_params(flatten_dict(cfg_dict["cebra"], "cebra"))
            mlflow.log_params(flatten_dict(cfg_dict["evaluation"], "evaluation"))
            mlflow.set_tag("dataset", dataset_name)
            logger.info("Parameters logged successfully")

            # 分類レポートの処理
            cls_report = run_dir / "classification_report.json"
            if cls_report.exists():
                logger.info("Processing classification report...")
                try:
                    report = json.loads(cls_report.read_text())
                    macro = report.get("macro avg", {})
                    accuracy = report.get("accuracy")
                    if macro:
                        mlflow.log_metrics(
                            {
                                "macro_precision": macro.get("precision", 0.0),
                                "macro_recall": macro.get("recall", 0.0),
                                "macro_f1": macro.get("f1-score", 0.0),
                            }
                        )
                    if accuracy is not None:
                        mlflow.log_metric("accuracy", accuracy)
                    
                    # Only upload artifact if not skipping artifacts
                    if not skip_artifacts:
                        log_artifact_if_exists(cls_report)
                    else:
                        logger.debug("Skipping classification_report.json artifact upload (--skip-artifacts flag)")
                    
                    logger.info("Classification report processed successfully")
                except Exception as exc:
                    logger.error(f"Error processing classification report: {exc}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

            # 回帰レポートの処理
            reg_report = run_dir / "regression_report.json"
            if reg_report.exists():
                logger.info("Processing regression report...")
                try:
                    metrics = json.loads(reg_report.read_text())
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)
                    
                    # Only upload artifact if not skipping artifacts
                    if not skip_artifacts:
                        log_artifact_if_exists(reg_report)
                    else:
                        logger.debug("Skipping regression_report.json artifact upload (--skip-artifacts flag)")
                    
                    logger.info("Regression report processed successfully")
                except Exception as exc:
                    logger.error(f"Error processing regression report: {exc}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

            # アーティファクトのアップロード
            if skip_artifacts:
                logger.info("Skipping artifact uploads (--skip-artifacts flag set)")
                mlflow.set_tag("artifact_upload_status", "skipped")
            else:
                logger.info("Discovering and uploading artifacts...")
                
                # 動的にアーティファクトを発見
                discovered_artifacts = discover_artifacts_in_run(run_dir)
                logger.info(f"Discovered {len(discovered_artifacts)} potential artifacts: {discovered_artifacts}")
                
                # 事前検証
                valid_artifacts, invalid_artifacts = validate_artifacts_before_upload(run_dir, discovered_artifacts, debug)
                
                if invalid_artifacts:
                    logger.warning(f"Found {len(invalid_artifacts)} invalid artifacts: {invalid_artifacts}")
                    mlflow.set_tag("invalid_artifacts", ",".join(invalid_artifacts))
                
                uploaded_count = 0
                failed_artifacts = []
                
                # 有効なアーティファクトのみアップロード
                for artifact_name in valid_artifacts:
                    artifact_path = run_dir / artifact_name
                    if log_artifact_if_exists(artifact_path):
                        uploaded_count += 1
                    else:
                        failed_artifacts.append(artifact_name)
                
                # 設定ファイルのアップロード
                config_artifacts = [
                    ("hydra_config.yaml", hydra_cfg),
                    ("overrides.yaml", run_dir / ".hydra" / "overrides.yaml")
                ]
                
                for config_name, config_path in config_artifacts:
                    if log_artifact_if_exists(config_path):
                        uploaded_count += 1
                    else:
                        failed_artifacts.append(config_name)

                # Visualization directories (if present) should also become artifacts
                visualization_dirs = [
                    run_dir / "visualization",
                    run_dir / "visualizations",
                    run_dir / "artifacts" / "visualization",
                    run_dir / "artifacts" / "visualizations",
                ]
                uploaded_visualizations = 0
                seen_visualization_paths = set()
                for viz_dir in visualization_dirs:
                    if not viz_dir.exists() or not viz_dir.is_dir():
                        continue
                    resolved_path = str(viz_dir.resolve())
                    if resolved_path in seen_visualization_paths:
                        continue
                    seen_visualization_paths.add(resolved_path)
                    if log_artifact_if_exists(viz_dir, artifact_path="visualization"):
                        uploaded_count += 1
                        uploaded_visualizations += 1
                        logger.info(f"Visualization directory uploaded from {viz_dir}")
                    else:
                        failed_artifacts.append(f"{viz_dir.name}/")
                        logger.warning(f"Failed to upload visualization directory from {viz_dir}")
                if uploaded_visualizations:
                    mlflow.set_tag("visualization_artifact_uploaded", str(uploaded_visualizations))
                
                total_artifacts = len(valid_artifacts) + len(invalid_artifacts)
                logger.info(f"Artifact upload summary: {uploaded_count}/{len(valid_artifacts)} valid artifacts uploaded successfully")
                logger.info(f"Total artifacts discovered: {len(discovered_artifacts)} (valid: {len(valid_artifacts)}, invalid: {len(invalid_artifacts)})")
                
                if failed_artifacts:
                    logger.warning(f"Failed to upload {len(failed_artifacts)} valid artifacts: {failed_artifacts}")
                    mlflow.set_tag("failed_artifacts", ",".join(failed_artifacts))
                
                if invalid_artifacts:
                    logger.info(f"Skipped {len(invalid_artifacts)} invalid artifacts: {invalid_artifacts}")
                
                # アーティファクトの詳細統計をMLflowに記録
                mlflow.set_tag("artifacts_discovered", str(len(discovered_artifacts)))
                mlflow.set_tag("artifacts_valid", str(len(valid_artifacts)))
                mlflow.set_tag("artifacts_invalid", str(len(invalid_artifacts)))
                mlflow.set_tag("artifacts_uploaded", str(uploaded_count))
                mlflow.set_tag("artifact_upload_status", f"{uploaded_count} successful, {len(failed_artifacts)} failed, {len(invalid_artifacts)} invalid")
            
        logger.info(f"Successfully processed run: {run_dir}")
        return True
        
    except Exception as exc:
        logger.error(f"Error processing run {run_dir}: {exc}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def iter_dataset_dirs(include: Optional[Iterable[str]]) -> Iterable[Path]:
    """Yield dataset directories, optionally filtering by a provided list."""
    if not RESULTS_ROOT.exists():
        return []
    include_set = {name.strip() for name in include} if include else None
    for dataset_dir in sorted(RESULTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if include_set is not None and dataset_dir.name not in include_set:
            continue
        yield dataset_dir


def backfill_runs(
    include_datasets: Optional[Iterable[str]],
    limit: Optional[int],
    resume_from: Optional[str],
    skip_artifacts: bool = False,
    skip_duplicates: bool = True,
    debug: bool = False,
) -> None:
    """Log existing runs beneath the results directory with enhanced error tracking."""
    logger.info(f"Starting backfill process with datasets: {include_datasets}, limit: {limit}, resume_from: {resume_from}")
    
    processed = 0
    successful = 0
    failed = 0
    resume_reached = resume_from is None
    
    try:
        for dataset_dir in iter_dataset_dirs(include_datasets):
            logger.info(f"Processing dataset directory: {dataset_dir}")
            run_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
            logger.info(f"Found {len(run_dirs)} run directories in {dataset_dir}")
            
            for run_dir in run_dirs:
                if not resume_reached:
                    if run_dir.name == resume_from:
                        resume_reached = True
                        logger.info(f"Resuming from run: {run_dir.name}")
                    else:
                        continue
                
                logger.info(f"Processing run {processed + 1}: {run_dir}")
                if ingest_run(run_dir, skip_artifacts, skip_duplicates, debug):
                    successful += 1
                    logger.info(f"Successfully processed run: {run_dir}")
                else:
                    failed += 1
                    logger.error(f"Failed to process run: {run_dir}")
                
                processed += 1
                if limit is not None and processed >= limit:
                    logger.info(f"Reached limit of {limit} runs")
                    break
            
            if limit is not None and processed >= limit:
                break
                
    except Exception as exc:
        logger.error(f"Error during backfill process: {exc}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
    
    logger.info(f"Backfill completed. Processed: {processed}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logger.warning(f"{failed} runs failed to process. Check the log file for details.")


def watch_runs(poll_seconds: int, skip_artifacts: bool = False, skip_duplicates: bool = True, debug: bool = False) -> None:
    """Poll for newly created run directories and log them with enhanced monitoring."""
    logger.info(f"Starting watch mode with {poll_seconds}s polling interval")
    
    seen = {
        run.resolve()
        for dataset_dir in RESULTS_ROOT.iterdir()
        if dataset_dir.is_dir()
        for run in dataset_dir.iterdir()
        if run.is_dir()
    }
    logger.info(f"Found {len(seen)} existing run directories")
    
    processed_count = 0
    while True:
        try:
            new_runs_found = 0
            for dataset_dir in RESULTS_ROOT.iterdir():
                if not dataset_dir.is_dir():
                    continue
                for run_dir in dataset_dir.iterdir():
                    if run_dir.is_dir():
                        resolved = run_dir.resolve()
                        if resolved not in seen:
                            logger.info(f"New run detected: {run_dir}")
                            if ingest_run(run_dir, skip_artifacts, skip_duplicates, debug):
                                processed_count += 1
                                logger.info(f"Successfully processed new run: {run_dir} (total: {processed_count})")
                            else:
                                logger.error(f"Failed to process new run: {run_dir}")
                            seen.add(resolved)
                            new_runs_found += 1
            
            if new_runs_found > 0:
                logger.info(f"Processed {new_runs_found} new runs in this cycle")
            else:
                logger.debug("No new runs detected in this cycle")
                
        except Exception as exc:
            logger.error(f"Error during watch cycle: {exc}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Hydra results into MLflow")
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:54085",
        help="MLflow tracking URI (defaults to local docker compose port)",
    )
    parser.add_argument("--watch", action="store_true", help="Keep polling for new runs")
    parser.add_argument(
        "--poll-seconds", type=int, default=30, help="Polling interval for --watch"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Limit ingestion to specific dataset names (can be provided multiple times)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of run directories to ingest (per invocation)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume ingestion from a specific run directory name (within the selected datasets)",
    )
    parser.add_argument(
        "--minio-env",
        default="/srv/mlstack/.env",
        help=(
            "Optional path to the stack .env file; credentials will be loaded if environment vars "
            "are missing (default: /srv/mlstack/.env)"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed troubleshooting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the import process without actually uploading to MLflow",
    )
    parser.add_argument(
        "--verify-minio",
        action="store_true",
        help="Verify Minio connection and credentials before starting",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip artifact uploads entirely (useful when artifacts are causing 500 errors)",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        default=True,
        help="Skip runs that already exist in MLflow (default: True)",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate runs to be created (overrides --skip-duplicates)",
    )
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="Validate artifacts before upload and show detailed diagnostics",
    )
    parser.add_argument(
        "--list-artifacts",
        action="store_true",
        help="List all artifacts in run directories without uploading",
    )
    args = parser.parse_args()

    # デバッグログの設定
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # ログレベルの設定
    logger.info(f"Starting MLflow backfill with arguments: {vars(args)}")

    # Minio環境設定の処理
    if args.minio_env:
        env_path = Path(args.minio_env).expanduser()
        if env_path.exists():
            logger.info(f"Loading environment from {env_path}")
            env_vars = load_env_file(env_path)
            for key, value in env_vars.items():
                if key == "MINIO_ROOT_USER":
                    os.environ.setdefault("AWS_ACCESS_KEY_ID", value)
                    logger.debug(f"Set AWS_ACCESS_KEY_ID from {key}")
                elif key == "MINIO_ROOT_PASSWORD":
                    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", value)
                    logger.debug(f"Set AWS_SECRET_ACCESS_KEY from {key}")
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", DEFAULT_HOST_MINIO_ENDPOINT)
            logger.info("Minio environment variables loaded successfully")
        else:
            logger.warning(f"Requested env file {env_path} was not found; relying on existing env vars.")
    else:
        logger.info("No minio-env specified, using existing environment variables")
    
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", DEFAULT_HOST_MINIO_ENDPOINT)
    logger.info(f"MLflow S3 endpoint set to: {os.environ.get('MLFLOW_S3_ENDPOINT_URL')}")

    # Minio接続の検証
    if args.verify_minio:
        logger.info("Verifying Minio connection...")
        try:
            import boto3
            s3_client = boto3.client(
                's3',
                endpoint_url=os.environ.get('MLFLOW_S3_ENDPOINT_URL'),
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
            )
            # バケット一覧を取得して接続をテスト
            response = s3_client.list_buckets()
            logger.info(f"Minio connection successful. Found {len(response['Buckets'])} buckets")
        except Exception as exc:
            logger.error(f"Minio connection failed: {exc}")
            logger.error("Please check your Minio credentials and endpoint")
            return

    # MLflow設定
    logger.info(f"Setting MLflow tracking URI to: {args.tracking_uri}")
    mlflow.set_tracking_uri(args.tracking_uri)
    
    # MLflow server health check
    logger.info("Checking MLflow server health...")
    if not check_mlflow_server_health(args.tracking_uri):
        logger.error("MLflow server health check failed. Please verify the server is running and accessible.")
        return
    
    # Artifact storage configuration check
    logger.info("Checking artifact storage configuration...")
    if not check_artifact_storage_config():
        logger.warning("Artifact storage configuration issues detected. Artifact uploads may fail.")
    
    # 重複スキップの設定
    skip_duplicates = args.skip_duplicates and not args.allow_duplicates
    if skip_duplicates:
        logger.info("Duplicate runs will be skipped")
    else:
        logger.info("Duplicate runs will be allowed (may create multiple runs with same name)")

    # アーティファクトリストモードの処理
    if args.list_artifacts:
        logger.info("LIST ARTIFACTS MODE: Listing artifacts without uploading")
        list_artifacts_in_runs(args.datasets, args.limit)
        return

    # ドライランモードの処理
    if args.dry_run:
        logger.info("DRY RUN MODE: No actual uploads will be performed")
        # ドライランモードでは実際のアップロードをスキップ
        return

    # バックフィル処理の実行
    try:
        backfill_runs(args.datasets, args.limit, args.resume_from, args.skip_artifacts, skip_duplicates, args.debug)
        if args.watch:
            logger.info("Starting watch mode...")
            watch_runs(args.poll_seconds, args.skip_artifacts, skip_duplicates, args.debug)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as exc:
        logger.error(f"Unexpected error in main process: {exc}")
        logger.error(f"Full traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
