"""Utilities for Pythia-based synthetic tabular data generation."""

from __future__ import annotations

import random
import inspect
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


CLASS_PREFIX_TEMPLATE = "Class_{label} | "


@dataclass
class ColumnSchema:
    dtype: str
    is_numeric: bool
    min_value: Optional[float]
    max_value: Optional[float]
    integer_coded: bool
    discrete_numeric_values: Optional[List[float]]
    categorical_values: Optional[List[str]]
    mode: Any


@dataclass
class TableSchema:
    columns: List[str]
    target_col: str
    column_schemas: Dict[str, ColumnSchema]


@dataclass
class ClassGenerationStats:
    class_label: int
    requested_rows: int
    accepted_rows_before_resample: int
    total_attempts: int
    rejected_rows: int
    parse_failures: int
    coercion_failures: int
    resampled_rows: int


@dataclass
class SplitGenerationStats:
    split_name: str
    rows_requested: int
    rows_generated: int
    class_counts_requested: Dict[str, int]
    class_counts_generated: Dict[str, int]
    class_stats: Dict[str, ClassGenerationStats]
    dp_stats: Optional[Dict[str, Any]] = None
    training_stats: Optional[Dict[str, Any]] = None


@dataclass
class DPConfig:
    """Differential-privacy settings for LoRA fine-tuning (Yu et al., ICLR 2022)."""
    target_epsilon: float = 5.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 16

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


def _snapshot_trainable_state(model: Any) -> Dict[str, Any]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _restore_trainable_state(model: Any, state: Dict[str, Any]) -> None:
    if not state:
        return
    named_params = dict(model.named_parameters())
    for name, value in state.items():
        if name in named_params:
            named_params[name].data.copy_(value.to(named_params[name].device))


# -------------------------
# General helpers
# -------------------------


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch may be absent until dependencies are installed.
        pass


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    return value


def _format_value_for_text(value: Any) -> str:
    """Format a scalar into deterministic text representation."""
    if pd.isna(value):
        return "NA"

    value = _to_python_scalar(value)

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.15g}"

    return str(value).strip()


# -------------------------
# Schema and serialization
# -------------------------


def derive_table_schema(df: pd.DataFrame, target_col: str) -> TableSchema:
    """Derive schema used to parse, postprocess, and validate synthetic rows."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from source dataframe.")

    column_schemas: Dict[str, ColumnSchema] = {}

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        mode_value = non_null.mode().iloc[0] if not non_null.empty else None

        is_numeric = pd.api.types.is_numeric_dtype(series)
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        integer_coded = False
        discrete_numeric_values: Optional[List[float]] = None
        categorical_values: Optional[List[str]] = None

        if is_numeric:
            numeric_values = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric_values.empty:
                min_value = float(numeric_values.min())
                max_value = float(numeric_values.max())

                unique_values = sorted(numeric_values.unique().tolist())
                if unique_values:
                    # Integer-coded numeric columns are rounded/clipped in postprocessing.
                    integer_coded = bool(np.all(np.isclose(numeric_values, np.round(numeric_values))))

                    # Keep low-cardinality numeric columns on observed support.
                    if len(unique_values) <= 25:
                        discrete_numeric_values = [float(v) for v in unique_values]
        else:
            categorical_values = sorted(non_null.astype(str).unique().tolist()) if not non_null.empty else []

        column_schemas[col] = ColumnSchema(
            dtype=str(series.dtype),
            is_numeric=is_numeric,
            min_value=min_value,
            max_value=max_value,
            integer_coded=integer_coded,
            discrete_numeric_values=discrete_numeric_values,
            categorical_values=categorical_values,
            mode=_to_python_scalar(mode_value),
        )

    return TableSchema(columns=df.columns.tolist(), target_col=target_col, column_schemas=column_schemas)


def serialize_row(row: pd.Series, columns: List[str], target_col: str) -> str:
    """Serialize a row into deterministic text with class prefix."""
    label = int(float(row[target_col]))
    row_parts = [f"{col}={_format_value_for_text(row[col])}" for col in columns]
    return CLASS_PREFIX_TEMPLATE.format(label=label) + " | ".join(row_parts)


def build_training_texts(df: pd.DataFrame, target_col: str) -> List[str]:
    """Create text corpus for causal language-model fine-tuning."""
    columns = df.columns.tolist()
    return [serialize_row(row, columns=columns, target_col=target_col) for _, row in df.iterrows()]


# -------------------------
# Parsing and postprocessing
# -------------------------


def parse_generated_text_to_raw_row(generated_text: str) -> Optional[Dict[str, str]]:
    """Parse generated text back into raw string key-value fields.

    Expected pattern (single line):
      Class_0 | col_a=... | col_b=... | ...
    """
    text = generated_text.strip()
    if not text:
        return None

    # Prefer content after class-prefix if present.
    class_idx = text.find("Class_")
    if class_idx >= 0:
        text = text[class_idx:]

    # Robust extraction: capture key=value pairs even when separators are noisy.
    pairs = re.findall(r"([A-Za-z0-9_\-]+)\s*=\s*([^|\n\r]+)", text)
    if not pairs:
        return None

    raw_values: Dict[str, str] = {}
    for key, value in pairs:
        key = key.strip()
        value = value.strip()
        if not key or key.startswith("Class_"):
            continue
        if key not in raw_values:
            raw_values[key] = value

    if not raw_values:
        return None

    return raw_values


def _coerce_single_value(raw_value: str, column_schema: ColumnSchema) -> Optional[Any]:
    """Coerce one raw string value to typed value under schema constraints."""
    if raw_value is None:
        return None

    txt = str(raw_value).strip()
    txt = txt.replace("<EOR>", "").replace("</s>", "").strip()
    if txt in {"", "NA", "None", "nan", "NaN", "null", "NULL"}:
        return None

    if column_schema.is_numeric:
        try:
            val = float(txt)
        except ValueError:
            return None

        if column_schema.min_value is not None and column_schema.max_value is not None:
            val = float(np.clip(val, column_schema.min_value, column_schema.max_value))

        # Snap to observed support for low-cardinality numeric columns.
        if column_schema.discrete_numeric_values:
            allowed = column_schema.discrete_numeric_values
            nearest = min(allowed, key=lambda x: abs(x - val))
            val = float(nearest)

        if column_schema.integer_coded:
            val = int(round(val))
            if column_schema.min_value is not None and column_schema.max_value is not None:
                val = int(np.clip(val, int(round(column_schema.min_value)), int(round(column_schema.max_value))))
            return val

        return val

    # categorical/object-like
    candidate = txt
    if column_schema.categorical_values:
        if candidate not in column_schema.categorical_values:
            # fallback to mode for unseen category strings
            if column_schema.mode is not None:
                return str(column_schema.mode)
            return column_schema.categorical_values[0] if column_schema.categorical_values else candidate
    return candidate


def _default_value_for_column(column_schema: ColumnSchema) -> Any:
    """Return a schema-safe fallback value for a column."""
    if column_schema.is_numeric:
        if column_schema.mode is not None:
            try:
                val = float(column_schema.mode)
            except Exception:
                val = None
            else:
                if column_schema.integer_coded:
                    return int(round(val))
                return val

        if column_schema.discrete_numeric_values:
            val = float(column_schema.discrete_numeric_values[0])
            return int(round(val)) if column_schema.integer_coded else val

        if column_schema.min_value is not None and column_schema.max_value is not None:
            mid = (column_schema.min_value + column_schema.max_value) / 2.0
            return int(round(mid)) if column_schema.integer_coded else float(mid)

        return 0 if column_schema.integer_coded else 0.0

    if column_schema.mode is not None:
        return str(column_schema.mode)
    if column_schema.categorical_values:
        return column_schema.categorical_values[0]
    return ""


def coerce_raw_row_to_schema(
    raw_values: Dict[str, str],
    schema: TableSchema,
    forced_class_label: int,
) -> Optional[Dict[str, Any]]:
    """Coerce and validate one parsed raw row to schema-compliant typed row."""
    row: Dict[str, Any] = {}
    non_target_cols = [c for c in schema.columns if c != schema.target_col]
    provided = sum(1 for c in non_target_cols if c in raw_values)
    min_required = max(3, int(0.1 * len(non_target_cols)))
    if provided < min_required:
        return None

    for col in schema.columns:
        col_schema = schema.column_schemas[col]

        if col == schema.target_col:
            row[col] = int(forced_class_label)
            continue

        raw_val = raw_values.get(col)
        coerced = _coerce_single_value(raw_val, col_schema) if raw_val is not None else None
        if coerced is None:
            coerced = _default_value_for_column(col_schema)

        row[col] = coerced

    return row


def cast_dataframe_to_schema(df_syn: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Cast a synthetic dataframe to source schema dtypes and order."""
    df = df_syn.copy()

    # Enforce exact column order and presence.
    for col in schema.columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[schema.columns]

    # Type conversion and range control.
    for col, col_schema in schema.column_schemas.items():
        if col_schema.is_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col_schema.min_value is not None and col_schema.max_value is not None:
                df[col] = df[col].clip(lower=col_schema.min_value, upper=col_schema.max_value)

            if col_schema.discrete_numeric_values:
                allowed = np.array(col_schema.discrete_numeric_values, dtype=float)
                if allowed.size > 0:
                    # snap each value to nearest allowed support value
                    values = df[col].to_numpy(dtype=float)
                    nearest_idx = np.abs(values.reshape(-1, 1) - allowed.reshape(1, -1)).argmin(axis=1)
                    df[col] = allowed[nearest_idx]

            if col_schema.integer_coded:
                df[col] = df[col].round()

            dtype_name = col_schema.dtype
            if dtype_name.startswith(("int", "uint")):
                df[col] = df[col].astype("int64")
            elif dtype_name.startswith("float"):
                df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype(str)
            if col_schema.categorical_values:
                valid = set(col_schema.categorical_values)
                fallback = str(col_schema.mode) if col_schema.mode is not None else next(iter(valid), "")
                df[col] = df[col].map(lambda x: x if x in valid else fallback)

    # Final strict cast to original dtypes where possible.
    for col, col_schema in schema.column_schemas.items():
        try:
            df[col] = df[col].astype(col_schema.dtype)
        except Exception:
            # keep coerced fallback dtype if exact astype fails
            pass

    return df


def validate_synthetic_dataframe(df_syn: pd.DataFrame, schema: TableSchema, expected_rows: int) -> None:
    """Validate structural and value-level constraints for synthetic dataframe."""
    if list(df_syn.columns) != schema.columns:
        raise ValueError("Synthetic dataframe columns do not match source schema order.")

    if len(df_syn) != expected_rows:
        raise ValueError(f"Synthetic dataframe row count mismatch: expected={expected_rows}, got={len(df_syn)}")

    if df_syn.isna().any().any():
        missing_cols = df_syn.columns[df_syn.isna().any()].tolist()
        raise ValueError(f"Synthetic dataframe contains missing values in columns: {missing_cols}")

    target_values = set(pd.to_numeric(df_syn[schema.target_col], errors="coerce").dropna().astype(int).tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target column '{schema.target_col}' contains non-binary values: {target_values}")

    # Bounds and discrete checks
    for col, col_schema in schema.column_schemas.items():
        if not col_schema.is_numeric:
            continue

        values = pd.to_numeric(df_syn[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Numeric column '{col}' has non-numeric values after coercion.")

        if col_schema.min_value is not None and (values < col_schema.min_value).any():
            raise ValueError(f"Column '{col}' contains values below min bound.")
        if col_schema.max_value is not None and (values > col_schema.max_value).any():
            raise ValueError(f"Column '{col}' contains values above max bound.")

        if col_schema.integer_coded:
            if not np.all(np.isclose(values, np.round(values))):
                raise ValueError(f"Integer-coded column '{col}' contains non-integer values.")


# -------------------------
# LLM training and generation
# -------------------------


def _lazy_import_training_stack():
    """Import heavy ML dependencies lazily."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def train_lora_model(
    training_texts: List[str],
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    hf_token: Optional[str] = None,
    restore_best_model: bool = True,
):
    """Fine-tune pretrained Pythia model with LoRA."""
    stack = _lazy_import_training_stack()

    torch = stack["torch"]
    Dataset = stack["Dataset"]
    LoraConfig = stack["LoraConfig"]
    TaskType = stack["TaskType"]
    get_peft_model = stack["get_peft_model"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    DataCollatorForLanguageModeling = stack["DataCollatorForLanguageModeling"]
    Trainer = stack["Trainer"]
    TrainerCallback = stack["TrainerCallback"]
    TrainingArguments = stack["TrainingArguments"]
    set_seed = stack["set_seed"]

    class _EpochLossCallback(TrainerCallback):
        def __init__(self):
            self.epoch_losses: List[float] = []
            self._batch_losses: List[float] = []
            self.best_epoch: Optional[int] = None
            self.best_loss: Optional[float] = None
            self.best_state: Optional[Dict[str, Any]] = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self._batch_losses.append(logs["loss"])

        def on_epoch_end(self, args, state, control, **kwargs):
            if self._batch_losses:
                avg = round(sum(self._batch_losses) / len(self._batch_losses), 4)
                self.epoch_losses.append(avg)
                self._batch_losses = []
                epoch_idx = len(self.epoch_losses)
                print(f"  [train] Epoch {epoch_idx}/{epochs} — loss: {avg:.4f}", flush=True)
                improved = self.best_loss is None or avg < self.best_loss
                if improved:
                    self.best_loss = avg
                    self.best_epoch = epoch_idx
                    model = kwargs.get("model")
                    if model is not None:
                        self.best_state = _snapshot_trainable_state(model)

    loss_cb = _EpochLossCallback()
    set_seed(seed)

    device_str = _describe_device(torch)
    print(f"[train] device={device_str} | model={model_name} | rows={len(training_texts)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] LoRA params trainable={trainable:,} / total={total:,} ({100*trainable/total:.2f}%)", flush=True)

    dataset = Dataset.from_dict({"text": training_texts})

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        # Dynamic padding: tokenize without padding, let the collator pad per-batch.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    print("[train] tokenizing training corpus...", flush=True)
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    lens = [len(x) for x in tokenized["input_ids"]]
    print(
        f"[train] token lengths: min={min(lens)} median={int(np.median(lens))} "
        f"p95={int(np.percentile(lens, 95))} max={max(lens)}",
        flush=True,
    )

    use_cuda = torch.cuda.is_available()
    steps_per_epoch = max(1, (len(tokenized) + batch_size - 1) // batch_size)
    print(
        f"[train] epochs={epochs} batch_size={batch_size} "
        f"steps/epoch={steps_per_epoch} total_steps~{steps_per_epoch * epochs}",
        flush=True,
    )

    with TemporaryDirectory(prefix="pythia_lora_") as tmp_out:
        requested_args = {
            "output_dir": tmp_out,
            "overwrite_output_dir": True,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": learning_rate,
            "logging_steps": 25,
            "save_strategy": "no",
            "report_to": [],
            "remove_unused_columns": False,
            "dataloader_pin_memory": use_cuda,
            "dataloader_num_workers": 2,
            "fp16": use_cuda,
            "disable_tqdm": False,
        }
        supported = inspect.signature(TrainingArguments.__init__).parameters
        filtered_args = {k: v for k, v in requested_args.items() if k in supported}

        training_args = TrainingArguments(**filtered_args)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
            ),
            callbacks=[loss_cb],
        )

        t0 = time.time()
        trainer.train()
        print(f"[train] fine-tuning complete in {time.time() - t0:.1f}s", flush=True)

    if restore_best_model and loss_cb.best_state is not None:
        _restore_trainable_state(model, loss_cb.best_state)
        print(
            f"[train] restored best epoch {loss_cb.best_epoch} "
            f"(loss={loss_cb.best_loss:.4f}) for generation",
            flush=True,
        )

    model.eval()
    return model, tokenizer, {
        "epoch_losses": loss_cb.epoch_losses,
        "best_epoch": loss_cb.best_epoch,
        "best_loss": loss_cb.best_loss,
        "epochs_ran": len(loss_cb.epoch_losses),
    }


def _lazy_import_dp_stack():
    """Import opacus lazily (only used in DP path).
    Uses opacus directly — dp-transformers is incompatible with torch>=2.x.
    """
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator

    return {
        "BatchMemoryManager": BatchMemoryManager,
        "PrivacyEngine": PrivacyEngine,
        "ModuleValidator": ModuleValidator,
    }


def train_lora_model_dp(
    training_texts: List[str],
    model_name: str,
    epochs: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    dp_config: DPConfig,
    hf_token: Optional[str] = None,
    restore_best_model: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Fine-tune Pythia with LoRA under (ε, δ)-DP using Opacus DP-SGD.

    Follows Yu et al., "Differentially Private Fine-tuning of Language Models"
    (ICLR 2022): DPSGD is applied only to the LoRA parameters; base weights are
    frozen, so post-processing guarantees the generator is (ε, δ)-DP w.r.t. the
    private training corpus.
    """
    stack = _lazy_import_training_stack()
    dp_stack = _lazy_import_dp_stack()

    torch = stack["torch"]
    Dataset = stack["Dataset"]
    LoraConfig = stack["LoraConfig"]
    TaskType = stack["TaskType"]
    get_peft_model = stack["get_peft_model"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    DataCollatorForLanguageModeling = stack["DataCollatorForLanguageModeling"]
    set_seed = stack["set_seed"]
    BatchMemoryManager = dp_stack["BatchMemoryManager"]
    PrivacyEngine = dp_stack["PrivacyEngine"]
    ModuleValidator = dp_stack["ModuleValidator"]

    set_seed(seed)

    device_str = _describe_device(torch)
    print(
        f"[train-dp] device={device_str} | model={model_name} | rows={len(training_texts)} "
        f"| eps={dp_config.target_epsilon} delta={dp_config.target_delta}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model = ModuleValidator.fix(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[train-dp] LoRA params trainable={trainable:,} / total={total:,} "
        f"({100*trainable/total:.2f}%)",
        flush=True,
    )

    dataset = Dataset.from_dict({"text": training_texts})

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    print("[train-dp] tokenizing training corpus...", flush=True)
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    device = _device_for_model(torch)
    model = model.to(device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    class _TorchDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self._ds = hf_dataset
            self._keys = list(hf_dataset.features.keys())

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, idx):
            item = self._ds[idx]
            return tuple(item[k] for k in self._keys)

    torch_dataset = _TorchDataset(tokenized)

    def _collate_fn(batch):
        keys = list(tokenized.features.keys())
        batch_dict = [{k: item[i] for i, k in enumerate(keys)} for item in batch]
        return data_collator(batch_dict)

    if dp_config.per_device_batch_size <= 0:
        raise ValueError("DP per-device batch size must be positive.")
    if dp_config.gradient_accumulation_steps <= 0:
        raise ValueError("DP gradient accumulation steps must be positive.")

    logical_batch_size = dp_config.effective_batch_size
    physical_batch_size = dp_config.per_device_batch_size

    dataloader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=logical_batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        target_epsilon=dp_config.target_epsilon,
        target_delta=dp_config.target_delta,
        max_grad_norm=dp_config.max_grad_norm,
        epochs=epochs,
    )

    steps_per_epoch = len(dataloader)
    print(
        f"[train-dp] epochs={epochs} per_device_bs={dp_config.per_device_batch_size} "
        f"grad_accum={dp_config.gradient_accumulation_steps} "
        f"effective_bs={logical_batch_size} logical_steps/epoch={steps_per_epoch} "
        f"total_logical_steps~{steps_per_epoch * epochs}",
        flush=True,
    )

    t0 = time.time()
    epoch_losses: List[float] = []
    best_epoch: Optional[int] = None
    best_loss: Optional[float] = None
    best_state: Optional[Dict[str, Any]] = None
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0
        if physical_batch_size < logical_batch_size:
            loader_context = BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=physical_batch_size,
                optimizer=optimizer,
            )
        else:
            loader_context = nullcontext(dataloader)

        with loader_context as memory_safe_loader:
            for step, batch in enumerate(memory_safe_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_items = next(iter(batch.values())).shape[0]
                outputs = model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_items
                total_items += batch_items
        avg_loss = total_loss / max(1, total_items)
        epoch_losses.append(round(avg_loss, 4))
        epoch_idx = epoch + 1
        print(f"  [DP] Epoch {epoch_idx}/{epochs} — loss: {avg_loss:.4f}", flush=True)
        improved = best_loss is None or avg_loss < best_loss
        if improved:
            best_loss = avg_loss
            best_epoch = epoch_idx
            best_state = _snapshot_trainable_state(model)

    print(f"[train-dp] DP fine-tuning complete in {time.time() - t0:.1f}s", flush=True)

    if restore_best_model and best_state is not None:
        _restore_trainable_state(model, best_state)
        print(
            f"[train-dp] restored best epoch {best_epoch} "
            f"(loss={best_loss:.4f}) for generation",
            flush=True,
        )

    achieved_eps_prv: Optional[float] = None
    achieved_eps_rdp: Optional[float] = None
    noise_multiplier: Optional[float] = None
    try:
        achieved_eps_prv = float(privacy_engine.get_epsilon(delta=dp_config.target_delta))
    except Exception:
        achieved_eps_prv = None
    try:
        noise_multiplier = float(optimizer.noise_multiplier)
    except Exception:
        noise_multiplier = None

    dp_stats: Dict[str, Any] = {
        "target_epsilon": dp_config.target_epsilon,
        "target_delta": dp_config.target_delta,
        "max_grad_norm": dp_config.max_grad_norm,
        "per_device_batch_size": dp_config.per_device_batch_size,
        "gradient_accumulation_steps": dp_config.gradient_accumulation_steps,
        "effective_batch_size": dp_config.effective_batch_size,
        "train_samples": int(len(training_texts)),
        "sample_rate": float(dp_config.effective_batch_size) / max(1, len(training_texts)),
        "achieved_epsilon_prv": achieved_eps_prv,
        "achieved_epsilon_rdp": achieved_eps_rdp,
        "noise_multiplier": noise_multiplier,
        "epoch_losses": epoch_losses,
        "best_epoch": best_epoch,
        "best_loss": (round(best_loss, 4) if best_loss is not None else None),
        "epochs_ran": len(epoch_losses),
    }

    # Unwrap GradSampleModule so .generate() works during inference.
    unwrapped = getattr(model, "_module", model)
    unwrapped.eval()
    return unwrapped, tokenizer, dp_stats


def _device_for_model(torch_module) -> Any:
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _describe_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        name = torch_module.cuda.get_device_name(0)
        total_gb = torch_module.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"cuda ({name}, {total_gb:.1f} GiB)"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


def _make_generation_prompt(schema: TableSchema, class_label: int) -> str:
    """Build a stronger constrained prompt to improve structured generation."""
    non_target_cols = [c for c in schema.columns if c != schema.target_col]
    first_col = non_target_cols[0] if non_target_cols else schema.target_col
    return f"Class_{int(class_label)} | {first_col}="


def _build_fallback_rows(
    schema: TableSchema,
    class_label: int,
    n_rows: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Create schema-valid fallback rows when model outputs are unusable."""
    rng = np.random.default_rng(seed + int(class_label))
    rows: List[Dict[str, Any]] = []

    for _ in range(n_rows):
        row: Dict[str, Any] = {}
        for col in schema.columns:
            cs = schema.column_schemas[col]
            if col == schema.target_col:
                row[col] = int(class_label)
                continue

            if cs.is_numeric:
                if cs.discrete_numeric_values:
                    val = float(rng.choice(cs.discrete_numeric_values))
                    row[col] = int(round(val)) if cs.integer_coded else val
                elif cs.min_value is not None and cs.max_value is not None:
                    if cs.integer_coded:
                        low = int(round(cs.min_value))
                        high = int(round(cs.max_value))
                        row[col] = int(rng.integers(low, high + 1))
                    else:
                        row[col] = float(rng.uniform(cs.min_value, cs.max_value))
                else:
                    row[col] = _default_value_for_column(cs)
            else:
                if cs.categorical_values:
                    row[col] = str(rng.choice(cs.categorical_values))
                else:
                    row[col] = _default_value_for_column(cs)
        rows.append(row)

    return rows


def generate_rows_for_class(
    model: Any,
    tokenizer: Any,
    schema: TableSchema,
    class_label: int,
    n_rows: int,
    max_length: int,
    temperature: float,
    top_p: float,
    max_retries_per_row: int,
    generation_batch_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, ClassGenerationStats]:
    """Generate and validate synthetic rows for one class label."""
    stack = _lazy_import_training_stack()
    torch = stack["torch"]

    set_global_seed(seed + 1000 * int(class_label))

    device = next(model.parameters()).device

    accepted_rows: List[Dict[str, Any]] = []
    total_attempts = 0
    parse_failures = 0
    coercion_failures = 0

    max_attempts = max(1, n_rows * max_retries_per_row)
    prompt = _make_generation_prompt(schema=schema, class_label=int(class_label))

    print(
        f"[gen] class={class_label} target={n_rows} batch={generation_batch_size} "
        f"max_new_tokens={max_length} max_attempts={max_attempts}",
        flush=True,
    )
    t0 = time.time()

    pbar = tqdm(
        total=n_rows,
        desc=f"gen class={class_label}",
        unit="row",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    try:
        while len(accepted_rows) < n_rows and total_attempts < max_attempts:
            batch = min(generation_batch_size, n_rows - len(accepted_rows), max_attempts - total_attempts)
            prompts = [prompt] * batch

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

            with torch.inference_mode():
                requested_gen_args = {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_length,
                    "min_new_tokens": min(64, max_length),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                generate_params = inspect.signature(model.generate).parameters
                supports_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in generate_params.values()
                )
                filtered_gen_args = (
                    requested_gen_args
                    if supports_var_kwargs
                    else {k: v for k, v in requested_gen_args.items() if k in generate_params}
                )

                outputs = model.generate(
                    **inputs,
                    **filtered_gen_args,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            accepted_before_batch = len(accepted_rows)
            for text in decoded:
                total_attempts += 1
                parsed = parse_generated_text_to_raw_row(text)
                if parsed is None:
                    parse_failures += 1
                    continue

                row = coerce_raw_row_to_schema(parsed, schema=schema, forced_class_label=int(class_label))
                if row is None:
                    coercion_failures += 1
                    continue

                accepted_rows.append(row)
                if len(accepted_rows) >= n_rows:
                    break

            pbar.update(len(accepted_rows) - accepted_before_batch)
            accept_rate = len(accepted_rows) / max(1, total_attempts)
            pbar.set_postfix({
                "attempts": total_attempts,
                "accept%": f"{100*accept_rate:.1f}",
                "parse_fail": parse_failures,
                "coerce_fail": coercion_failures,
            })
    finally:
        pbar.close()

    elapsed = time.time() - t0
    rate = (len(accepted_rows) / elapsed) if elapsed > 0 else 0.0
    print(
        f"[gen] class={class_label} done in {elapsed:.1f}s "
        f"(accepted={len(accepted_rows)}/{n_rows} @ {rate:.1f} rows/s, "
        f"parse_fail={parse_failures}, coerce_fail={coercion_failures})",
        flush=True,
    )

    accepted_before_resample = len(accepted_rows)
    resampled_rows = 0

    # If still short, resample from valid accepted rows within same class.
    if len(accepted_rows) < n_rows:
        needed = n_rows - len(accepted_rows)
        resampled_rows = needed

        if accepted_rows:
            fill_idx = np.random.choice(len(accepted_rows), size=needed, replace=True)
            accepted_rows.extend([accepted_rows[int(i)].copy() for i in fill_idx])
        else:
            # Last-resort fallback: synthesize schema-valid random rows.
            accepted_rows.extend(
                _build_fallback_rows(
                    schema=schema,
                    class_label=int(class_label),
                    n_rows=needed,
                    seed=seed,
                )
            )

    class_df = pd.DataFrame(accepted_rows, columns=schema.columns)
    class_df = cast_dataframe_to_schema(class_df, schema)

    stats = ClassGenerationStats(
        class_label=int(class_label),
        requested_rows=int(n_rows),
        accepted_rows_before_resample=int(accepted_before_resample),
        total_attempts=int(total_attempts),
        rejected_rows=int(max(0, total_attempts - accepted_before_resample)),
        parse_failures=int(parse_failures),
        coercion_failures=int(coercion_failures),
        resampled_rows=int(resampled_rows),
    )

    return class_df, stats


def _class_counts(df: pd.DataFrame, target_col: str) -> Dict[int, int]:
    target = pd.to_numeric(df[target_col], errors="coerce").round().astype(int)
    return {0: int((target == 0).sum()), 1: int((target == 1).sum())}


def generate_synthetic_for_split(
    split_name: str,
    source_df: pd.DataFrame,
    model_name: str,
    target_col: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    temperature: float,
    top_p: float,
    max_retries_per_row: int,
    seed: int,
    generation_batch_size: Optional[int] = None,
    hf_token: Optional[str] = None,
    dp_config: Optional[DPConfig] = None,
    restore_best_model: bool = True,
) -> Tuple[pd.DataFrame, SplitGenerationStats]:
    """End-to-end generation for one split: train, sample, postprocess, validate."""
    if generation_batch_size is None:
        generation_batch_size = batch_size

    print(f"[split:{split_name}] rows={len(source_df)} cols={len(source_df.columns)} target='{target_col}'", flush=True)

    schema = derive_table_schema(source_df, target_col=target_col)
    training_texts = build_training_texts(source_df, target_col=target_col)

    dp_stats: Optional[Dict[str, Any]] = None
    training_stats: Optional[Dict[str, Any]] = None
    if dp_config is not None:
        model, tokenizer, dp_stats = train_lora_model_dp(
            training_texts=training_texts,
            model_name=model_name,
            epochs=epochs,
            learning_rate=learning_rate,
            max_length=max_length,
            seed=seed,
            dp_config=dp_config,
            hf_token=hf_token,
            restore_best_model=restore_best_model,
        )
        training_stats = {
            "epoch_losses": dp_stats.get("epoch_losses", []),
            "best_epoch": dp_stats.get("best_epoch"),
            "best_loss": dp_stats.get("best_loss"),
            "epochs_ran": dp_stats.get("epochs_ran"),
        }
    else:
        model, tokenizer, training_stats = train_lora_model(
            training_texts=training_texts,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            seed=seed,
            hf_token=hf_token,
            restore_best_model=restore_best_model,
        )

    # Move model to target device once (generation reuses it across classes).
    stack = _lazy_import_training_stack()
    torch = stack["torch"]
    device = _device_for_model(torch)
    model = model.to(device)
    print(f"[split:{split_name}] model placed on device={device}", flush=True)

    requested_counts = _class_counts(source_df, target_col=target_col)
    print(f"[split:{split_name}] per-class targets: {requested_counts}", flush=True)

    generated_parts: List[pd.DataFrame] = []
    class_stats: Dict[str, ClassGenerationStats] = {}

    for class_label in [0, 1]:
        n_rows = requested_counts.get(class_label, 0)
        if n_rows <= 0:
            continue

        class_df, stats = generate_rows_for_class(
            model=model,
            tokenizer=tokenizer,
            schema=schema,
            class_label=class_label,
            n_rows=n_rows,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            max_retries_per_row=max_retries_per_row,
            generation_batch_size=generation_batch_size,
            seed=seed,
        )

        generated_parts.append(class_df)
        class_stats[str(class_label)] = stats

        # Release batch-level KV/activation cache between classes to keep VRAM bounded.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[split:{split_name}] postprocessing + validation...", flush=True)
    synthetic_df = pd.concat(generated_parts, axis=0, ignore_index=True)
    synthetic_df = synthetic_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    synthetic_df = cast_dataframe_to_schema(synthetic_df, schema)
    validate_synthetic_dataframe(synthetic_df, schema=schema, expected_rows=len(source_df))

    generated_counts = _class_counts(synthetic_df, target_col=target_col)

    split_stats = SplitGenerationStats(
        split_name=split_name,
        rows_requested=int(len(source_df)),
        rows_generated=int(len(synthetic_df)),
        class_counts_requested={str(k): int(v) for k, v in requested_counts.items()},
        class_counts_generated={str(k): int(v) for k, v in generated_counts.items()},
        class_stats=class_stats,
        dp_stats=dp_stats,
        training_stats=training_stats,
    )

    return synthetic_df, split_stats


def stats_to_dict(stats: SplitGenerationStats) -> Dict[str, Any]:
    """Convert nested dataclass stats into JSON-serializable dict."""
    return {
        "split_name": stats.split_name,
        "rows_requested": stats.rows_requested,
        "rows_generated": stats.rows_generated,
        "class_counts_requested": stats.class_counts_requested,
        "class_counts_generated": stats.class_counts_generated,
        "class_stats": {
            key: {
                "class_label": val.class_label,
                "requested_rows": val.requested_rows,
                "accepted_rows_before_resample": val.accepted_rows_before_resample,
                "total_attempts": val.total_attempts,
                "rejected_rows": val.rejected_rows,
                "parse_failures": val.parse_failures,
                "coercion_failures": val.coercion_failures,
                "resampled_rows": val.resampled_rows,
            }
            for key, val in stats.class_stats.items()
        },
        "dp_stats": stats.dp_stats,
        "training_stats": stats.training_stats,
    }


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
