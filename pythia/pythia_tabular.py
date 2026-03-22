"""Utilities for Pythia-based synthetic tabular data generation."""

from __future__ import annotations

import random
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
    TrainingArguments = stack["TrainingArguments"]
    set_seed = stack["set_seed"]

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_dict({"text": training_texts})

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    use_cuda = torch.cuda.is_available()

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
            "fp16": use_cuda,
        }
        supported = inspect.signature(TrainingArguments.__init__).parameters
        filtered_args = {k: v for k, v in requested_args.items() if k in supported}

        training_args = TrainingArguments(**filtered_args)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        trainer.train()

    model.eval()
    return model, tokenizer


def _device_for_model(torch_module) -> Any:
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


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

    set_global_seed(seed + int(class_label))

    device = _device_for_model(torch)
    model = model.to(device)

    accepted_rows: List[Dict[str, Any]] = []
    total_attempts = 0
    parse_failures = 0
    coercion_failures = 0

    max_attempts = max(1, n_rows * max_retries_per_row)
    prompt = _make_generation_prompt(schema=schema, class_label=int(class_label))

    while len(accepted_rows) < n_rows and total_attempts < max_attempts:
        batch = min(generation_batch_size, n_rows - len(accepted_rows), max_attempts - total_attempts)
        prompts = [prompt] * batch

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            requested_gen_args = {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_length,
                "min_new_tokens": min(64, max_length),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            supported_gen_args = inspect.signature(model.generate).parameters
            filtered_gen_args = {k: v for k, v in requested_gen_args.items() if k in supported_gen_args}

            outputs = model.generate(
                **inputs,
                **filtered_gen_args,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
) -> Tuple[pd.DataFrame, SplitGenerationStats]:
    """End-to-end generation for one split: train, sample, postprocess, validate."""
    if generation_batch_size is None:
        generation_batch_size = batch_size

    schema = derive_table_schema(source_df, target_col=target_col)
    training_texts = build_training_texts(source_df, target_col=target_col)

    model, tokenizer = train_lora_model(
        training_texts=training_texts,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )

    requested_counts = _class_counts(source_df, target_col=target_col)

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
    }


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
