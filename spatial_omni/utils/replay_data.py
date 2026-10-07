"""Replay manifest normalization and local/OSS audio loading."""

from __future__ import annotations

import configparser
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        value = content.get("text")
        return str(value).strip() if value is not None else ""
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict) and item.get("text") is not None:
            text = str(item["text"]).strip()
        else:
            text = ""
        if text:
            parts.append(text)
    return "\n".join(parts)


def _content_audio_paths(content: Any) -> List[str]:
    items = content if isinstance(content, list) else [content]
    paths: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("audio")
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
        elif isinstance(value, list):
            if not all(isinstance(path, str) and path.strip() for path in value):
                raise ValueError("audio list entries must be non-empty strings")
            paths.extend(path.strip() for path in value)
        elif value is not None:
            raise ValueError("audio must be a non-empty string or string list")
    return paths


def _record_location(source_path: Optional[str], record_index: Optional[int]) -> str:
    source = source_path or "<record>"
    if record_index is None:
        return source
    return f"{source}:{record_index + 1}"


def _build_chat_prompt(context: Sequence[Tuple[str, str]]) -> str:
    if not context:
        return ""
    roles = {role for role, _ in context}
    if roles.issubset({"system", "user"}):
        return "\n\n".join(text for _, text in context)
    return "\n\n".join(f"{role.capitalize()}: {text}" for role, text in context)


def normalize_replay_record(
    record: Any,
    *,
    record_index: Optional[int] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize flat Spatial-Omni QA or Qwen ChatML into replay QA fields."""

    location = _record_location(source_path, record_index)
    if isinstance(record, dict) and record.get("audio_path") is not None:
        return dict(record)

    if isinstance(record, list):
        data: Dict[str, Any] = {"messages": record}
    elif isinstance(record, dict):
        data = record
    else:
        raise ValueError(f"{location}: expected an object or messages list")

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location}: missing non-empty messages")

    target_index = None
    answer = ""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        candidate = _content_text(message.get("content"))
        if candidate:
            target_index = index
            answer = candidate
            break
    if target_index is None:
        raise ValueError(f"{location}: missing non-empty assistant answer")
    for message in messages[target_index + 1 :]:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if _content_text(content) or _content_audio_paths(content):
            raise ValueError(
                f"{location}: found non-empty messages after the target assistant answer"
            )

    audio_paths: List[str] = []
    context: List[Tuple[str, str]] = []
    user_texts: List[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if index >= target_index:
            continue
        role = str(message.get("role") or "user").lower()
        if role == "human":
            role = "user"
        if role == "user":
            audio_paths.extend(_content_audio_paths(message.get("content")))
        text = _content_text(message.get("content"))
        if text:
            context.append((role, text))
            if role == "user":
                user_texts.append(text)

    audio_paths = list(dict.fromkeys(audio_paths))
    if len(audio_paths) != 1:
        raise ValueError(
            f"{location}: expected exactly one audio input, found {len(audio_paths)}"
        )

    prompt = _build_chat_prompt(context)
    if not prompt:
        raise ValueError(f"{location}: no textual system/user prompt before the answer")

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    task_name = (
        data.get("task_name")
        or meta.get("task")
        or data.get("source")
        or "qwen_audio_sft"
    )
    base_id = (
        data.get("pair_id")
        or data.get("uniq_id")
        or meta.get("sample_id")
        or meta.get("id")
    )
    pair_id = data.get("pair_id")
    if pair_id is None and base_id is not None:
        pair_id = f"{base_id}::{task_name}"

    normalized = {
        "audio_path": audio_paths[0],
        "prompt": prompt,
        "question": user_texts[-1] if user_texts else prompt,
        "answer": answer,
        "pair_id": pair_id,
        "task_name": str(task_name),
        "data_source": data.get("source") or meta.get("source"),
        "_replay_source_format": "qwen_chatml",
    }
    return normalized


def iter_qa_records(qa_path: str, max_samples: Optional[int] = None) -> Iterator[Any]:
    """Yield JSON/JSONL records without first materializing a second full list."""

    qa_path = os.path.abspath(os.path.expanduser(qa_path))
    qa_path_lower = qa_path.lower()
    yielded = 0
    if qa_path_lower.endswith(".jsonl"):
        with open(qa_path, encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if max_samples is not None and yielded >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{qa_path}:{line_number}: invalid JSON: {exc}") from exc
                yielded += 1
                yield record
        return

    if qa_path_lower.endswith(".json"):
        with open(qa_path, encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if (
            isinstance(payload, list)
            and payload
            and all(
                isinstance(message, dict)
                and "role" in message
                and "content" in message
                for message in payload
            )
        ):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and (
            "messages" in payload or "audio_path" in payload
        ):
            records = [payload]
        elif isinstance(payload, dict) and "records" in payload:
            records = payload["records"]
        elif isinstance(payload, dict) and "data" in payload:
            records = payload["data"]
        else:
            raise ValueError(
                f"{qa_path}: expected a record, messages list, or records/data container"
            )
        if not isinstance(records, list):
            raise ValueError(f"{qa_path}: JSON records/data must be a list")
        for record in records:
            if max_samples is not None and yielded >= max_samples:
                break
            yielded += 1
            yield record
        return

    raise ValueError(f"Unsupported QA format: {qa_path}")


class _Credentials:
    __slots__ = ("endpoint", "access_key_id", "access_key_secret", "security_token")

    def __init__(
        self,
        endpoint: str,
        access_key_id: str,
        access_key_secret: str,
        security_token: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.security_token = security_token

    def __repr__(self) -> str:
        return f"_Credentials(endpoint={self.endpoint!r}, credentials=<redacted>)"


def _first(mapping: Any, *keys: str) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _credentials_from_mapping(mapping: Any, location: str) -> _Credentials:
    endpoint = _first(mapping, "endpoint")
    access_key_id = _first(mapping, "access_key_id", "accessKeyID", "accesskeyid")
    access_key_secret = _first(
        mapping, "access_key_secret", "accessKeySecret", "accesskeysecret"
    )
    security_token = _first(
        mapping, "security_token", "sts_token", "stsToken", "ststoken"
    )
    missing = [
        name
        for name, value in (
            ("endpoint", endpoint),
            ("access key ID", access_key_id),
            ("access key secret", access_key_secret),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"{location}: missing {', '.join(missing)}")
    return _Credentials(endpoint, access_key_id, access_key_secret, security_token)


def _load_oss_config(config_path: str) -> Tuple[Optional[_Credentials], Dict[str, _Credentials]]:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"OSS config not found: {path}. Pass --oss-config or set SO_OSS_CONFIG."
        )

    with path.open("r", encoding="utf-8-sig") as handle:
        config_text = handle.read()
    if config_text.lstrip().startswith("{"):
        try:
            payload = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}: invalid Qwen OSS JSON config at line {exc.lineno}, "
                f"column {exc.colno}"
            ) from None
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: Qwen OSS JSON config must be an object")
        buckets = {
            str(bucket): _credentials_from_mapping(meta, f"{path}:{bucket}")
            for bucket, meta in payload.items()
            if isinstance(meta, dict)
        }
        if not buckets:
            raise ValueError(f"{path}: no bucket credentials found")
        return None, buckets

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(config_text)
    except configparser.Error:
        # ConfigParser errors may echo the offending line, which can contain
        # an access key secret. Keep the public error deliberately terse.
        raise ValueError(f"{path}: invalid OSS INI config") from None
    section_name = next(
        (name for name in parser.sections() if name.lower() == "credentials"), None
    )
    if section_name is None:
        raise ValueError(f"{path}: missing [Credentials] section")
    return _credentials_from_mapping(parser[section_name], str(path)), {}


class OssAudioReader:
    """Decode local or OSS audio while preserving sample rate and channels."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        cache_dir: Optional[str] = None,
        retries: int = 3,
        connect_timeout: float = 30.0,
    ) -> None:
        self.config_path = os.path.abspath(
            os.path.expanduser(
                config_path
                or os.environ.get("SO_OSS_CONFIG", "~/.ossutilconfig")
            )
        )
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir else None
        )
        self.retries = max(1, int(retries))
        self.connect_timeout = float(connect_timeout)
        self._default_credentials: Optional[_Credentials] = None
        self._bucket_credentials: Dict[str, _Credentials] = {}
        self._buckets: Dict[str, Any] = {}
        self._pid = os.getpid()
        self._config_loaded = False

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        # Bucket clients are not guaranteed to be pickle-safe across oss2
        # versions. Workers should also load credentials from their own config
        # instead of receiving credential objects serialized by the parent.
        state["_default_credentials"] = None
        state["_bucket_credentials"] = {}
        state["_buckets"] = {}
        state["_config_loaded"] = False
        return state

    @staticmethod
    def is_oss_path(path: str) -> bool:
        return isinstance(path, str) and path.startswith("oss://")

    @staticmethod
    def _split_uri(uri: str) -> Tuple[str, str]:
        if not uri.startswith("oss://"):
            raise ValueError(f"Not an OSS URI: {uri}")
        bucket, separator, key = uri[6:].partition("/")
        if not separator or not bucket or not key:
            raise ValueError(f"Invalid OSS URI: {uri}")
        return bucket, key

    def _ensure_process(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._buckets = {}

    def _ensure_config(self) -> None:
        if self._config_loaded:
            return
        self._default_credentials, self._bucket_credentials = _load_oss_config(
            self.config_path
        )
        self._config_loaded = True

    def _get_bucket(self, bucket_name: str) -> Any:
        self._ensure_process()
        self._ensure_config()
        if bucket_name in self._buckets:
            return self._buckets[bucket_name]
        credentials = self._bucket_credentials.get(bucket_name) or self._default_credentials
        if credentials is None:
            available = ", ".join(sorted(self._bucket_credentials))
            raise KeyError(
                f"Bucket {bucket_name!r} is absent from {self.config_path}; "
                f"configured buckets: {available or '<none>'}"
            )
        try:
            import oss2
        except ImportError as exc:
            raise ImportError(
                "OSS replay requires oss2. Install project requirements or `pip install oss2`."
            ) from exc
        if credentials.security_token:
            auth = oss2.StsAuth(
                credentials.access_key_id,
                credentials.access_key_secret,
                credentials.security_token,
            )
        else:
            auth = oss2.Auth(credentials.access_key_id, credentials.access_key_secret)
        bucket = oss2.Bucket(
            auth,
            credentials.endpoint,
            bucket_name,
            connect_timeout=self.connect_timeout,
        )
        self._buckets[bucket_name] = bucket
        return bucket

    def _cache_path(self, uri: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        _, key = self._split_uri(uri)
        suffix = Path(key).suffix
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix):
            suffix = ".bin"
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}{suffix}"

    @staticmethod
    def _is_non_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status", None)
        return status in {400, 401, 403, 404}

    def _download(self, uri: str) -> bytes:
        bucket_name, key = self._split_uri(uri)
        bucket = self._get_bucket(bucket_name)
        last_error: Optional[Exception] = None
        attempts = 0
        for attempt in range(1, self.retries + 1):
            attempts = attempt
            try:
                data = bucket.get_object(key).read()
                if not data:
                    raise ValueError(f"OSS object is empty: {uri}")
                return data
            except Exception as exc:
                last_error = exc
                if self._is_non_retryable(exc) or attempt >= self.retries:
                    break
                time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))
        error_name = type(last_error).__name__ if last_error is not None else "unknown"
        raise RuntimeError(
            f"Failed to read OSS audio after {attempts} attempt(s): {uri} "
            f"({error_name})"
        ) from last_error

    def _prepare_cache_path(self, cache_path: Path) -> None:
        if self.cache_dir is None:
            raise RuntimeError("cache path requested without a cache directory")
        self.cache_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.cache_dir, 0o700)
        if cache_path.parent.is_symlink():
            raise RuntimeError(f"Refusing symlinked OSS cache directory: {cache_path.parent}")
        cache_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(cache_path.parent, 0o700)
        if cache_path.is_symlink():
            raise RuntimeError(f"Refusing symlinked OSS cache file: {cache_path}")
        if cache_path.is_file():
            os.chmod(cache_path, 0o600)

    def _write_cache(self, cache_path: Path, data: bytes) -> None:
        self._prepare_cache_path(cache_path)
        temporary = cache_path.with_name(
            f".{cache_path.name}.tmp-{os.getpid()}-{threading.get_ident()}-"
            f"{uuid.uuid4().hex}"
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, cache_path)
            os.chmod(cache_path, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _decode_audio(
        data: bytes,
        *,
        dtype: str,
        always_2d: bool,
    ) -> Tuple[np.ndarray, int]:
        return sf.read(io.BytesIO(data), dtype=dtype, always_2d=always_2d)

    def read_bytes(self, uri: str) -> bytes:
        cache_path = self._cache_path(uri)
        if cache_path is not None:
            self._prepare_cache_path(cache_path)
            if cache_path.is_file():
                data = cache_path.read_bytes()
                if data:
                    return data

        data = self._download(uri)
        if cache_path is not None:
            self._write_cache(cache_path, data)
        return data

    def read_audio(
        self,
        path: str,
        *,
        dtype: str = "float32",
        always_2d: bool = True,
    ) -> Tuple[np.ndarray, int]:
        if not self.is_oss_path(path):
            return sf.read(path, dtype=dtype, always_2d=always_2d)

        cache_path = self._cache_path(path)
        if cache_path is not None:
            self._prepare_cache_path(cache_path)
        if cache_path is not None and cache_path.is_file():
            try:
                cached_data = cache_path.read_bytes()
                if cached_data:
                    return self._decode_audio(
                        cached_data, dtype=dtype, always_2d=always_2d
                    )
            except (FileNotFoundError, OSError):
                pass
            except Exception:
                # A partial cache write from an interrupted older run must not
                # make the object permanently unreadable.
                try:
                    cache_path.unlink()
                except FileNotFoundError:
                    pass

        data = self._download(path)
        try:
            audio = self._decode_audio(data, dtype=dtype, always_2d=always_2d)
        except Exception as exc:
            raise ValueError(f"Failed to decode OSS audio {path}: {exc}") from exc
        if cache_path is not None:
            self._write_cache(cache_path, data)
        return audio
