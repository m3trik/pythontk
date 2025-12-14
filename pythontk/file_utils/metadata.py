# !/usr/bin/python
# coding=utf-8
import os
import json
from typing import Dict, Optional, Any, List
from pythontk.core_utils.namedtuple_container import NamedTupleContainer


class MetadataInternal:
    """Internal utilities for handling file metadata on Windows and Linux."""

    enable_sidecar = False

    @staticmethod
    def _get_sidecar_path(file_path: str) -> str:
        """Get the path to the sidecar metadata file."""
        return f"{file_path}.metadata.json"

    @staticmethod
    def _load_sidecar(file_path: str) -> Dict[str, Any]:
        """Load metadata from the sidecar file."""
        sidecar_path = MetadataInternal._get_sidecar_path(file_path)
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    @staticmethod
    def _save_sidecar(file_path: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to the sidecar file."""
        sidecar_path = MetadataInternal._get_sidecar_path(file_path)
        # Load existing to update
        current_data = MetadataInternal._load_sidecar(file_path)
        current_data.update(metadata)

        # Remove None values
        current_data = {k: v for k, v in current_data.items() if v is not None}

        try:
            with open(sidecar_path, "w") as f:
                json.dump(current_data, f, indent=4)

            # Hide the file on Windows
            if os.name == "nt":
                try:
                    import ctypes

                    FILE_ATTRIBUTE_HIDDEN = 0x02
                    ctypes.windll.kernel32.SetFileAttributesW(
                        sidecar_path, FILE_ATTRIBUTE_HIDDEN
                    )
                except ImportError:
                    pass
        except OSError as e:
            print(f"Error saving sidecar metadata: {e}")

    @classmethod
    def _get(cls, file_path: str, *keys: str) -> Dict[str, Optional[str]]:
        """Retrieves metadata from a specified file.
        Supports Windows (via Shell.Application) and Linux (via extended attributes).

        Parameters:
            file_path (str): The full path to the file from which metadata will be retrieved.
            *keys (str): The metadata keys to retrieve.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Returns:
            Dict[str, Optional[str]]: A dictionary containing the requested metadata key-value pairs.

        Example:
            metadata = _get(
                "C:\\path\\to\\your\\file.txt",
                "Custom_PlaneName",
                "Custom_ModuleType",
                "Comments"
            )
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        metadata = {}

        if os.name == "nt":
            try:
                import win32com.client
            except ImportError:
                return {}

            shell = win32com.client.Dispatch("Shell.Application")
            folder = shell.NameSpace(os.path.dirname(file_path))
            item = folder.ParseName(os.path.basename(file_path))

            for key in keys:
                metadata[key] = item.ExtendedProperty(key)

            # Overlay sidecar metadata if available
            if cls.enable_sidecar:
                sidecar_data = cls._load_sidecar(file_path)
                for key in keys:
                    if key in sidecar_data:
                        metadata[key] = sidecar_data[key]

        else:  # POSIX (Linux/Mac)
            # Map common friendly names to xattr keys (Freedesktop/XDG standards)
            key_map = {
                "Comments": "user.comment",
                "Title": "user.title",
                "Subject": "user.subject",
                "Authors": "user.dublincore.creator",
                "Keywords": "user.xdg.tags",
                "Tags": "user.xdg.tags",
                "Copyright": "user.copyright",
            }

            for key in keys:
                xattr_key = key_map.get(key, f"user.{key.lower()}")
                try:
                    if hasattr(os, "getxattr"):
                        value_bytes = os.getxattr(file_path, xattr_key)
                        metadata[key] = value_bytes.decode("utf-8")
                    else:
                        metadata[key] = None
                except (OSError, AttributeError):
                    metadata[key] = None

        return metadata

    @classmethod
    def _set(cls, file_path: str, **metadata: Dict[str, Any]) -> None:
        """Attaches or modifies metadata for a specified file.
        Supports Windows (via Property System) and Linux (via extended attributes).

        Supported Keys (Friendly Names):
        - Comments (str)
        - Title (str)
        - Subject (str)
        - Authors (str or list)
        - Keywords / Tags (str or list)
        - Copyright (str)
        - Category (str)
        - Status (str)
        - Rating (int): 0-99 (1-12: 1*, 13-37: 2*, 38-62: 3*, 63-87: 4*, 88-99: 5*)

        Parameters:
            file_path (str): The full path to the file to which metadata will be attached or modified.
            **metadata: Key-value pairs representing the metadata properties and their values.
                If a value is None, the corresponding metadata property will be removed.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Example:
            _set(
                "C:\\path\\to\\your\\file.txt",
                Comments="This is a sample comment.",
                Title="My Document Title",
                Tags=["python", "script"],
                Rating=75
            )
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        if os.name == "nt":
            try:
                from win32com.propsys import propsys, pscon
                from win32com.shell import shellcon
                import pythoncom
            except ImportError:
                print("Error: win32com.propsys is required to set metadata.")
                return

            try:
                # Normalize path for Windows API
                file_path = os.path.normpath(file_path)
                store = propsys.SHGetPropertyStoreFromParsingName(
                    file_path, None, shellcon.GPS_READWRITE, propsys.IID_IPropertyStore
                )
            except Exception as e:
                if cls.enable_sidecar:
                    print(
                        f"Error opening property store for {file_path}: {e}. Falling back to sidecar."
                    )
                    cls._save_sidecar(file_path, metadata)
                    return
                else:
                    raise RuntimeError(
                        f"Error accessing file metadata: {e}.\n"
                        "This is common with cloud storage (Dropbox/OneDrive). "
                        "Enable sidecar support (Metadata.enable_sidecar = True) to fix this."
                    )

            # Map common friendly names to System property keys
            key_map = {
                "Comments": "System.Comment",
                "Title": "System.Title",
                "Subject": "System.Subject",
                "Authors": "System.Author",
                "Keywords": "System.Keywords",
                "Tags": "System.Keywords",
                "Copyright": "System.Copyright",
                "Category": "System.Category",
                "Status": "System.ContentStatus",
                "Rating": "System.Rating",
            }

            for key, value in metadata.items():
                canonical_name = key_map.get(key, key)
                try:
                    pkey = propsys.PSGetPropertyKeyFromName(canonical_name)
                except Exception:
                    print(f"Warning: Could not resolve property key for '{key}'")
                    continue

                # Handle list/tuple values for string properties (join with semicolon)
                if isinstance(value, (list, tuple)):
                    value = ";".join(map(str, value))

                try:
                    if value is None:
                        store.SetValue(
                            pkey, propsys.PROPVARIANTType(None, pythoncom.VT_EMPTY)
                        )
                    else:
                        store.SetValue(pkey, value)
                except Exception as e:
                    print(f"Error setting value for '{key}': {e}")

            try:
                store.Commit()
            except Exception as e:
                if cls.enable_sidecar:
                    print(f"Error committing metadata changes: {e}")
                    # Fallback to sidecar if commit fails
                    cls._save_sidecar(file_path, metadata)
                else:
                    raise RuntimeError(
                        f"Error committing metadata: {e}.\n"
                        "This is common with cloud storage (Dropbox/OneDrive). "
                        "Enable sidecar support (Metadata.enable_sidecar = True) to fix this."
                    )

        else:  # POSIX (Linux/Mac)
            if not hasattr(os, "setxattr"):
                print("Error: os.setxattr is not available on this system.")
                return

            # Map common friendly names to xattr keys
            key_map = {
                "Comments": "user.comment",
                "Title": "user.title",
                "Subject": "user.subject",
                "Authors": "user.dublincore.creator",
                "Keywords": "user.xdg.tags",
                "Tags": "user.xdg.tags",
                "Copyright": "user.copyright",
            }

            for key, value in metadata.items():
                xattr_key = key_map.get(key, f"user.{key.lower()}")

                # Handle list/tuple values
                if isinstance(value, (list, tuple)):
                    # Linux tags are often comma separated
                    value = ",".join(map(str, value))

                try:
                    if value is None:
                        try:
                            os.removexattr(file_path, xattr_key)
                        except OSError:
                            pass
                    else:
                        os.setxattr(file_path, xattr_key, str(value).encode("utf-8"))
                except OSError as e:
                    print(f"Error setting xattr '{key}': {e}")

    @classmethod
    def _get_tag(cls, file_path: str, key: str) -> Optional[str]:
        """Retrieves a value from the file's tags/keywords using a 'Key:Value' format.

        Parameters:
            file_path (str): Path to the file.
            key (str): The key to look for.

        Returns:
            str: The value associated with the key, or None if not found.
        """
        meta = cls._get(file_path, "Keywords")
        tags = meta.get("Keywords")

        if not tags:
            return None

        # Handle various return types from ExtendedProperty or xattr
        if isinstance(tags, str):
            # Windows uses semicolon, Linux often uses comma
            if ";" in tags:
                tags = [t.strip() for t in tags.split(";") if t.strip()]
            elif "," in tags:
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            else:
                tags = [tags.strip()]
        elif not isinstance(tags, (list, tuple)):
            return None

        prefix = f"{key}:"
        for tag in tags:
            if tag.startswith(prefix):
                return tag[len(prefix) :].strip()
        return None

    @classmethod
    def _set_tag(cls, file_path: str, key: str, value: Optional[str]) -> None:
        """Sets a 'Key:Value' tag in the file's metadata.

        This allows storing custom key-value pairs within the standard 'Tags'/'Keywords' field.

        Parameters:
            file_path (str): Path to the file.
            key (str): The key for the tag.
            value (str): The value to set. If None, the key is removed.
        """
        # Get existing tags
        meta = cls._get(file_path, "Keywords")
        tags = meta.get("Keywords")

        # Normalize to list
        if tags is None:
            tag_list = []
        elif isinstance(tags, str):
            if ";" in tags:
                tag_list = [t.strip() for t in tags.split(";") if t.strip()]
            elif "," in tags:
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            else:
                tag_list = [tags.strip()] if tags.strip() else []
        elif isinstance(tags, (list, tuple)):
            tag_list = list(tags)
        else:
            tag_list = []

        # Remove existing key
        prefix = f"{key}:"
        tag_list = [t for t in tag_list if not t.startswith(prefix)]

        # Add new key if value is not None
        if value is not None:
            tag_list.append(f"{key}:{value}")

        # Write back
        cls._set(file_path, Keywords=tag_list)

    @classmethod
    def _batch_get(cls, file_paths: List[str], *keys: str) -> NamedTupleContainer:
        """Retrieves metadata for multiple files and returns a NamedTupleContainer.

        Parameters:
            file_paths (List[str]): List of file paths to query.
            *keys (str): The metadata keys to retrieve.

        Returns:
            NamedTupleContainer: A container holding the metadata for each file.
                The container will have fields: 'filepath' + requested keys.
        """
        fields = ["filepath"] + list(keys)
        data = []

        for fp in file_paths:
            if not os.path.exists(fp):
                continue
            meta = cls._get(fp, *keys)
            row = [fp] + [meta.get(k) for k in keys]
            data.append(tuple(row))

        return NamedTupleContainer(named_tuples=data, fields=fields)

    @classmethod
    def _batch_set(cls, file_paths: List[str], **metadata: Dict[str, Any]) -> None:
        """Sets the same metadata for multiple files.

        Parameters:
            file_paths (List[str]): List of file paths to update.
            **metadata: Key-value pairs of metadata to set.
        """
        for fp in file_paths:
            if os.path.exists(fp):
                cls._set(fp, **metadata)


class Metadata(MetadataInternal):
    """
    Public interface for metadata operations.
    Inherits from MetadataInternal and provides unified get/set methods
    that handle single/batch operations and tag modes automatically.
    """

    @classmethod
    def get(cls, file_path: Any, *keys: str, mode: str = "metadata") -> Any:
        """
        Unified get method for metadata and tags.

        Args:
            file_path: Single path (str) or list of paths.
            *keys: Metadata keys or Tag keys to retrieve.
            mode: 'metadata' (default) or 'tag'.

        Returns:
            - If file_path is list: NamedTupleContainer (batch operation)
            - If file_path is str and mode='metadata': Dict of metadata
            - If file_path is str and mode='tag':
                - If single key: str value
                - If multiple keys: Dict of {key: value}
        """
        # Handle batch operation
        if isinstance(file_path, (list, tuple)):
            if mode == "tag":
                # For batch tags, we'll need to implement a custom batch_get_tag logic
                # For now, let's assume batch_get is for metadata columns
                # We could extend batch_get to support tags if needed
                pass
            return cls._batch_get(file_path, *keys)

        # Handle single file operation
        if mode == "tag":
            if len(keys) == 1:
                return cls._get_tag(file_path, keys[0])
            else:
                return {k: cls._get_tag(file_path, k) for k in keys}

        # Default metadata get
        return cls._get(file_path, *keys)

    @classmethod
    def set(cls, file_path: Any, mode: str = "metadata", **kwargs) -> None:
        """
        Unified set method for metadata and tags.

        Args:
            file_path: Single path (str) or list of paths.
            mode: 'metadata' (default) or 'tag'.
            **kwargs: Key-value pairs to set.
        """
        # Handle batch operation
        if isinstance(file_path, (list, tuple)):
            if mode == "tag":
                for fp in file_path:
                    for k, v in kwargs.items():
                        cls._set_tag(fp, k, v)
            else:
                cls._batch_set(file_path, **kwargs)
            return

        # Handle single file operation
        if mode == "tag":
            for k, v in kwargs.items():
                cls._set_tag(file_path, k, v)
        else:
            cls._set(file_path, **kwargs)
