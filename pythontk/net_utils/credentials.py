import platform
import os

try:
    import win32cred
    import win32api
    import pywintypes
except ImportError:
    win32cred = None
    win32api = None
    pywintypes = None

try:
    import keyring
except ImportError:
    keyring = None


class Credentials:
    """
    Abstractions for OS-level secure credential storage.

    Priority Order:
    1. 'keyring' library (if installed) - Supports Windows/Mac/Linux(Gnome/KDE)
    2. Native Windows Credential Manager (via pywin32)
    3. Environment Variables (Fallback for CI/Headless)
    """

    @staticmethod
    def get_password(target_name: str) -> str:
        """
        Retrieve a password from the OS secure store or environment.

        Args:
            target_name (str): The name/address of the target (e.g., "192.168.1.5" or "dev-server").

        Returns:
            str: The password if found, else None.
        """
        cred = Credentials.get_credential(target_name)
        return cred.get("password") if cred else None

    @staticmethod
    def get_credential(target_name: str) -> dict | None:
        """
        Retrieve full credentials (username and password).
        """
        # 1. Try 'keyring' library if available
        if keyring:
            try:
                pwd = keyring.get_password("pythontk", target_name)
                if pwd:
                    # Keyring doesn't always store usernames tightly coupled,
                    # but we can try to fetch a separate entry or default it.
                    # For simple usage, we might just return the password.
                    # To keep API consistent, we try to finding a stored user or return 'unknown'.
                    return {"username": "keyring_user", "password": pwd}
            except Exception:
                pass

        system = platform.system()

        # 2. Native Windows
        if system == "Windows":
            result = Credentials._get_windows_creds(target_name)
            if result:
                return result

        # 3. Environment Variable Fallback (Linux Servers / CI)
        return Credentials._get_env_credential(target_name)

    @staticmethod
    def set_credential(
        target_name: str, username: str, password: str, persist: str = "local_machine"
    ) -> bool:
        """
        Save credentials to the OS secure store.
        Note: Environment variables cannot be set persistently this way.
        """
        # 1. Try 'keyring' library
        if keyring:
            try:
                keyring.set_password("pythontk", target_name, password)
                return True
            except Exception as e:
                print(f"Keyring failed: {e}")
                # Fall through to native methods

        system = platform.system()

        # 2. Native Windows
        if system == "Windows":
            return Credentials._set_windows_creds(
                target_name, username, password, persist
            )

        print(f"Warning: No secure storage backend available for {system}.")
        print("To support Linux secret storage, install the 'keyring' package:")
        print("  pip install keyring")
        return False

    @staticmethod
    def _get_env_credential(target: str) -> dict | None:
        """
        Fallback: Check environment variables.
        Name Mapping: "server_guac" -> "SERVER_GUAC_PASSWORD"
        """
        safe_target = (
            target.upper().replace(" ", "_").replace("-", "_").replace(".", "_")
        )

        # Try generic patterns
        keys_to_try = [
            f"{safe_target}_PASSWORD",
            f"{safe_target}_SECRET",
            f"{safe_target}_KEY",
        ]

        for key in keys_to_try:
            val = os.environ.get(key)
            if val:
                user = os.environ.get(f"{safe_target}_USER", "env_user")
                return {"username": user, "password": val}

        return None

    @staticmethod
    def _get_windows_creds(target: str) -> dict | None:
        """
        Fetch credentials from Windows Credential Manager using pywin32.
        """
        if not win32cred:
            return None

        try:
            # CRED_TYPE_GENERIC is usually 1
            creds = win32cred.CredRead(target, win32cred.CRED_TYPE_GENERIC)

            blob = creds.get("CredentialBlob", b"")

            try:
                password = blob.decode("utf-16-le")
            except UnicodeDecodeError:
                password = blob.decode("utf-8", errors="ignore")

            return {"username": creds.get("UserName"), "password": password}

        except pywintypes.error:
            # specific "Element not found" error usually
            return None
        except Exception as e:
            # General storage error
            print(f"Error accessing Windows Credentials: {e}")
            return None

    @staticmethod
    def _set_windows_creds(
        target: str, username: str, password: str, persist: str
    ) -> bool:
        """
        Save credentials to Windows Credential Manager.
        """
        if not win32cred:
            return False

        try:
            cred_type = win32cred.CRED_TYPE_GENERIC

            persist_map = {
                "session": win32cred.CRED_PERSIST_SESSION,
                "local_machine": win32cred.CRED_PERSIST_LOCAL_MACHINE,
                "enterprise": win32cred.CRED_PERSIST_ENTERPRISE,
            }
            persist_val = persist_map.get(
                persist.lower(), win32cred.CRED_PERSIST_LOCAL_MACHINE
            )

            # Windows generic credentials expect the password blob to be just the password directly.
            # While reading often gets UTF-16LE, writing raw bytes of UTF-16LE is the safest bidirectional approach.
            # Try plain string first to satisfy PyWin32 wrapper oddities
            blob = password

            cred_data = {
                "Type": cred_type,
                "TargetName": target,
                "UserName": username,
                "CredentialBlob": blob,
                "Persist": persist_val,
                "Comment": "Managed by pythontk",
            }

            win32cred.CredWrite(cred_data)
            return True

        except Exception as e:
            print(f"Error writing Windows Credentials: {e}")
            return False
