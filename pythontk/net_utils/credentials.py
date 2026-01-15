import platform

try:
    import win32cred
    import win32api
    import pywintypes
except ImportError:
    win32cred = None
    win32api = None
    pywintypes = None


class Credentials:
    """
    Abstractions for OS-level secure credential storage.
    Currently supports Windows Credential Manager via pywin32.
    """

    @staticmethod
    def get_password(target_name: str) -> str:
        """
        Retrieve a password from the OS secure store.

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
        Retrieve full credentials (username and password) from the OS secure store.

        Args:
            target_name (str): The name/address of the target.

        Returns:
            dict: Dictionary with 'username' and 'password' keys, or None if not found.
        """
        system = platform.system()

        if system == "Windows":
            return Credentials._get_windows_creds(target_name)

        return None

    @staticmethod
    def set_credential(
        target_name: str, username: str, password: str, persist: str = "local_machine"
    ) -> bool:
        """
        Save credentials to the OS secure store.

        Args:
            target_name (str): The name/address of the target.
            username (str): The username.
            password (str): The password.
            persist (str): Persistence level ('session', 'local_machine', 'enterprise').
                           Defaults to 'local_machine'.

        Returns:
            bool: True if successful, False otherwise.
        """
        system = platform.system()

        if system == "Windows":
            return Credentials._set_windows_creds(
                target_name, username, password, persist
            )

        return False

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
