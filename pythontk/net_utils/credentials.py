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
        system = platform.system()

        if system == "Windows":
            return Credentials._get_windows_creds(target_name)

        # Future implementation for other OSs
        # elif system == "Darwin": ...

        return None

    @staticmethod
    def _get_windows_creds(target: str) -> str:
        """
        Fetch credentials from Windows Credential Manager using pywin32.
        """
        if not win32cred:
            return None

        try:
            # CRED_TYPE_GENERIC is usually 1
            creds = win32cred.CredRead(target, win32cred.CRED_TYPE_GENERIC)
            # 'CredentialBlob' contains the password as bytes
            blob = creds.get("CredentialBlob", b"")

            # Windows usually stores these as UTF-16LE, but sometimes simple ASCII/UTF-8 depending on how it was saved.
            # However, standard generic creds saved by Windows UI are often manageable.
            # Attempt to decode. The 'check_logs.py' used utf-16-le.
            try:
                return blob.decode("utf-16-le")
            except UnicodeDecodeError:
                # Fallback
                return blob.decode("utf-8", errors="ignore")

        except pywintypes.error:
            # specific "Element not found" error usually
            return None
        except Exception as e:
            # General storage error
            print(f"Error accessing Windows Credentials: {e}")
            return None
