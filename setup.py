import setuptools
import pythontk as ptk


long_description = ptk.get_file_contents("docs/README.md")
description = ptk.get_text_between_delimiters(
    long_description,
    "<!-- short_description_start -->",
    "<!-- short_description_end -->",
    as_string=True,
)

setuptools.setup(
    name=ptk.__package__,
    version=ptk.__version__,
    author="Ryan Simpson",
    author_email="m3trik@outlook.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/m3trik/{ptk.__package__}",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    data_files=ptk.get_dir_contents(
        ptk.__package__, "filepath", exc_files=["*.py", "*.pyc", "*.json"]
    ),  # ie. ('uitk/ui/0', ['uitk/ui/0/init.ui']),
)

# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
