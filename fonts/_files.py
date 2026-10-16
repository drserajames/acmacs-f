"""Where the build put af's fonts: this module is installed beside them (fonts/CMakeLists.txt).

It is not in af/fonts in git on purpose. An editable install keeps af/fonts/__init__.py in the
source tree but puts what CMake installs (the fonts, and this file) in site-packages, as it does
the compiled optimiser; importing this module finds that directory in every kind of install.
"""

from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
