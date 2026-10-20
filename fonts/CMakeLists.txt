# The figure fonts: TeX Gyre Heros (Helvetica metrics) and TeX Gyre Cursor (Courier metrics),
# GUST Font License (an instance of the LPPL 1.3c; fonts/GUST-FONT-LICENSE.txt); and Courier
# Prime Regular (Courier metrics, a heavier stroke), SIL Open Font License 1.1
# (fonts/OFL-COURIER-PRIME.txt).
#
# They are not in git. Like alglib, the archive is fetched at a pinned version and checked
# against its SHA-256: a mismatch stops the build. Its Heros and Cursor files, their README and
# MANIFEST, the licence and a SOURCE.toml (where they came from, and each file's SHA-256) are
# installed into af/fonts, beside the package that finds them (af/fonts/__init__.py). Figures
# name a font by its file name there, so nothing depends on what a machine has installed.
# Offline builds point CMake at a local copy of the archive, and the hash is still checked:
#   pip install . -Ccmake.define.AF_FONTS_URL=/path/to/tex_gyre-TDS_distr-2_609-31_03_2026.zip

set(AF_FONTS_VERSION "2.609")
set(AF_FONTS_URL
    "https://www.gust.org.pl/projects/e-foundry/tex-gyre/whole/tex_gyre-TDS_distr-2_609-31_03_2026.zip"
    CACHE STRING "TeX Gyre TDS archive: URL or local path")
set(AF_FONTS_SHA256 "062dbab5a83b1409426dbd8af7b5cb554261c7b7ac9ea91f7ff339930ab68330")

include(FetchContent)
FetchContent_Declare(texgyre
  URL "${AF_FONTS_URL}"
  URL_HASH "SHA256=${AF_FONTS_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(texgyre)   # fonts only: there is nothing to build

set(af_fonts_otf "${texgyre_SOURCE_DIR}/fonts/opentype/public/tex-gyre")
set(af_fonts_doc "${texgyre_SOURCE_DIR}/doc/fonts/tex-gyre")
set(af_fonts_files "")
foreach(family heros cursor)
  foreach(style regular bold italic bolditalic)
    list(APPEND af_fonts_files "${af_fonts_otf}/texgyre${family}-${style}.otf")
  endforeach()
endforeach()
foreach(family Heros Cursor)
  list(APPEND af_fonts_files "${af_fonts_doc}/README-TeX-Gyre-${family}.txt"
                             "${af_fonts_doc}/MANIFEST-TeX-Gyre-${family}.txt")
endforeach()
list(APPEND af_fonts_files "${CMAKE_CURRENT_SOURCE_DIR}/GUST-FONT-LICENSE.txt")

# Courier Prime: the tree pages' substitution labels. At their size TeX Gyre Cursor's stems are
# 0.36 pt against the shipped pages' Courier at 0.60; Courier Prime's are 0.72, with Courier's
# advance width (R5 visual review, 19 Oct 2026). One file from Google Fonts' repository at a
# pinned commit, checked against its SHA-256 like the archive above. Offline:
#   pip install . -Ccmake.define.AF_COURIER_PRIME_URL=/path/to/CourierPrime-Regular.ttf
set(AF_COURIER_PRIME_VERSION "3.018")
set(AF_COURIER_PRIME_URL
    "https://github.com/google/fonts/raw/cc9ce5318f8f50a32695e38ef0be8b0e453c7304/ofl/courierprime/CourierPrime-Regular.ttf"
    CACHE STRING "Courier Prime Regular: URL or local path")
set(AF_COURIER_PRIME_SHA256 "72f793376f8e2841656bf21d77a5de010f2929bd6956a22ee848ad0c7eb978af")
set(af_courier_prime "${CMAKE_CURRENT_BINARY_DIR}/CourierPrime-Regular.ttf")
set(af_have "")
if(EXISTS "${af_courier_prime}")
  file(SHA256 "${af_courier_prime}" af_have)
endif()
if(NOT af_have STREQUAL AF_COURIER_PRIME_SHA256)
  if(EXISTS "${AF_COURIER_PRIME_URL}")
    file(COPY_FILE "${AF_COURIER_PRIME_URL}" "${af_courier_prime}")
  else()
    file(DOWNLOAD "${AF_COURIER_PRIME_URL}" "${af_courier_prime}" STATUS af_status
         EXPECTED_HASH "SHA256=${AF_COURIER_PRIME_SHA256}")
    list(GET af_status 0 af_code)
    if(NOT af_code EQUAL 0)
      message(FATAL_ERROR "af: Courier Prime download failed: ${af_status}")
    endif()
  endif()
  file(SHA256 "${af_courier_prime}" af_have)
  if(NOT af_have STREQUAL AF_COURIER_PRIME_SHA256)
    message(FATAL_ERROR "af: ${AF_COURIER_PRIME_URL} has SHA-256 ${af_have}, "
                        "not the pinned ${AF_COURIER_PRIME_SHA256}")
  endif()
endif()
list(APPEND af_fonts_files "${af_courier_prime}" "${CMAKE_CURRENT_SOURCE_DIR}/OFL-COURIER-PRIME.txt")

# SOURCE.toml: the archive, every installed file with its SHA-256, and Courier Prime's source.
set(af_fonts_source "# written by fonts/CMakeLists.txt at build time\n")
string(APPEND af_fonts_source "archive_url = \"${AF_FONTS_URL}\"\n")
string(APPEND af_fonts_source "archive_sha256 = \"${AF_FONTS_SHA256}\"\n")
string(APPEND af_fonts_source "version = \"${AF_FONTS_VERSION}\"\n\n[files]\n")
foreach(file IN LISTS af_fonts_files)
  if(NOT EXISTS "${file}")
    message(FATAL_ERROR "af: no font file ${file} (the TeX Gyre archive, or Courier Prime)")
  endif()
  file(SHA256 "${file}" af_file_sha256)
  get_filename_component(af_file_name "${file}" NAME)
  string(APPEND af_fonts_source "\"${af_file_name}\" = \"${af_file_sha256}\"\n")
endforeach()
string(APPEND af_fonts_source "\n[courier_prime]\n")
string(APPEND af_fonts_source "url = \"${AF_COURIER_PRIME_URL}\"\n")
string(APPEND af_fonts_source "sha256 = \"${AF_COURIER_PRIME_SHA256}\"\n")
string(APPEND af_fonts_source "version = \"${AF_COURIER_PRIME_VERSION}\"\n")
string(APPEND af_fonts_source "licence = \"OFL-COURIER-PRIME.txt\"\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/SOURCE.toml" "${af_fonts_source}")

install(FILES ${af_fonts_files} "${CMAKE_CURRENT_BINARY_DIR}/SOURCE.toml"
        "${CMAKE_CURRENT_SOURCE_DIR}/_files.py" DESTINATION af/fonts)
message(STATUS "af: TeX Gyre ${AF_FONTS_VERSION} fonts (Heros, Cursor) and Courier Prime "
               "${AF_COURIER_PRIME_VERSION} for af/fonts")
