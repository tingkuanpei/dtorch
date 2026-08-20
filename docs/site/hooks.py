"""MkDocs build hook: make shared site assets available to both language sites.

The two language sites are separate MkDocs projects whose docs_dirs (docs/en,
docs/cn) do not contain the shared assets under docs/site. This hook registers
those assets as build files, so `extra_javascript` can reference them while a
single copy lives in the repository.

Note: docs/site/index.html (the root language-redirect page) is deliberately
not listed here — the deploy workflow copies it to the site root, it must not
be copied into the language sites.
"""

from pathlib import Path

from mkdocs.structure.files import File

# Shared assets, relative to docs/site/ (and to the built site).
ASSETS = ["assets/js/lang_switch.js"]


def on_files(files, config, **kwargs):
    """Add the shared assets as build files, mirrored at the same paths."""
    site_root = Path(config["config_file_path"]).parent / "docs" / "site"
    for rel in ASSETS:
        files.append(
            File(
                path=rel,
                src_dir=str(site_root),
                dest_dir=config["site_dir"],
                use_directory_urls=config["use_directory_urls"],
            )
        )
