#!/usr/bin/env python3
"""Build the private Broadcom Vulkan driver from a clean Mesa source tree."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory


def export_source(source, destination):
    if (source / ".git").exists():
        with subprocess.Popen(["git", "-C", source, "archive", "HEAD"], stdout=subprocess.PIPE) as archive:
            with tarfile.open(fileobj=archive.stdout, mode="r|") as files:
                files.extractall(destination, filter="data")
            if archive.wait():
                raise RuntimeError("Could not export the Mesa checkout")
    else:
        shutil.copytree(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("patch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--meson", required=True)
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()

    with TemporaryDirectory(prefix="upscaler-mesa-") as temporary:
        source = Path(temporary) / "src"
        build = Path(temporary) / "build"
        export_source(args.source, source)
        subprocess.run(["git", "apply", args.patch.resolve()], cwd=source, check=True)
        subprocess.run([args.meson, "setup", build, source, "--buildtype=release",
                        "-Dvulkan-drivers=broadcom", "-Dgallium-drivers=[]", "-Dplatforms=[]",
                        "-Dvulkan-layers=[]", "-Dvideo-codecs=[]", "-Dtools=[]",
                        "-Dglx=disabled", "-Degl=disabled", "-Dgbm=disabled", "-Dopengl=false",
                        "-Dgles1=disabled", "-Dgles2=disabled", "-Dllvm=disabled",
                        "-Dzlib=disabled", "-Dzstd=enabled", "-Dshared-glapi=disabled",
                        "-Dbuild-tests=false"], check=True)
        subprocess.run([args.meson, "compile", "-C", build, "-j", str(args.jobs), "vulkan_broadcom"], check=True)

        driver = next(build.rglob(args.output.name))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staged = args.output.with_suffix(".tmp")
        shutil.copyfile(driver, staged)
        staged.replace(args.output)


if __name__ == "__main__":
    main()
