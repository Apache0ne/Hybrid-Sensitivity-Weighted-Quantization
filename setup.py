from setuptools import find_namespace_packages, find_packages, setup


packages = find_packages(where=".", include=["histogram", "histogram.*"])
packages += find_packages(where="ComfyUI-master", include=["comfy", "comfy.*"])
packages += find_namespace_packages(where="ComfyUI-master", include=["comfy.*"])
packages = sorted(set(packages))


setup(
    packages=packages,
    package_dir={"": ".", "comfy": "ComfyUI-master/comfy"},
    include_package_data=True,
    package_data={
        "comfy": ["**/*"],
        "histogram": ["*.py"],
    },
    py_modules=[
        "hswq_sd15_comfy_loader",
        "hswq_sd15_mapping",
        "quantize_sd15_hswq_v1",
    ],
)
