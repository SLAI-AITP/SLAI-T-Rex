# coding=utf-8
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Setup for pip package."""

import sys
import os
import time
import multiprocessing
import subprocess
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by GTSSpeed.")

OPS_DESCRIPTION = 'GTSSpeed for Operators of Ascend'
OPS_BUILD_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
if os.getenv('ENV_PIPELINE_STARTTIME') is not None:
    OPS_BUILD_TIME = os.getenv('ENV_PIPELINE_STARTTIME')
OPS_VERSION = "1.0."+OPS_BUILD_TIME
OPS_AUTHOR = 'GTSOps'
OPS_LONG_DESCRIPTION = 'GTSSpeed for Operators of Ascend'
OPS_URL = ''
OPS_KEYWORDS = 'Ascend, Operator, GTSpeed'
OPS_LICENSE = 'See CANN-S'
PYPI_PATTERN = "" if os.getenv('PYPI_PATTERN') is None else os.getenv('PYPI_PATTERN')
OPS_PACKAGE_NAME = PYPI_PATTERN+'gtsspeed'
OPS_CONTACT_NAMES = 'AIComputingLab'
ABI_FLAG = False

###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


def readlines(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]

try:
    import torch
    import torch_npu
    torch_npu.npu.is_available()
    ABI_FLAG=torch.compiled_with_cxx11_abi()
except Exception as exp:
    raise ValueError('Can not find torch_npu: torch_npu not installed or failed') from exp

install_requires = readlines("requirements.txt")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def __init__(self, args):
        super().__init__(args)
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError as osexp:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions)) from osexp

        self.pytorch_dir = os.path.dirname(torch.__file__)
        self.python_exe = subprocess.check_output(["which", "python"]).decode().strip()

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        sitedir = os.path.abspath(os.path.join(self.pytorch_dir, ".."))
        print("====extdir ", extdir)
        print("====sitedir ", sitedir)
        pkg_type = os.environ.get("PKG_TYPE")
        cmake_args = ["-DCMAKE_BUILD_TYPE=" + ("Release" if pkg_type is not None and pkg_type == "CANN-S" else "Debug"),
                      "-B", extdir+"/build",
                      "-DTORCH_HIGHER_THAN_PTA6=ON",
                      "-DUSE_CXX11_ABI=" + ("ON" if ABI_FLAG == True else "OFF"),
                      "-DSITE_PACKAGES_PATH={}".format(sitedir),
                      "-DLIBRARY_OUTPUT_PATH={}/".format(extdir)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(extdir))
        self.spawn(["cmake", ext.sourcedir] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", extdir + "/build", "--", "-j{}".format(multiprocessing.cpu_count())])
        os.chdir(cwd)

setuptools.setup(
    name=OPS_PACKAGE_NAME,
    version=OPS_VERSION,
    author_email='None',
    description=OPS_DESCRIPTION,
    long_description="gtsspeed",
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url=OPS_URL,
    author=OPS_CONTACT_NAMES,
    maintainer=OPS_CONTACT_NAMES,
    # The licence under which the project is released
    license=OPS_LICENSE,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    keywords=OPS_KEYWORDS,
    ext_modules=[CMakeExtension("gtsops", "gtsspeed/ops")],
    cmdclass={"build_ext": CMakeBuild, }
)
