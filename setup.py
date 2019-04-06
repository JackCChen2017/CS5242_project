# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Dependencies for Dataflow workers.'''

from setuptools import setup, find_packages
import os

NAME = 'trainer'
VERSION = '1.0'
REQUIRED_PACKAGES = []


REQUIRED_PACKAGES = [
	'falcon==1.2.0',
	'inflect==0.2.5',
	'audioread==2.1.5',
	'librosa==0.5.1',
	'matplotlib==2.0.2',
	'numpy==1.14.0',
	'scipy==1.0.0',
	'tqdm==4.11.2',
	'Unidecode==0.4.20',
	'pyaudio==0.2.11',
	'sounddevice==0.3.10',
	'lws',
	'keras'
]
setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='test'
)