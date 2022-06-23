# Copyright 2022 Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os, time

class Logger():
    def __init__(self, file='logs/log.txt', clear_file=False, add_time=False, time_format=None, print_to_stdout=False):
        d = os.path.dirname(file)
        if d:
            os.makedirs(d, exist_ok=True)
        self.file = file
        if clear_file and os.path.exists(file):
            os.remove(file)
        self.add_time = add_time
        if time_format is None:
            time_format = '%Y-%m-%d %H:%M:%S'
        self.time_format = time_format
        self.print_to_stdout = print_to_stdout
        
    def log(self, text, avoid_time=False, suppress_stdout=False):
        if self.add_time and not avoid_time:
            text = text.split('\n')
            time_text = f'[{time.strftime(self.time_format)}]\t'
            for i in range(len(text)):
                if text[i].strip(): 
                    text[i] = time_text + text[i]
            text = '\n'.join(text)
            
        with open(self.file, 'a+') as file:
            print(text, file=file)
        
        if self.print_to_stdout and not suppress_stdout:
            print(text)